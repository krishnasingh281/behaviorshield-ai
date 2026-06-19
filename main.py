import os
import json
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from src.features import build_feature_vector
from src.model import AdvancedBehaviorRiskEngine

app = FastAPI(title="BehaviorShield Dynamic Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500",
                   "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROFILE_DIR = "./user_data"
os.makedirs(PROFILE_DIR, exist_ok=True)

# LRU-style cap: evict oldest when registry exceeds 100 users
USER_MODEL_REGISTRY = {}
MAX_REGISTRY_SIZE = 100


def augment_real_data(base_vector, num_samples=150):
    """
    Takes 1 real typing sample and generates 150 realistic variations.
    Swipe velocity is excluded from augmentation — it's unreliable on
    desktop (mouse ≠ touch), so we give it a fixed neutral value and
    let the model treat it as low-weight.
    """
    np.random.seed(42)
    dataset = []
    for _ in range(num_samples):
        dataset.append([
            max(1.0,  base_vector[0] + np.random.normal(0, 8)),   # mean_dwell
            max(0.1,  base_vector[1] + np.random.normal(0, 2)),   # std_dwell
            max(1.0,  base_vector[2] + np.random.normal(0, 15)),  # mean_flight
            max(0.01, base_vector[3] + np.random.normal(0, 0.01)) # mean_swipe_velocity
        ])
    return dataset


def evict_if_needed():
    """Remove oldest entry if registry is full."""
    if len(USER_MODEL_REGISTRY) >= MAX_REGISTRY_SIZE:
        oldest_key = next(iter(USER_MODEL_REGISTRY))
        del USER_MODEL_REGISTRY[oldest_key]
        print(f"[REGISTRY] Evicted {oldest_key} from RAM cache.")


@app.get("/api/health")
async def health():
    return {"status": "ok", "enrolled_users": len(USER_MODEL_REGISTRY)}


@app.get("/api/status/{user_id}")
async def check_enrolled(user_id: str):
    """Check if a user profile exists before opening WebSocket."""
    enrolled = os.path.exists(f"{PROFILE_DIR}/{user_id}.json")
    return {"user_id": user_id, "enrolled": enrolled}


@app.post("/api/enroll/{user_id}")
async def enroll_new_user(user_id: str, request: Request):
    """
    Enrollment endpoint.
    Receives raw events, extracts features, augments, trains, saves.
    """
    payload = await request.json()
    events  = payload.get("events", [])

    if len(events) < 4:
        return {"status": "ERROR", "message": "Not enough events to build a profile."}

    feature_dict = build_feature_vector({"events": events})

    mean_dwell  = feature_dict.get("mean_dwell", 80.0)
    std_dwell   = feature_dict.get("std_dwell", 10.0)
    mean_flight = feature_dict.get("mean_flight", 80.0)

    # Desktop swipe is near-zero and unreliable — use a fixed neutral value
    # so it doesn't poison the baseline. The model weights it at only 15%.
    swipe_velocity = 0.05

    base_vector = [mean_dwell, std_dwell, mean_flight, swipe_velocity]
    training_matrix = augment_real_data(base_vector)

    file_path = f"{PROFILE_DIR}/{user_id}.json"
    with open(file_path, "w") as f:
        json.dump(training_matrix, f)

    evict_if_needed()
    engine = AdvancedBehaviorRiskEngine()
    engine.train_initial_baseline(training_matrix)
    USER_MODEL_REGISTRY[user_id] = engine

    print(f"[ENROLLMENT] Profile saved for {user_id}.")
    print(f"  dwell={mean_dwell:.1f}ms  flight={mean_flight:.1f}ms  std={std_dwell:.1f}")
    return {"status": "SUCCESS", "message": "Behavioral profile registered."}


def load_engine_from_disk(user_id: str):
    """Load a trained engine from disk into RAM."""
    file_path = f"{PROFILE_DIR}/{user_id}.json"
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        training_matrix = json.load(f)
    engine = AdvancedBehaviorRiskEngine()
    engine.train_initial_baseline(training_matrix)
    evict_if_needed()
    USER_MODEL_REGISTRY[user_id] = engine
    print(f"[SYSTEM] Loaded {user_id} from disk into RAM.")
    return engine


@app.websocket("/ws/auth/{user_id}")
async def continuous_authentication_stream(websocket: WebSocket, user_id: str):
    await websocket.accept()

    # Load engine from RAM or disk
    risk_engine = USER_MODEL_REGISTRY.get(user_id) or load_engine_from_disk(user_id)

    if not risk_engine:
        await websocket.send_json({
            "error": "No profile found. Please enroll first.",
            "action": "ENROLL_REQUIRED"
        })
        await websocket.close(code=4001)
        return

    await websocket.send_json({
        "status": "CONNECTED",
        "message": f"Behavioral engine active for {user_id}."
    })

    try:
        while True:
            raw_data = await websocket.receive_text()
            payload  = json.loads(raw_data)
            events   = payload.get("events", [])
            context  = payload.get("context", {
                "km_from_last_login": 0,
                "hours_since_last_login": 1,
                "is_trusted_device": True
            })

            if len(events) < 4:
                # Not enough data yet — send a neutral holding frame
                await websocket.send_json({
                    "status": "WAITING",
                    "message": "Not enough events yet. Keep typing.",
                    "risk_score": 0,
                    "verdict": "CLEAR",
                    "metrics_computed": {}
                })
                continue

            feature_dict   = build_feature_vector({"events": events})
            ai_input_vector = [
                feature_dict.get("mean_dwell", 0.0),
                feature_dict.get("std_dwell",  0.0),
                feature_dict.get("mean_flight", 0.0),
                0.05  # fixed neutral swipe — desktop has no reliable touch data
            ]

            verdict_report = risk_engine.evaluate_session(
                ai_input_vector, context_data=context
            )

            response_frame = {
                "status":          "PROCESSED",
                "risk_score":      verdict_report["risk_score"],
                "verdict":         verdict_report["verdict"],
                "behavior_confidence": verdict_report["behavior_confidence"],
                "context_risk":    verdict_report["context_risk"],
                "z_scores":        verdict_report["z_scores"],
                "metrics_computed": feature_dict
            }
            await websocket.send_json(response_frame)

            if verdict_report["verdict"] == "CRITICAL_ANOMALY_TERMINATE_SESSION":
                await websocket.send_json({
                    "action": "FORCE_LOGOUT",
                    "reason": "Behavioral signature does not match enrolled profile."
                })
                await websocket.close(code=4403)
                break

    except WebSocketDisconnect:
        print(f"[WebSocket] {user_id} disconnected.")