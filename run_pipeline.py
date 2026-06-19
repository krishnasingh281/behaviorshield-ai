from src.features import build_feature_vector

# Simulated raw payload tracking a fast pin entry followed by a screen swipe
mock_client_payload = {
    "session_id": "test_session_001",
    "events": [
        {"type": "key_down", "target": "num_4", "timestamp": 1000},
        {"type": "key_up", "target": "num_4", "timestamp": 1080}, # Dwell: 80ms
        
        {"type": "key_down", "target": "num_2", "timestamp": 1150}, # Flight: 70ms
        {"type": "key_up", "target": "num_2", "timestamp": 1240}, # Dwell: 90ms
        
        {"type": "touch_start", "target": "feed", "timestamp": 1500, "x": 100, "y": 500},
        {"type": "touch_end", "target": "feed", "timestamp": 1700, "x": 100, "y": 200} # Swipe delta
    ]
}

if __name__ == "__main__":
    print("--- Executing Phase 1 Pipeline Verification ---")
    
    # Process raw interaction data
    vector = build_feature_vector(mock_client_payload)
    
    print("\nGenerated Feature Vector Structure:")
    for key, value in vector.items():
        print(f"  {key}: {value}")