import numpy as np
from src.model import AdvancedBehaviorRiskEngine

def generate_profile_sample(style):
    """Generates explicit static testing snapshots based on edge cases."""
    if style == "clean_baseline":
        return [85.0, 6.0, 70.0, 1.8]  # Quick, precise, regular user
    elif style == "elderly_drift":
        return [115.0, 14.0, 105.0, 1.4] # Drifting slightly slower, minor variance
    elif style == "under_duress":
        return [60.0, 38.0, 50.0, 3.2]   # Highly erratic speeds, rapid sharp movements
    elif style == "attacker":
        return [175.0, 28.0, 210.0, 0.5] # Completely mismatched cadence signature
    return [0.0, 0.0, 0.0, 0.0]

if __name__ == "__main__":
    print("=== Training Sophisticated Context-Aware AI Core ===")
    
    # 1. Setup simulated background database of genuine interactions
    np.random.seed(42)
    base_sample = generate_profile_sample("clean_baseline")
    training_set = [
        [
            base_sample[0] + np.random.normal(0, 5),
            base_sample[1] + np.random.normal(0, 1),
            base_sample[2] + np.random.normal(0, 6),
            base_sample[3] + np.random.normal(0, 0.1)
        ] for _ in range(200)
    ]
    
    engine = AdvancedBehaviorRiskEngine()
    engine.train_initial_baseline(training_set)
    print("Initial personal baseline model built successfully.")

    # --- EDGE CASE SIMULATION 1: Genuine User is Traveling ---
    print("\n--- Edge Case 1: Genuine User traveling (New Location / Flight Drift) ---")
    travel_context = {"km_from_last_login": 1200, "hours_since_last_login": 2, "is_trusted_device": True}
    res_1 = engine.evaluate_session(generate_profile_sample("clean_baseline"), context_data=travel_context)
    print(f"  Calculated Risk: {res_1['risk_score']} | Context Threat: {res_1['context_risk']}")
    print(f"  Verdict Action: {res_1['verdict']} (Success: System avoided false lockout)")

    # --- EDGE CASE SIMULATION 2: Gradual Physical Drift ---
    print("\n--- Edge Case 2: Legitimate User slow performance (Age / Fatigue / Injury) ---")
    normal_context = {"km_from_last_login": 5, "hours_since_last_login": 24, "is_trusted_device": True}
    res_2 = engine.evaluate_session(generate_profile_sample("elderly_drift"), context_data=normal_context)
    print(f"  Calculated Risk: {res_2['risk_score']} | Behavior Confidence: {res_2['behavior_confidence']}")
    print(f"  Verdict Action: {res_2['verdict']} (Success: Step-up authentication checks user safely)")

    # --- EDGE CASE SIMULATION 3: Attack Under Duress ---
    print("\n--- Edge Case 3: Legitimate Session Under Coercion / Hostile Duress ---")
    res_3 = engine.evaluate_session(generate_profile_sample("under_duress"), context_data=normal_context)
    print(f"  Calculated Risk: {res_3['risk_score']} | Behavior Confidence: {res_3['behavior_confidence']}")
    print(f"  Verdict Action: {res_3['verdict']} (Success: Captured erratic biological response)")

    # --- EDGE CASE SIMULATION 4: Direct Malicious Attacker ---
    print("\n--- Edge Case 4: Straightforward Account Takeover (Attacker Access) ---")
    attacker_context = {"km_from_last_login": 850, "hours_since_last_login": 1, "is_trusted_device": False}
    res_4 = engine.evaluate_session(generate_profile_sample("attacker"), context_data=attacker_context)
    print(f"  Calculated Risk: {res_4['risk_score']} | Verdict Action: {res_4['verdict']}")