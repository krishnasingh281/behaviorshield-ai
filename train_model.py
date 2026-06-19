import numpy as np
from src.model import BehaviorAnomalyDetector

def generate_synthetic_profile(num_samples, profile_type="genuine"):
    """
    Generates synthetic feature vector sets mimicking distinct user behaviors.
    """
    np.random.seed(42)
    dataset = []
    
    for _ in range(num_samples):
        if profile_type == "genuine":
            # Genuine user: fast, consistent typing; quick cursor movements
            mean_dwell = np.random.normal(85, 8)       # Mean ~85ms, low variance
            std_dwell = np.random.normal(6, 1.5)
            mean_flight = np.random.normal(70, 10)     # Mean ~70ms
            mean_swipe = np.random.normal(1.8, 0.2)    # Smooth mouse speed
        else:
            # Attacker: slower, hesitant typing; erratic/clumsy cursor movements
            mean_dwell = np.random.normal(160, 35)     # Way slower, high variance
            std_dwell = np.random.normal(25, 8)
            mean_flight = np.random.normal(190, 45)    # Hesitant gaps
            mean_swipe = np.random.normal(0.6, 0.4)    # Jagged mouse velocity
            
        dataset.append([mean_dwell, std_dwell, mean_flight, mean_swipe])
        
    return dataset

if __name__ == "__main__":
    print("--- Phase 2: Starting Model Training Pipeline ---")
    
    # 1. Generate normal training baseline for the genuine user
    print("Generating 200 sessions of genuine behavioral telemetry data...")
    genuine_train_data = generate_synthetic_profile(200, profile_type="genuine")
    
    # 2. Initialize and train our detector
    detector = BehaviorAnomalyDetector(contamination=0.03)
    detector.train(genuine_train_data)
    print("Model successfully trained on user's behavioral rhythm.")
    
    # 3. Simulate and test a new legitimate session from the genuine user
    print("\n--- Live Test 1: Legitimate User Access Attempt ---")
    genuine_test_sample = generate_synthetic_profile(1, profile_type="genuine")[0]
    trust_score, is_flagged = detector.compute_anomaly_score(genuine_test_sample)
    print(f"  Calculated Identity Trust Score: {trust_score}%")
    print(f"  Security Flag Triggered: {is_flagged} (Access Granted)")
    
    # 4. Simulate and test an unauthorized Account Takeover (ATO) attempt
    print("\n--- Live Test 2: Attacker Account Takeover Attempt ---")
    attacker_test_sample = generate_synthetic_profile(1, profile_type="attacker")[0]
    trust_score, is_flagged = detector.compute_anomaly_score(attacker_test_sample)
    print(f"  Calculated Identity Trust Score: {trust_score}%")
    print(f"  Security Flag Triggered: {is_flagged} (Session Terminated / Step-Up Challenge Required)")