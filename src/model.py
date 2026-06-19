import numpy as np
from sklearn.ensemble import IsolationForest

class AdvancedBehaviorRiskEngine:
    def __init__(self, contamination=0.04):
        """
        Advanced continuous authentication engine that combines unsupervised 
        behavior profiling with context risk signals.
        """
        self.behavior_model = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=42
        )
        self.is_trained = False
        self.historical_baseline_matrix = []

    def train_initial_baseline(self, feature_matrices):
        """
        Trains the initial profile using the genuine user's clean samples.
        """
        X = np.array(feature_matrices)
        self.historical_baseline_matrix = list(feature_matrices)
        self.behavior_model.fit(X)
        self.is_trained = True

    def _calculate_context_risk(self, context_data):
        """
        Evaluates risk based on environmental indicators.
        Returns a score between 0.0 (Safe) and 1.0 (Highly Suspicious).
        """
        if not context_data:
            return 0.0
            
        risk_score = 0.0
        
        # 1. Geolocation Speed Check (e.g., Impossible travel speeds)
        # If km_from_last_login is high but time elapsed is small
        km_travelled = context_data.get("km_from_last_login", 0)
        hours_elapsed = context_data.get("hours_since_last_login", 1)
        
        if hours_elapsed > 0:
            implied_speed = km_travelled / hours_elapsed
            if implied_speed > 900:  # Faster than a commercial airliner
                risk_score += 0.6
            elif implied_speed > 120: # Faster than standard highway transit
                risk_score += 0.3
                
        # 2. Device Familiarity Check
        if not context_data.get("is_trusted_device", True):
            risk_score += 0.25
            
        return min(risk_score, 1.0)

    def evaluate_session(self, live_features, context_data=None):
        """
        Blends behavioral anomaly scores and context risk indicators to 
        produce an adaptive, resilient security verdict.
        """
        if not self.is_trained:
            raise ValueError("AI Engine has not been trained yet.")

        # 1. Extract Behavioral Raw Scores (-0.5 to 0.5)
        X_test = np.array([live_features])
        raw_behavior_score = self.behavior_model.decision_function(X_test)[0]
        
        # Normalize behavioral anomaly score to a 0.0 (Abnormal) to 1.0 (Normal) scale
        # Typically normal samples sit above 0.0; anomalies dip below 0.0
        behavior_confidence = 1.0 / (1.0 + np.exp(-12 * raw_behavior_score))

        # 2. Extract Contextual Risk Factor
        context_risk = self._calculate_context_risk(context_data)

        # 3. Blended Adaptive Risk Scoring Formulation
        # If behavior matches perfectly (confidence is high), suppress false alarms from travel.
        # If behavior looks like an attacker, any context abnormality accelerates account lockout.
        final_risk_score = (1.0 - behavior_confidence) * 0.7 + (context_risk * 0.3)
        
        # 4. Generate Security Verdict
        if final_risk_score > 0.65:
            verdict = "CRITICAL_ANOMALY_TERMINATE_SESSION"
        elif final_risk_score > 0.40:
            verdict = "SUSPICIOUS_STEP_UP_CHALLENGE"
        else:
            verdict = "ACCESS_ALLOWED"
            # EDGE CASE FIX: If legitimate user behavior is slowly drifting (aging/injury),
            # safely incorporate this successful session to evolve the baseline over time.
            if behavior_confidence > 0.60:
                self._adapt_baseline(live_features)

        return {
            "risk_score": round(float(final_risk_score), 3),
            "behavior_confidence": round(float(behavior_confidence), 3),
            "context_risk": round(float(context_risk), 3),
            "verdict": verdict
        }

    def _adapt_baseline(self, new_valid_sample):
        """
        Maintains a rolling historical baseline to support gradual physical profile shifts.
        """
        self.historical_baseline_matrix.append(new_valid_sample)
        # Keep only the last 500 records to prevent memory bloat and structural bias
        if len(self.historical_baseline_matrix) > 500:
            self.historical_baseline_matrix.pop(0)
            
        # Re-fit the model on the updated user data footprint
        self.behavior_model.fit(np.array(self.historical_baseline_matrix))