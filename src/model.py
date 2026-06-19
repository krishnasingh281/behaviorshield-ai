import numpy as np

class AdvancedBehaviorRiskEngine:
    """
    A self-contained behavioral risk engine that uses a statistical
    Isolation-Forest-style anomaly scorer + context-aware weighting.
    No sklearn required — runs on pure numpy.
    """

    def __init__(self):
        # Trained baseline state
        self.baseline_mean = None       # np.array shape (4,)
        self.baseline_std = None        # np.array shape (4,)
        self.is_trained = False

        # Adaptive learning: running history of recent sessions
        self.session_history = []       # list of np.arrays shape (4,)
        self.max_history = 50           # rolling window size

        # Feature names for logging / debugging
        self.feature_names = [
            "mean_dwell",
            "std_dwell",
            "mean_flight",
            "mean_swipe_velocity"
        ]

    # ------------------------------------------------------------------ #
    #  TRAINING                                                            #
    # ------------------------------------------------------------------ #

    def train_initial_baseline(self, training_matrix):
        """
        Accepts the augmented training matrix (list of 4-feature vectors).
        Computes and stores the mean + std baseline profile.
        """
        data = np.array(training_matrix, dtype=float)   # shape: (N, 4)

        self.baseline_mean = np.mean(data, axis=0)
        self.baseline_std  = np.std(data,  axis=0)

        # Prevent division-by-zero for features with zero variance
        self.baseline_std = np.where(
            self.baseline_std < 1e-6,
            1.0,
            self.baseline_std
        )

        self.is_trained = True
        print(f"[MODEL] Baseline trained. Mean: {self.baseline_mean.round(2)}")
        print(f"[MODEL] Std dev:  {self.baseline_std.round(2)}")

    # ------------------------------------------------------------------ #
    #  SCORING                                                             #
    # ------------------------------------------------------------------ #

    def _compute_behavior_confidence(self, input_vector):
        """
        Computes per-feature Z-scores against the stored baseline.
        Returns:
          - z_scores         : how many std-devs each feature deviates
          - behavior_confidence : 0-100 score (100 = perfect match)
        """
        vec = np.array(input_vector, dtype=float)

        # Z-score: how many standard deviations away from baseline mean
        z_scores = np.abs((vec - self.baseline_mean) / self.baseline_std)

        # Convert Z to a 0-1 confidence per feature using a soft cap at z=3
        # z=0 → conf=1.0 (perfect), z=3 → conf=0.0 (major anomaly)
        per_feature_conf = np.clip(1.0 - (z_scores / 3.0), 0.0, 1.0)

        # Weighted average — typing features (0,1,2) are more reliable than swipe (3)
        weights = np.array([0.35, 0.20, 0.30, 0.15])
        behavior_confidence = float(np.dot(per_feature_conf, weights) * 100)

        return z_scores, behavior_confidence

    def _compute_context_risk(self, context_data):
        """
        Evaluates contextual signals that can legitimately explain anomalies
        (e.g. traveling) or amplify suspicion (e.g. unknown device + new city).

        Returns:
          - context_risk  : 0-100 additive risk score
          - context_note  : human-readable explanation
        """
        context_risk = 0.0
        notes = []

        km       = context_data.get("km_from_last_login", 0)
        hours    = context_data.get("hours_since_last_login", 1)
        trusted  = context_data.get("is_trusted_device", True)

        # --- Geolocation delta ---
        if km > 1000:
            # Very far away — strong signal, but could be travel
            context_risk += 30
            notes.append(f"location shift {km}km")
        elif km > 200:
            context_risk += 15
            notes.append(f"location shift {km}km")
        elif km > 50:
            context_risk += 5

        # --- Untrusted device (unrecognised fingerprint) ---
        if not trusted:
            context_risk += 35
            notes.append("unrecognised device")

        # --- Time anomaly: very quick re-login after a long gap ---
        # A real user traveling 1000km in 1 hour is physically impossible
        if km > 500 and hours < 2:
            context_risk += 25
            notes.append("impossible travel speed")

        context_note = ", ".join(notes) if notes else "no contextual anomalies"
        return min(context_risk, 100.0), context_note

    # ------------------------------------------------------------------ #
    #  ADAPTIVE LEARNING                                                   #
    # ------------------------------------------------------------------ #

    def _adapt_baseline(self, input_vector, behavior_confidence):
        """
        If the session looks legitimate (high confidence), slowly shift
        the baseline toward this new observation — handles natural drift
        like aging, injury, or device changes.
        Only adapts if confidence > 70 (safe zone).
        """
        if behavior_confidence < 70:
            return  # Don't learn from suspicious sessions

        self.session_history.append(np.array(input_vector, dtype=float))
        if len(self.session_history) > self.max_history:
            self.session_history.pop(0)

        if len(self.session_history) >= 10:
            recent = np.array(self.session_history)
            # Exponential blend: 80% old baseline, 20% recent sessions
            self.baseline_mean = 0.80 * self.baseline_mean + 0.20 * np.mean(recent, axis=0)
            self.baseline_std  = 0.80 * self.baseline_std  + 0.20 * np.std(recent,  axis=0)
            self.baseline_std  = np.where(self.baseline_std < 1e-6, 1.0, self.baseline_std)

    # ------------------------------------------------------------------ #
    #  VERDICT ENGINE                                                      #
    # ------------------------------------------------------------------ #

    def _compute_final_risk(self, behavior_confidence, context_risk):
        """
        Combines behavioral + contextual signals into a final 0-100 risk score.

        Design principles:
          - High context risk can amplify a moderate behavior anomaly
          - But if behavior is a perfect match, context alone won't block (travel use case)
          - If behavior is terrible AND context is bad → hard block
        """
        behavior_risk = 100.0 - behavior_confidence  # invert confidence to risk

        # Context risk is an amplifier, not an override
        # When behavior is clean (low behavior_risk), context has less weight
        blend_weight = 0.65  # behavior gets 65% weight
        raw_score = (blend_weight * behavior_risk) + ((1 - blend_weight) * context_risk)

        # Amplification: if both are bad simultaneously, push score up harder
        if behavior_risk > 60 and context_risk > 40:
            raw_score = min(raw_score * 1.25, 100.0)

        return min(raw_score, 100.0)

    def _determine_verdict(self, risk_score, context_note):
        """
        Maps the final risk score to an actionable verdict.

        Thresholds:
          0–35   → CLEAR         (full access)
          36–59  → STEP_UP_AUTH  (soft OTP challenge)
          60–79  → LIMIT_ACCESS  (read-only mode, flag for review)
          80–100 → CRITICAL_ANOMALY_TERMINATE_SESSION
        """
        if risk_score < 36:
            return "CLEAR"
        elif risk_score < 60:
            return "STEP_UP_AUTH"
        elif risk_score < 80:
            return "LIMIT_ACCESS"
        else:
            return "CRITICAL_ANOMALY_TERMINATE_SESSION"

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                          #
    # ------------------------------------------------------------------ #

    def evaluate_session(self, input_vector, context_data=None):
        """
        Main entry point. Call once per scoring window (every 2 seconds).

        Args:
            input_vector  : list/array of 4 floats
                            [mean_dwell, std_dwell, mean_flight, mean_swipe_velocity]
            context_data  : dict with optional keys:
                            km_from_last_login    (int/float)
                            hours_since_last_login (int/float)
                            is_trusted_device      (bool)

        Returns:
            dict with:
                risk_score          : float 0–100
                verdict             : str (action to take)
                behavior_confidence : float 0–100
                context_risk        : float 0–100
                context_note        : str
                z_scores            : dict {feature_name: z_value}
        """
        if not self.is_trained:
            raise RuntimeError(
                "Engine not trained. Call train_initial_baseline() first."
            )

        if context_data is None:
            context_data = {
                "km_from_last_login": 0,
                "hours_since_last_login": 1,
                "is_trusted_device": True
            }

        # 1. Behavioral scoring
        z_scores, behavior_confidence = self._compute_behavior_confidence(input_vector)

        # 2. Contextual scoring
        context_risk, context_note = self._compute_context_risk(context_data)

        # 3. Combined risk
        risk_score = self._compute_final_risk(behavior_confidence, context_risk)

        # 4. Verdict
        verdict = self._determine_verdict(risk_score, context_note)

        # 5. Adaptive learning (only on safe sessions)
        self._adapt_baseline(input_vector, behavior_confidence)

        return {
            "risk_score":           round(risk_score, 2),
            "verdict":              verdict,
            "behavior_confidence":  round(behavior_confidence, 2),
            "context_risk":         round(context_risk, 2),
            "context_note":         context_note,
            "z_scores": {
                name: round(float(z), 3)
                for name, z in zip(self.feature_names, z_scores)
            }
        }