import numpy as np
from sklearn.decomposition import PCA
try:
    from sklearn.ensemble._iforest import IsolationForest
except ImportError:
    from sklearn.ensemble import IsolationForest

class BehaviorAnomalyDetector:
    def __init__(self, contamination=0.05):
        """
        Initializes the unsupervised Isolation Forest model.
        Contamination represents the expected proportion of outliers in the data.
        """
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42
        )
        self.is_trained = False

    def train(self, feature_matrices):
        """
        Trains the model on a clean matrix of the genuine user's behavioral features.
        Expects a 2D numpy array or list of lists: [[mean_dwell, std_dwell, mean_flight, velocity], ...]
        """
        X = np.array(feature_matrices)
        self.model.fit(X)
        self.is_trained = True

    def compute_anomaly_score(self, single_feature_vector):
        """
        Scores a live web feature vector.
        Returns:
            - trust_score (float): A value between 0 and 100 (higher means more genuine).
            - is_anomaly (bool): True if flagged as malicious/abnormal.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before running inference.")
            
        # Format the single flat vector row for scikit-learn prediction
        X_test = np.array([single_feature_vector])
        
        # decision_function returns raw scores (negative values are anomalies)
        raw_score = self.model.decision_function(X_test)[0]
        
        # Convert prediction (-1 for anomaly, 1 for normal) into a boolean flag
        prediction = self.model.predict(X_test)[0]
        is_anomaly = True if prediction == -1 else False
        
        # Map the raw score into a consumer-friendly 0-100 trust score metric
        # Typically raw scores sit between -0.5 and 0.5
        trust_score = min(max(int((raw_score + 0.5) * 100), 0), 100)
        
        return trust_score, is_anomaly