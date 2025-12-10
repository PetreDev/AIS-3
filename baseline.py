"""
Baseline Approach Module
Implements a simple rule-based baseline for URL security classification.
"""

import numpy as np


class RuleBasedBaseline:
    """
    Simple rule-based baseline classifier.
    
    Rule: If number_of_suspicious_features > K, then BLOCK, else ALLOW
    
    Suspicious features counted:
    - High number of special characters (> threshold)
    - High number of parameters (> threshold)
    - Presence of suspicious keywords
    - Unusually long URL (> threshold)
    """
    
    def __init__(self, K=2):
        """
        Initialize the baseline classifier.
        
        Args:
            K: Threshold for number of suspicious features (default: 2)
        """
        self.K = K
        
        # Thresholds for suspicious features
        self.special_chars_threshold = 5
        self.parameters_threshold = 3
        self.url_length_threshold = 150
        self.suspicious_keywords = ["union", "select", "drop", "script", "alert", "insert"]
    
    def count_suspicious_features(self, features_row):
        """
        Count the number of suspicious features in a URL.
        
        Args:
            features_row: Series or dict with feature values
            
        Returns:
            Number of suspicious features (integer)
        """
        count = 0
        
        # Check special characters
        if features_row['num_special_chars'] > self.special_chars_threshold:
            count += 1
        
        # Check parameters
        if features_row['num_parameters'] > self.parameters_threshold:
            count += 1
        
        # Check suspicious keywords
        if features_row['has_suspicious_keywords'] == 1:
            count += 1
        
        # Check URL length
        if features_row['url_length'] > self.url_length_threshold:
            count += 1
        
        return count
    
    def predict(self, features_row):
        """
        Predict action for a given URL feature vector.
        
        Args:
            features_row: Series or dict with feature values
            
        Returns:
            Action: 0 = ALLOW, 1 = BLOCK
        """
        num_suspicious = self.count_suspicious_features(features_row)
        
        if num_suspicious > self.K:
            return 1  # BLOCK
        else:
            return 0  # ALLOW
    
    def predict_batch(self, features_df):
        """
        Predict actions for a batch of URLs.
        
        Args:
            features_df: DataFrame with feature columns
            
        Returns:
            Array of predictions (0 = ALLOW, 1 = BLOCK)
        """
        predictions = []
        for idx, row in features_df.iterrows():
            pred = self.predict(row)
            predictions.append(pred)
        return np.array(predictions)
    
    def tune_threshold(self, features_df, labels, K_values=None):
        """
        Automatically tune the threshold K to maximize accuracy.
        
        Args:
            features_df: DataFrame with feature columns
            labels: True labels (0 = benign, 1 = attack)
            K_values: List of K values to try (default: [0, 1, 2, 3, 4])
            
        Returns:
            Best K value and corresponding accuracy
        """
        if K_values is None:
            K_values = [0, 1, 2, 3, 4]
        
        best_K = self.K
        best_accuracy = 0.0
        
        for K in K_values:
            self.K = K
            predictions = self.predict_batch(features_df)
            accuracy = np.mean(predictions == labels)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_K = K
        
        self.K = best_K
        print(f"Tuned K to {best_K} with accuracy {best_accuracy:.4f}")
        return best_K, best_accuracy

