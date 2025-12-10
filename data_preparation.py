"""
Data Preparation Module
Handles loading, feature extraction, and state discretization for the CSIC 2010 HTTP dataset.
Professional preprocessing pipeline with data validation, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.utils import resample
from urllib.parse import urlparse, parse_qs
import re
import math
import os
import warnings
warnings.filterwarnings('ignore')


class DataPreparator:
    """Handles data loading, feature extraction, and state discretization."""
    
    def __init__(self, dataset_path='dataset/csic_database.csv'):
        """
        Initialize the data preparator.
        
        Args:
            dataset_path: Path to the CSIC 2010 dataset CSV file
        """
        self.dataset_path = dataset_path
        
        # Extended suspicious keywords for SQL injection, XSS, and other attacks
        self.suspicious_keywords = [
            "union", "select", "drop", "script", "alert", "insert", "delete", "update",
            "exec", "execute", "eval", "javascript", "onerror", "onload", "onclick",
            "iframe", "object", "embed", "base64", "char", "chr", "concat", "cast",
            "declare", "exec", "xp_", "sp_", "cmd", "powershell", "bash", "sh"
        ]
        
        # Special characters that may indicate attacks
        self.special_chars = ['?', '&', '=', '%', "'", '"', '<', '>', ';', '@', '{', '}', '[', ']', '(', ')', '|', '\\', '/', '*']
        
        # SQL injection patterns
        self.sql_patterns = [
            r"(\bOR\b|\bAND\b).*?=.*?=",  # OR/AND injection
            r"'.*?'.*?=.*?'",  # String comparison injection
            r"--",  # SQL comment
            r"/\*.*?\*/",  # SQL block comment
            r"\bUNION\b.*?\bSELECT\b",  # UNION SELECT
            r"\bDROP\b.*?\bTABLE\b",  # DROP TABLE
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r"<script.*?>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",  # Event handlers
            r"<iframe.*?>",
            r"<object.*?>",
        ]
        
        # Discretization parameters (will be set after fitting)
        self.binning_bounds = {}
        self.quantile_bounds = {}
        self.num_bins_per_feature = {}
        self.discretization_method = None
        
        # Normalization parameters (will be set after fitting)
        self.normalize_features = True  # Enable normalization by default
        self.scaler = None  # Will be set to RobustScaler or MinMaxScaler
        self.normalization_method = 'robust'  # 'robust' (handles outliers) or 'minmax'
        
        # Class imbalance handling parameters
        self.handle_imbalance = True  # Enable class imbalance handling by default
        self.imbalance_method = 'upsample'  # 'upsample', 'downsample', or 'none'
        self.target_ratio = 0.5  # Target ratio for minority class (0.5 = balanced)
        
    def load_data(self):
        """
        Load and clean the CSIC 2010 dataset with professional preprocessing.
        If file doesn't exist, generate synthetic data.
        
        Returns:
            DataFrame with URL and label columns
        """
        if os.path.exists(self.dataset_path):
            print(f"Loading dataset from {self.dataset_path}...")
            
            # Read CSV with error handling
            try:
                df = pd.read_csv(self.dataset_path, low_memory=False)
            except Exception as e:
                print(f"Error reading CSV: {e}")
                print("Generating synthetic data instead...")
                return self._generate_synthetic_data()
            
            # Extract URL and classification columns
            if 'URL' in df.columns and 'classification' in df.columns:
                data = df[['URL', 'classification']].copy()
            else:
                # Try alternative column names
                url_cols = [col for col in df.columns if 'url' in col.lower()]
                label_cols = [col for col in df.columns if 'classification' in col.lower() or 'label' in col.lower()]
                
                if not url_cols or not label_cols:
                    print("Could not find URL or classification columns. Generating synthetic data...")
                    return self._generate_synthetic_data()
                
                data = df[[url_cols[0], label_cols[0]]].copy()
                data.columns = ['URL', 'classification']
            
            print(f"Initial dataset size: {len(data)} samples")
            
            # Professional data cleaning
            data = self._clean_data(data)
            
            # Remove duplicates
            initial_size = len(data)
            data = data.drop_duplicates(subset=['URL'], keep='first')
            duplicates_removed = initial_size - len(data)
            if duplicates_removed > 0:
                print(f"Removed {duplicates_removed} duplicate URLs")
            
            # Ensure labels are 0 (benign) or 1 (attack)
            # In CSIC dataset: 0 = Normal (benign), 1 = Attack
            data['label'] = data['classification'].apply(lambda x: 1 if x == 1 else 0)
            data = data[['URL', 'label']].copy()
            
            # Remove any rows with missing or invalid URLs
            data = data[data['URL'].notna() & (data['URL'] != 'nan') & (data['URL'] != '')].copy()
            data = data[data['URL'].str.len() > 0].copy()
            
            # Data quality check
            is_imbalanced = self._validate_data(data)
            
            print(f"\nFinal dataset size: {len(data)} samples")
            print(f"  Benign samples: {sum(data['label'] == 0)} ({100*sum(data['label'] == 0)/len(data):.1f}%)")
            print(f"  Attack samples: {sum(data['label'] == 1)} ({100*sum(data['label'] == 1)/len(data):.1f}%)")
            
            # Handle class imbalance if detected and enabled
            if is_imbalanced and self.handle_imbalance:
                print("\nHandling class imbalance...")
                data = self._balance_classes(data)
                print(f"After balancing: {len(data)} samples")
                print(f"  Benign samples: {sum(data['label'] == 0)} ({100*sum(data['label'] == 0)/len(data):.1f}%)")
                print(f"  Attack samples: {sum(data['label'] == 1)} ({100*sum(data['label'] == 1)/len(data):.1f}%)")
            
        else:
            print(f"Dataset not found at {self.dataset_path}. Generating synthetic data...")
            data = self._generate_synthetic_data()
        
        return data
    
    def _clean_data(self, data):
        """
        Professional data cleaning pipeline.
        
        Args:
            data: DataFrame with URL and classification columns
            
        Returns:
            Cleaned DataFrame
        """
        # Convert URL to string and handle NaN
        data['URL'] = data['URL'].astype(str)
        
        # Remove 'HTTP/1.1' suffix if present
        data['URL'] = data['URL'].str.replace(' HTTP/1.1', '', regex=False)
        data['URL'] = data['URL'].str.replace(' HTTP/1.0', '', regex=False)
        
        # Remove leading/trailing whitespace
        data['URL'] = data['URL'].str.strip()
        
        # Normalize URL encoding (basic normalization)
        # Note: We keep encoded characters as they may be attack indicators
        
        # Remove empty URLs
        data = data[data['URL'].str.len() > 0]
        
        return data
    
    def _validate_data(self, data):
        """
        Validate data quality and print statistics.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            bool: True if class imbalance is detected
        """
        print("\nData Quality Checks:")
        print(f"  Total samples: {len(data)}")
        print(f"  Missing URLs: {data['URL'].isna().sum()}")
        print(f"  Missing labels: {data['label'].isna().sum()}")
        print(f"  URL length range: {data['URL'].str.len().min()} - {data['URL'].str.len().max()}")
        print(f"  Average URL length: {data['URL'].str.len().mean():.1f}")
        
        # Check class balance
        class_balance = data['label'].value_counts(normalize=True)
        imbalance_ratio = abs(class_balance[0] - class_balance.get(1, 0))
        if imbalance_ratio > 0.3:
            print(f"  Warning: Significant class imbalance detected (ratio: {imbalance_ratio:.2f})")
            print(f"    Class distribution: {dict(class_balance)}")
        return imbalance_ratio > 0.3
    
    def _balance_classes(self, data):
        """
        Handle class imbalance using upsampling or downsampling.
        
        Args:
            data: DataFrame with 'label' column
            
        Returns:
            Balanced DataFrame
        """
        if self.imbalance_method == 'none':
            return data
        
        # Separate majority and minority classes
        majority_class = data[data['label'] == data['label'].value_counts().idxmax()]
        minority_class = data[data['label'] == data['label'].value_counts().idxmin()]
        
        majority_count = len(majority_class)
        minority_count = len(minority_class)
        
        print(f"  Majority class: {majority_count} samples")
        print(f"  Minority class: {minority_count} samples")
        
        if self.imbalance_method == 'upsample':
            # Upsample minority class to match target ratio
            target_minority = int(majority_count * self.target_ratio / (1 - self.target_ratio))
            if target_minority > minority_count:
                minority_upsampled = resample(
                    minority_class,
                    replace=True,
                    n_samples=target_minority,
                    random_state=42
                )
                balanced_data = pd.concat([majority_class, minority_upsampled])
                print(f"  Upsampled minority class to {target_minority} samples")
            else:
                balanced_data = data
                print(f"  No upsampling needed")
        
        elif self.imbalance_method == 'downsample':
            # Downsample majority class to match target ratio
            target_majority = int(minority_count * (1 - self.target_ratio) / self.target_ratio)
            if target_majority < majority_count:
                majority_downsampled = resample(
                    majority_class,
                    replace=False,
                    n_samples=target_majority,
                    random_state=42
                )
                balanced_data = pd.concat([majority_downsampled, minority_class])
                print(f"  Downsampled majority class to {target_majority} samples")
            else:
                balanced_data = data
                print(f"  No downsampling needed")
        
        else:
            balanced_data = data
        
        # Shuffle the balanced dataset
        balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return balanced_data
    
    def _generate_synthetic_data(self, n_samples=15000):
        """
        Generate synthetic HTTP request data for testing.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with URL and label columns
        """
        np.random.seed(42)
        urls = []
        labels = []
        
        # Generate benign URLs
        benign_templates = [
            "http://example.com/page?id={}&name={}",
            "http://example.com/products?category={}&page={}",
            "http://example.com/user/profile?userid={}",
            "http://example.com/search?q={}",
            "http://example.com/index.html",
        ]
        
        # Generate attack URLs
        attack_templates = [
            "http://example.com/page?id=1' OR '1'='1",
            "http://example.com/page?cmd=script>alert('xss')",
            "http://example.com/page?id=1 UNION SELECT * FROM users",
            "http://example.com/page?id=1; DROP TABLE users;",
            "http://example.com/page?id=1' UNION SELECT password FROM users",
        ]
        
        for i in range(n_samples):
            if np.random.random() < 0.3:  # 30% attacks
                template = np.random.choice(attack_templates)
                label = 1
            else:
                template = np.random.choice(benign_templates)
                label = 0
            
            url = template.format(
                np.random.randint(1, 1000),
                np.random.randint(1, 1000)
            )
            urls.append(url)
            labels.append(label)
        
        return pd.DataFrame({'URL': urls, 'label': labels})
    
    def extract_features(self, data):
        """
        Extract comprehensive numerical features from URLs using professional feature engineering.
        
        Features extracted:
        - Basic: URL length, special chars, digits, parameters
        - Advanced: Entropy, path depth, encoding patterns, attack patterns
        - Security: SQL injection patterns, XSS patterns, suspicious keywords
        
        Args:
            data: DataFrame with 'URL' column
            
        Returns:
            DataFrame with extracted features
        """
        print("Extracting features from URLs (professional feature engineering)...")
        
        features = pd.DataFrame()
        urls = data['URL'].astype(str)
        
        # ========== Basic Features ==========
        # URL length
        features['url_length'] = urls.str.len()
        
        # Count special characters
        features['num_special_chars'] = urls.apply(
            lambda url: sum(1 for char in url if char in self.special_chars)
        )
        
        # Count digits
        features['num_digits'] = urls.apply(
            lambda url: sum(1 for char in url if char.isdigit())
        )
        
        # Count parameters (properly count query parameters)
        features['num_parameters'] = urls.apply(self._count_parameters)
        
        # ========== Advanced Features ==========
        # URL entropy (measure of randomness/complexity)
        features['url_entropy'] = urls.apply(self._calculate_entropy)
        
        # Path depth (number of path segments)
        features['path_depth'] = urls.apply(self._get_path_depth)
        
        # Has file extension
        features['has_file_extension'] = urls.apply(self._has_file_extension)
        
        # Ratio of special characters
        features['special_char_ratio'] = features['num_special_chars'] / (features['url_length'] + 1)
        
        # Ratio of digits
        features['digit_ratio'] = features['num_digits'] / (features['url_length'] + 1)
        
        # ========== Security-Specific Features ==========
        # Presence of suspicious keywords (binary)
        features['has_suspicious_keywords'] = urls.apply(
            lambda url: 1 if any(keyword in url.lower() for keyword in self.suspicious_keywords) else 0
        )
        
        # Count of suspicious keywords
        features['suspicious_keyword_count'] = urls.apply(
            lambda url: sum(1 for keyword in self.suspicious_keywords if keyword in url.lower())
        )
        
        # SQL injection pattern detection
        features['has_sql_pattern'] = urls.apply(self._detect_sql_patterns)
        
        # XSS pattern detection
        features['has_xss_pattern'] = urls.apply(self._detect_xss_patterns)
        
        # URL encoding patterns (double encoding, etc.)
        features['has_encoding_patterns'] = urls.apply(self._detect_encoding_patterns)
        
        # Has suspicious characters combinations
        features['has_suspicious_combos'] = urls.apply(self._detect_suspicious_combinations)
        
        # ========== URL Structure Features ==========
        # Has query string
        features['has_query'] = urls.str.contains(r'\?', regex=True).astype(int)
        
        # Has fragment
        features['has_fragment'] = urls.str.contains(r'#', regex=True).astype(int)
        
        # Parameter value length (average)
        features['avg_param_length'] = urls.apply(self._get_avg_param_length)
        
        # Add label
        features['label'] = data['label'].values
        
        # Handle infinite or NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        # Outlier handling: clip extreme values to reduce their impact
        # Clip values beyond 3 standard deviations (for continuous features only)
        continuous_features = ['url_length', 'num_special_chars', 'num_digits', 'num_parameters',
                              'url_entropy', 'path_depth', 'special_char_ratio', 'digit_ratio',
                              'suspicious_keyword_count', 'avg_param_length']
        
        for col in continuous_features:
            if col in features.columns:
                mean = features[col].mean()
                std = features[col].std()
                if std > 0:
                    # Clip outliers to ±3 standard deviations
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std
                    features[col] = features[col].clip(lower=lower_bound, upper=upper_bound)
        
        print(f"Feature extraction complete. Shape: {features.shape}")
        print(f"  Features extracted: {len(features.columns) - 1} (excluding label)")
        
        return features
    
    def _count_parameters(self, url):
        """Count number of query parameters in URL."""
        try:
            parsed = urlparse(url)
            if parsed.query:
                return len(parse_qs(parsed.query))
            return 0
        except:
            # Fallback: count '&' characters
            return max(url.count('&'), url.count('='))
    
    def _calculate_entropy(self, url):
        """Calculate Shannon entropy of URL string."""
        if not url or len(url) == 0:
            return 0
        try:
            prob = [float(url.count(c)) / len(url) for c in dict.fromkeys(list(url))]
            entropy = -sum([p * math.log(p) / math.log(2.0) for p in prob if p > 0])
            return entropy
        except:
            return 0
    
    def _get_path_depth(self, url):
        """Get depth of URL path (number of segments)."""
        try:
            parsed = urlparse(url)
            path = parsed.path.strip('/')
            if not path:
                return 0
            return len([s for s in path.split('/') if s])
        except:
            return 0
    
    def _has_file_extension(self, url):
        """Check if URL has a file extension."""
        try:
            parsed = urlparse(url)
            path = parsed.path
            if '.' in path:
                ext = path.split('.')[-1].split('?')[0].split('#')[0]
                return 1 if ext and len(ext) <= 5 else 0
            return 0
        except:
            return 0
    
    def _detect_sql_patterns(self, url):
        """Detect SQL injection patterns in URL."""
        url_lower = url.lower()
        for pattern in self.sql_patterns:
            if re.search(pattern, url_lower, re.IGNORECASE):
                return 1
        return 0
    
    def _detect_xss_patterns(self, url):
        """Detect XSS patterns in URL."""
        for pattern in self.xss_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return 1
        return 0
    
    def _detect_encoding_patterns(self, url):
        """Detect URL encoding patterns (double encoding, etc.)."""
        # Check for high percentage of encoded characters
        encoded_chars = url.count('%')
        if len(url) > 0 and encoded_chars / len(url) > 0.1:
            return 1
        # Check for double encoding patterns
        if '%25' in url or '%2525' in url:
            return 1
        return 0
    
    def _detect_suspicious_combinations(self, url):
        """Detect suspicious character combinations."""
        suspicious_combos = [
            '..',  # Path traversal
            '//',  # Protocol confusion
            '<?',  # PHP tags
            '<%',  # ASP tags
        ]
        for combo in suspicious_combos:
            if combo in url:
                return 1
        return 0
    
    def _get_avg_param_length(self, url):
        """Get average length of parameter values."""
        try:
            parsed = urlparse(url)
            if parsed.query:
                params = parse_qs(parsed.query)
                if params:
                    lengths = [len(str(v)) for values in params.values() for v in values]
                    return np.mean(lengths) if lengths else 0
            return 0
        except:
            return 0
    
    def fit_normalization(self, features):
        """
        Fit normalization scaler on training data.
        Normalizes features to improve discretization quality.
        
        Args:
            features: DataFrame with extracted features (without label)
        """
        if not self.normalize_features:
            return
        
        # Core features for normalization (continuous features used in state encoding)
        feature_cols = ['url_length', 'num_special_chars', 'num_digits', 'num_parameters']
        
        # Only normalize features that exist and are continuous
        cols_to_normalize = [col for col in feature_cols if col in features.columns]
        
        if not cols_to_normalize:
            print("No features to normalize")
            return
        
        # Use RobustScaler (handles outliers better) or MinMaxScaler
        if self.normalization_method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = MinMaxScaler()
        
        # Fit scaler on training data
        self.scaler.fit(features[cols_to_normalize])
        print(f"Normalization fitted using {self.normalization_method} scaling")
    
    def normalize_features_data(self, features):
        """
        Apply normalization to features.
        
        Args:
            features: DataFrame with features
            
        Returns:
            DataFrame with normalized features
        """
        if not self.normalize_features or self.scaler is None:
            return features
        
        # Core features for normalization
        feature_cols = ['url_length', 'num_special_chars', 'num_digits', 'num_parameters']
        cols_to_normalize = [col for col in feature_cols if col in features.columns]
        
        if not cols_to_normalize:
            return features
        
        # Create a copy to avoid modifying original
        features_normalized = features.copy()
        
        # Normalize only the specified columns
        features_normalized[cols_to_normalize] = self.scaler.transform(features[cols_to_normalize])
        
        return features_normalized
    
    def fit_discretization(self, features, method='binning'):
        """
        Fit discretization parameters based on training data.
        Optionally normalizes features before discretization for better bin boundaries.
        
        Args:
            features: DataFrame with extracted features (without label)
            method: 'binning' for fixed intervals or 'quantization' for quantiles
        """
        self.discretization_method = method
        
        # Normalize features before discretization if enabled
        if self.normalize_features:
            # Fit normalization first
            self.fit_normalization(features)
            # Normalize features for discretization
            features_for_discretization = self.normalize_features_data(features)
        else:
            features_for_discretization = features
        
        # Core features for discretization (used in state encoding)
        feature_cols = ['url_length', 'num_special_chars', 'num_digits', 'num_parameters']
        
        if method == 'binning':
            # Fixed interval binning
            # URL length: 0-100 = bin 0, 100-200 = bin 1, >200 = bin 2
            # Special chars: 0-5 = bin 0, 6-10 = bin 1, >10 = bin 2
            self.binning_bounds = {
                'url_length': [0, 100, 200, float('inf')],
                'num_special_chars': [0, 5, 10, float('inf')],
                'num_digits': [0, 5, 15, float('inf')],
                'num_parameters': [0, 2, 5, float('inf')]
            }
            
            # Calculate number of bins per feature
            self.num_bins_per_feature = {
                'url_length': 3,
                'num_special_chars': 3,
                'num_digits': 3,
                'num_parameters': 3
            }
            
        elif method == 'quantization':
            # Quantile-based discretization (4 quantiles)
            # Handle features with low variance or many zeros
            self.quantile_bounds = {}
            self.num_bins_per_feature = {}
            
            for col in feature_cols:
                if col not in features_for_discretization.columns:
                    continue
                    
                col_data = features_for_discretization[col].dropna()
                
                # Handle constant or near-constant features
                if col_data.nunique() <= 1:
                    self.quantile_bounds[col] = [col_data.min(), col_data.max() + 1]
                    self.num_bins_per_feature[col] = 1
                    continue
                
                # Use quantiles, handling edge cases
                try:
                    quantiles = np.quantile(col_data, [0, 0.25, 0.5, 0.75, 1.0])
                    # Ensure quantiles are unique
                    quantiles = np.unique(quantiles)
                    if len(quantiles) < 2:
                        quantiles = [col_data.min(), col_data.max() + 1]
                    
                    self.quantile_bounds[col] = quantiles
                    self.num_bins_per_feature[col] = len(quantiles) - 1
                except Exception as e:
                    # Fallback to simple min/max
                    self.quantile_bounds[col] = [col_data.min(), col_data.max() + 1]
                    self.num_bins_per_feature[col] = 1
        
        print(f"Discretization fitted using {method} method")
    
    def discretize_feature(self, value, feature_name):
        """
        Discretize a single feature value.
        
        Args:
            value: Feature value to discretize
            feature_name: Name of the feature
            
        Returns:
            Bin index (integer)
        """
        if self.discretization_method == 'binning':
            bounds = self.binning_bounds[feature_name]
            for bin_idx in range(len(bounds) - 1):
                if bounds[bin_idx] <= value < bounds[bin_idx + 1]:
                    return bin_idx
            return len(bounds) - 2  # Last bin
        
        elif self.discretization_method == 'quantization':
            bounds = self.quantile_bounds[feature_name]
            for bin_idx in range(len(bounds) - 1):
                if bounds[bin_idx] <= value < bounds[bin_idx + 1]:
                    return bin_idx
            return len(bounds) - 2  # Last bin
        
        return 0
    
    def encode_state(self, features_row):
        """
        Encode features into a discrete state index.
        
        State is encoded as a combination of core features:
        - url_length_bin
        - num_special_chars_bin
        - num_digits_bin
        - num_parameters_bin
        - has_suspicious_keywords (binary)
        - has_sql_pattern (binary)
        - has_xss_pattern (binary)
        
        Args:
            features_row: Series or dict with feature values
            
        Returns:
            Discrete state index (integer)
        """
        # Extract core features for discretization
        feature_cols = ['url_length', 'num_special_chars', 'num_digits', 'num_parameters']
        feature_values = {col: features_row[col] for col in feature_cols if col in features_row}
        
        # Normalize features if normalization is enabled
        if self.normalize_features and self.scaler is not None:
            # Create a small DataFrame for normalization
            import pandas as pd
            feature_df = pd.DataFrame([feature_values])
            normalized_df = self.normalize_features_data(feature_df)
            feature_values = normalized_df.iloc[0].to_dict()
        
        # Discretize continuous features (core features)
        url_length_bin = self.discretize_feature(feature_values.get('url_length', features_row['url_length']), 'url_length')
        specchar_bin = self.discretize_feature(feature_values.get('num_special_chars', features_row['num_special_chars']), 'num_special_chars')
        digits_bin = self.discretize_feature(feature_values.get('num_digits', features_row['num_digits']), 'num_digits')
        params_bin = self.discretize_feature(feature_values.get('num_parameters', features_row['num_parameters']), 'num_parameters')
        
        # Binary features (already discrete)
        keyword_bin = int(features_row.get('has_suspicious_keywords', 0))
        sql_bin = int(features_row.get('has_sql_pattern', 0))
        xss_bin = int(features_row.get('has_xss_pattern', 0))
        
        # Combine bins into a single state index using base conversion
        # State encoding: [url_length][specchar][digits][params][keyword][sql][xss]
        specchar_max = self.num_bins_per_feature.get('num_special_chars', 4)
        digits_max = self.num_bins_per_feature.get('num_digits', 4)
        params_max = self.num_bins_per_feature.get('num_parameters', 4)
        url_length_max = self.num_bins_per_feature.get('url_length', 4)
        
        # Calculate state index
        state = (url_length_bin * specchar_max * digits_max * params_max * 2 * 2 * 2 +
                specchar_bin * digits_max * params_max * 2 * 2 * 2 +
                digits_bin * params_max * 2 * 2 * 2 +
                params_bin * 2 * 2 * 2 +
                keyword_bin * 2 * 2 +
                sql_bin * 2 +
                xss_bin)
        
        return state
    
    def get_num_states(self):
        """
        Calculate total number of possible states.
        
        Returns:
            Total number of discrete states
        """
        url_length_bins = self.num_bins_per_feature.get('url_length', 4)
        specchar_bins = self.num_bins_per_feature.get('num_special_chars', 4)
        digits_bins = self.num_bins_per_feature.get('num_digits', 4)
        params_bins = self.num_bins_per_feature.get('num_parameters', 4)
        keyword_bins = 2  # Binary feature
        sql_bins = 2      # Binary feature
        xss_bins = 2      # Binary feature
        
        return url_length_bins * specchar_bins * digits_bins * params_bins * keyword_bins * sql_bins * xss_bins
    
    def split_data(self, data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split data into training, validation, and test sets.
        
        Args:
            data: DataFrame with features and labels
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Ensure minimum 10,000 samples
        if len(data) < 10000:
            print(f"Warning: Dataset has only {len(data)} samples, minimum 10,000 required")
        
        # First split: train vs (val + test)
        train_data, temp_data = train_test_split(
            data, test_size=(val_ratio + test_ratio), random_state=42, stratify=data['label']
        )
        
        # Second split: val vs test
        val_data, test_data = train_test_split(
            temp_data, test_size=test_ratio/(val_ratio + test_ratio), 
            random_state=42, stratify=temp_data['label']
        )
        
        print(f"Data split:")
        print(f"  Training: {len(train_data)} samples")
        print(f"  Validation: {len(val_data)} samples")
        print(f"  Test: {len(test_data)} samples")
        
        return train_data, val_data, test_data

