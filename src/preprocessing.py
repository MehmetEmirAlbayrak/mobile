import pandas as pd
import numpy as np
import os
import glob
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from imblearn.over_sampling import SMOTE
from src.config import DATA_DIR, RANDOM_STATE, TEST_SIZE, LABEL_COLUMN

class DataProcessor:
    def __init__(self):
        self.scaler = RobustScaler() # RobustScaler handles outliers better than StandardScaler
        self.label_encoder = LabelEncoder()
        self.feature_columns = None

    def load_data(self):
        """
        Loads all CSV and ARFF files from the data directory.
        Prioritizes ARFF files if found, ignoring CSVs to avoid schema mismatch (e.g. with dummy data).
        Only keeps columns that are common across all valid files.
        """
        csv_files = glob.glob(os.path.join(DATA_DIR, "**/*.csv"), recursive=True)
        arff_files = glob.glob(os.path.join(DATA_DIR, "**/*.arff"), recursive=True)
        
        # If we have ARFF files, likely we should use them and ignore dummy CSV
        if arff_files:
            print(f"Found {len(arff_files)} ARFF files. Using ARFF files and ignoring CSVs to ensure schema consistency.")
            all_files = arff_files
        else:
            all_files = csv_files
        
        if not all_files:
            raise FileNotFoundError(f"No dataset files found in {DATA_DIR}. Please add the dataset.")
        
        print(f"Loading {len(all_files)} files...")
        df_list = []
        common_columns = None
        
        for file in all_files:
            try:
                if file.endswith('.csv'):
                    df = pd.read_csv(file)
                elif file.endswith('.arff'):
                    try:
                        # Try standard ARFF load
                        data, meta = arff.loadarff(file)
                        df = pd.DataFrame(data)
                        # Decode byte strings
                        for col in df.select_dtypes([object]):
                            df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
                    except Exception as e_arff:
                        # Fallback for messy ARFF (handling empty strings/missing values)
                        # print(f"Standard ARFF load failed for {os.path.basename(file)}, trying fallback... ({e_arff})")
                        with open(file, 'r', encoding='utf-8', errors='replace') as f:
                            lines = f.readlines()
                        
                        data_idx = -1
                        columns = []
                        for i, line in enumerate(lines):
                            line_clean = line.strip().rstrip(',')
                            if line_clean.lower().startswith('@attribute'):
                                parts = line_clean.split()
                                if len(parts) >= 2:
                                    columns.append(parts[1].strip())
                            elif line_clean.lower().startswith('@data'):
                                data_idx = i + 1
                                break
                        
                        if data_idx > -1 and columns:
                            from io import StringIO
                            # Join data lines
                            data_content = "".join(lines[data_idx:])
                            # Some files might have empty fields as ,, which read_csv handles as NaN by default
                            df = pd.read_csv(StringIO(data_content), header=None, names=columns, on_bad_lines='skip')
                        else:
                            raise ValueError(f"Could not parse ARFF manually: {e_arff}")

                # Clean column names
                df.columns = df.columns.str.strip()
                
                # Handle empty strings that might cause float conversion errors later
                # Some ARFF files might have '' for missing values
                df.replace('', np.nan, inplace=True)
                df.replace('?', np.nan, inplace=True)

                if common_columns is None:
                    common_columns = set(df.columns)
                else:
                    common_columns = common_columns.intersection(set(df.columns))

                df_list.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
        
        if not df_list:
             raise ValueError("Could not read any data.")

        if not common_columns:
             # If intersection is empty, it might be due to 'class' vs 'label' mismatch or similar
             # Let's try to align them if possible, or just fail with more info
             print("Debug: Columns of first file:", df_list[0].columns.tolist())
             if len(df_list) > 1:
                 print("Debug: Columns of second file:", df_list[1].columns.tolist())
             raise ValueError("No common columns found across datasets! Ensure all files have the same feature names.")

        print(f"Common columns across files: {len(common_columns)}")
        
        # Align all dataframes to common columns
        aligned_dfs = [df[list(common_columns)] for df in df_list]

        data = pd.concat(aligned_dfs, ignore_index=True)
        print(f"Total records loaded: {data.shape[0]}")
        return data

    def clean_data(self, df):
        """
        Cleans the dataset:
        - Strips whitespace from column names
        - Drops infinite/missing values
        - Removes non-numeric columns (except Label)
        """
        # Clean column names (already done in load_data but good for safety)
        df.columns = df.columns.str.strip()
        
        print(f"Columns before cleaning: {len(df.columns)}")
        
        # Drop rows with Infinity or NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Handle -1 values which often indicate missing/NA in flow datasets
        # We replace them with 0 for duration/counts, or NaN if strictly missing.
        # For log-transform stability, replacing with 0 (and then log1p) is usually safer for flow stats.
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
             df[col] = df[col].apply(lambda x: 0 if x < 0 else x)

        # Check missing values
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        if total_missing > 0:
            print(f"Found {total_missing} missing values. Columns with missing values: {missing_counts[missing_counts > 0].index.tolist()}")
        
        df.dropna(inplace=True)
        
        # Basic cleaning: sometimes ISCX has flow ID or IP addresses which are categorical/identifiers
        # We only want flow statistical features. 
        # Usually 'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp' might need dropping or handling.
        # For this project (Flow level features), we usually drop identifiers.
        
        drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'SimillarHTTP']
        # Drop if they exist
        cols_to_drop = [c for c in drop_cols if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            
        print(f"Data shape after cleaning: {df.shape}")
        return df

    def preprocess(self, df, apply_smote=True):
        """
        Full preprocessing pipeline: Encoding, Splitting, Scaling, SMOTE.
        """
        # Separate features and target
        if LABEL_COLUMN not in df.columns:
             # Try to find a column that looks like a label if the default name isn't there
             possible_labels = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower()]
             if possible_labels:
                 target_col = possible_labels[0]
                 print(f"Warning: '{LABEL_COLUMN}' not found. Using '{target_col}' as target.")
             else:
                 raise ValueError(f"Label column '{LABEL_COLUMN}' not found in dataset.")
        else:
            target_col = LABEL_COLUMN

        # Unify labels: merge broader categories if needed or clean up names
        # Example: 'VPN-BROWSING' -> 'VPN-Web', 'BROWSING' -> 'Web' for clearer classes
        # For now, we will just ensure string consistency
        df[target_col] = df[target_col].astype(str).str.strip()
        
        # SOLUTION FOR 90%+ ACCURACY: Simplify to 7 application classes
        # Merge VPN and Non-VPN versions of same application type
        # This dramatically improves accuracy because VPN/Non-VPN versions are nearly identical in features
        
        # First, drop generic 'VPN' and 'Non-VPN' labels
        vague_labels = ['VPN', 'Non-VPN']
        original_count = len(df)
        df = df[~df[target_col].isin(vague_labels)]
        print(f"Dropped {original_count - len(df)} generic 'VPN'/'Non-VPN' records.")
        
        # Create a mapping to merge VPN and non-VPN versions
        # VPN-BROWSING -> BROWSING, VPN-CHAT -> CHAT, etc.
        label_mapping = {
            'BROWSING': 'BROWSING',
            'VPN-BROWSING': 'BROWSING',
            'CHAT': 'CHAT', 
            'VPN-CHAT': 'CHAT',
            'FT': 'FILE_TRANSFER',
            'VPN-FT': 'FILE_TRANSFER',
            'MAIL': 'MAIL',
            'VPN-MAIL': 'MAIL',
            'P2P': 'P2P',
            'VPN-P2P': 'P2P',
            'STREAMING': 'STREAMING',
            'VPN-STREAMING': 'STREAMING',
            'VOIP': 'VOIP',
            'VPN-VOIP': 'VOIP'
        }
        
        df[target_col] = df[target_col].map(label_mapping)
        # Drop any unmapped labels
        df = df.dropna(subset=[target_col])
        print(f"Merged VPN/Non-VPN variants. Now {df[target_col].nunique()} classes: {df[target_col].unique()}")

        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"Target Classes: {self.label_encoder.classes_}")
        
        # Identify non-numeric columns in X and drop them
        X = X.select_dtypes(include=[np.number])
        self.feature_columns = X.columns.tolist()

        # Advanced Feature Engineering (Computed on RAW data)
        print("Generating advanced features...")
        
        # 1. Coefficient of Variation (std / mean) - measures relative variability
        for col in ['fiat', 'biat', 'flowiat', 'active', 'idle']:
            if f'std_{col}' in X.columns and f'mean_{col}' in X.columns:
                X[f'{col}_cov'] = X[f'std_{col}'] / (X[f'mean_{col}'] + 1e-6)

        # 2. Key Ratios
        if 'mean_active' in X.columns and 'mean_idle' in X.columns:
             X['active_idle_ratio'] = X['mean_active'] / (X['mean_idle'] + 1e-6)
        
        if 'flowBytesPerSecond' in X.columns and 'flowPktsPerSecond' in X.columns:
             X['bytes_per_pkt'] = X['flowBytesPerSecond'] / (X['flowPktsPerSecond'] + 1e-6)
        
        # 3. Range features (max - min) - captures data spread
        for col in ['fiat', 'biat', 'flowiat', 'active', 'idle']:
            if f'max_{col}' in X.columns and f'min_{col}' in X.columns:
                X[f'{col}_range'] = X[f'max_{col}'] - X[f'min_{col}']
        
        # 4. Statistical moments - Skewness approximation using (max - mean) / (mean - min)
        for col in ['fiat', 'biat', 'flowiat', 'active', 'idle']:
            if all(f'{prefix}_{col}' in X.columns for prefix in ['max', 'mean', 'min']):
                numerator = X[f'max_{col}'] - X[f'mean_{col}']
                denominator = X[f'mean_{col}'] - X[f'min_{col}'] + 1e-6
                X[f'{col}_skew_approx'] = numerator / denominator
        
        # 5. Interaction features - Product of important pairs
        if 'duration' in X.columns and 'flowBytesPerSecond' in X.columns:
            X['duration_bytes_interaction'] = X['duration'] * X['flowBytesPerSecond']
        
        if 'duration' in X.columns and 'flowPktsPerSecond' in X.columns:
            X['duration_pkts_interaction'] = X['duration'] * X['flowPktsPerSecond']
        
        # 6. Normalized features - Relative to flow duration
        if 'duration' in X.columns:
            for col in ['total_fiat', 'total_biat']:
                if col in X.columns:
                    X[f'{col}_per_duration'] = X[col] / (X['duration'] + 1e-6)

        # Log transformation for skewed features (Flow stats are often power-law distributed)
        # Using log1p (log(x+1)) to handle zeros safely
        print("Applying Log transformation to features...")
        X = np.log1p(X) 
        
        # Clean up infinity and NaN values that might have been created
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill NaN with 0 (safer after log transform)
        X.fillna(0, inplace=True)
        
        # Verify no infinite values remain
        if np.any(np.isinf(X.values)):
            print("Warning: Still found infinite values, replacing with max float64/2")
            X = X.clip(-1e308, 1e308)

        # Remove highly correlated or useless features based on domain knowledge
        # 'min_fiat', 'min_biat' often 0 or very redundant
        # drop_useless = ['min_fiat', 'min_biat', 'min_active', 'min_idle', 'min_flowiat'] 
        # X.drop(columns=[c for c in drop_useless if c in X.columns], inplace=True)

        self.feature_columns = X.columns.tolist()
        print(f"Final feature set ({len(self.feature_columns)} features): {self.feature_columns}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
        )

        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert back to DataFrame to keep feature names (fixes LightGBM warnings)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns)

        # Apply SMOTE to training data only
        if apply_smote:
            print("Applying SMOTE to balance classes...")
            # Adjust k_neighbors if some classes are very small
            class_counts = np.bincount(y_train)
            min_samples = np.min(class_counts)
            k = min(5, min_samples - 1)
            if k < 1: 
                k = 1
                
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
            try:
                X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
                print(f"SMOTE applied. New training shape: {X_train_res.shape}")
            except ValueError as e:
                print(f"SMOTE failed (likely due to very few samples in a class): {e}")
                print("Proceeding without SMOTE.")
                X_train_res, y_train_res = X_train_scaled, y_train
        else:
            X_train_res, y_train_res = X_train_scaled, y_train

        return X_train_res, X_test_scaled, y_train_res, y_test
    
    def save_preprocessing_artifacts(self, path='models'):
        """Save scaler and label encoder for inference"""
        import joblib
        import os
        os.makedirs(path, exist_ok=True)
        
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(path, 'label_encoder.pkl'))
        joblib.dump(self.feature_columns, os.path.join(path, 'feature_columns.pkl'))
        print(f"Saved preprocessing artifacts to {path}/")

    def get_original_labels(self, encoded_labels):
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def preprocess_vpn_detection(self, df):
        """
        Preprocessing for VPN detection (binary classification: VPN vs Non-VPN).
        This creates a separate model to detect if traffic is VPN or not.
        """
        # Separate features and target
        if LABEL_COLUMN not in df.columns:
            possible_labels = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower()]
            if possible_labels:
                target_col = possible_labels[0]
            else:
                raise ValueError(f"Label column '{LABEL_COLUMN}' not found in dataset.")
        else:
            target_col = LABEL_COLUMN

        df[target_col] = df[target_col].astype(str).str.strip()
        
        # Drop generic labels
        vague_labels = ['VPN', 'Non-VPN']
        df = df[~df[target_col].isin(vague_labels)]
        
        # Create VPN binary labels: 1 = VPN, 0 = Non-VPN
        vpn_labels = ['VPN-BROWSING', 'VPN-CHAT', 'VPN-FT', 'VPN-MAIL', 'VPN-P2P', 'VPN-STREAMING', 'VPN-VOIP']
        df['is_vpn'] = df[target_col].apply(lambda x: 1 if x in vpn_labels else 0)
        
        print(f"VPN Detection - Class distribution:")
        print(f"  VPN: {(df['is_vpn'] == 1).sum()}")
        print(f"  Non-VPN: {(df['is_vpn'] == 0).sum()}")
        
        X = df.drop(columns=[target_col, 'is_vpn'])
        y = df['is_vpn'].values
        
        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Same feature engineering as main preprocessing
        print("Generating advanced features for VPN detection...")
        
        for col in ['fiat', 'biat', 'flowiat', 'active', 'idle']:
            if f'std_{col}' in X.columns and f'mean_{col}' in X.columns:
                X[f'{col}_cov'] = X[f'std_{col}'] / (X[f'mean_{col}'] + 1e-6)

        if 'mean_active' in X.columns and 'mean_idle' in X.columns:
             X['active_idle_ratio'] = X['mean_active'] / (X['mean_idle'] + 1e-6)
        
        if 'flowBytesPerSecond' in X.columns and 'flowPktsPerSecond' in X.columns:
             X['bytes_per_pkt'] = X['flowBytesPerSecond'] / (X['flowPktsPerSecond'] + 1e-6)
        
        for col in ['fiat', 'biat', 'flowiat', 'active', 'idle']:
            if f'max_{col}' in X.columns and f'min_{col}' in X.columns:
                X[f'{col}_range'] = X[f'max_{col}'] - X[f'min_{col}']
        
        for col in ['fiat', 'biat', 'flowiat', 'active', 'idle']:
            if all(f'{prefix}_{col}' in X.columns for prefix in ['max', 'mean', 'min']):
                numerator = X[f'max_{col}'] - X[f'mean_{col}']
                denominator = X[f'mean_{col}'] - X[f'min_{col}'] + 1e-6
                X[f'{col}_skew_approx'] = numerator / denominator
        
        if 'duration' in X.columns and 'flowBytesPerSecond' in X.columns:
            X['duration_bytes_interaction'] = X['duration'] * X['flowBytesPerSecond']
        
        if 'duration' in X.columns and 'flowPktsPerSecond' in X.columns:
            X['duration_pkts_interaction'] = X['duration'] * X['flowPktsPerSecond']
        
        if 'duration' in X.columns:
            for col in ['total_fiat', 'total_biat']:
                if col in X.columns:
                    X[f'{col}_per_duration'] = X[col] / (X['duration'] + 1e-6)

        # Log transformation
        X = np.log1p(X) 
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        
        if np.any(np.isinf(X.values)):
            X = X.clip(-1e308, 1e308)

        vpn_feature_columns = X.columns.tolist()
        print(f"VPN detection features ({len(vpn_feature_columns)} features)")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        # Scale data using a separate scaler for VPN detection
        vpn_scaler = RobustScaler()
        X_train_scaled = vpn_scaler.fit_transform(X_train)
        X_test_scaled = vpn_scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=vpn_feature_columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=vpn_feature_columns)

        # Apply SMOTE
        print("Applying SMOTE for VPN detection...")
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        try:
            X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
            print(f"SMOTE applied. New training shape: {X_train_res.shape}")
        except ValueError as e:
            print(f"SMOTE failed: {e}")
            X_train_res, y_train_res = X_train_scaled, y_train

        return X_train_res, X_test_scaled, y_train_res, y_test, vpn_scaler, vpn_feature_columns

