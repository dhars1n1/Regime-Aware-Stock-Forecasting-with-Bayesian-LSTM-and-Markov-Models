"""
Data Preprocessing Pipeline for Regime-Aware Bayesian LSTM

This module handles:
1. Loading regime-labeled data
2. Feature engineering with regime information
3. Sequence creation for LSTM training
4. Data scaling and validation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class RegimeAwareDataProcessor:
    """
    Data processor for regime-aware stock forecasting
    
    Handles feature engineering, sequence creation, and data preparation
    for Bayesian LSTM training with market regime information.
    """
    
    def __init__(self, 
                 sequence_length: int = 20, 
                 use_regime_label: bool = True,
                 feature_columns: Union[List[str], str] = 'auto',
                 target_column: str = 'log_return',
                 regime_column: str = 'regime_label',
                 exclude_columns: List[str] = None):
        """
        Initialize data processor
        
        Args:
            sequence_length: Number of timesteps in each sequence
            use_regime_label: Whether to use hard regime labels vs probabilities
            feature_columns: List of feature columns or 'auto' for auto-detection
            target_column: Name of target column
            regime_column: Name of regime column
            exclude_columns: Columns to exclude from features
        """
        self.sequence_length = sequence_length
        self.use_regime_label = use_regime_label
        self.feature_columns_config = feature_columns
        self.target_column = target_column
        self.regime_column = regime_column
        self.exclude_columns = exclude_columns or ['Date', 'regime_viterbi', 'regime_0_prob', 'regime_1_prob', 'regime_2_prob']
        
        self.scalers = {}
        self.regime_encoder = None
        self.feature_columns = []
        self.detected_features = None
        
    def load_data(self, data_path: str = "data/data_with_regimes.csv") -> pd.DataFrame:
        """Load data with regime information"""
        print(f"📥 Loading data from {data_path}")
        
        df = pd.read_csv(data_path, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        
        if 'regime_label' in df.columns:
            print(f"  Regime distribution:")
            regime_counts = df['regime_label'].value_counts()
            for regime, count in regime_counts.items():
                percentage = count / len(df) * 100
                print(f"    {regime}: {count} ({percentage:.1f}%)")
        
        return df
    
    def detect_features(self, df: pd.DataFrame) -> List[str]:
        """
        Automatically detect feature columns from the dataset
        
        Args:
            df: Input dataframe
            
        Returns:
            List of feature column names
        """
        print("🔍 Auto-detecting feature columns...")
        
        # Start with all numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove excluded columns
        excluded_set = set(self.exclude_columns + [self.target_column])
        
        # Get feature columns
        feature_columns = [col for col in numeric_columns if col not in excluded_set]
        
        print(f"  Total columns: {len(df.columns)}")
        print(f"  Numeric columns: {len(numeric_columns)}")  
        print(f"  Excluded: {len(excluded_set)} columns")
        print(f"  Detected features: {len(feature_columns)} columns")
        print(f"  Feature columns: {feature_columns[:10]}{'...' if len(feature_columns) > 10 else ''}")
        
        self.detected_features = feature_columns
        return feature_columns
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get the final list of feature columns to use
        
        Args:
            df: Input dataframe
            
        Returns:
            List of feature column names
        """
        if self.feature_columns_config == 'auto':
            return self.detect_features(df)
        else:
            # Validate provided columns exist
            missing_cols = [col for col in self.feature_columns_config if col not in df.columns]
            if missing_cols:
                print(f"⚠️ Missing columns: {missing_cols}")
                available_cols = [col for col in self.feature_columns_config if col in df.columns]
                print(f"  Using available columns: {available_cols}")
                return available_cols
            return self.feature_columns_config
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create regime-aware features for LSTM training
        
        Features included:
        - Market data: log returns, volume, VIX
        - Technical indicators: RSI, MACD, Bollinger Bands
        - Regime information: encoded labels or probabilities
        - Temporal features: lagged returns, rolling volatility
        """
        print("🔧 Engineering features...")
        
        # Ensure log_return exists
        if 'log_return' not in df.columns and 'Close' in df.columns:
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
            print("  ✅ Created log_return column")
        
        # Core market features (most predictive)
        base_features = []
        if 'log_return' in df.columns:
            base_features.append('log_return')
        if 'Volume' in df.columns:
            base_features.append('Volume') 
        if 'VIX' in df.columns:
            base_features.append('VIX')
            
        # Technical indicators
        technical_features = []
        for feature in ['RSI', 'MACD_diff', 'BB_high', 'BB_low', 'OBV']:
            if feature in df.columns and df[feature].notna().mean() > 0.8:
                technical_features.append(feature)
        
        # Lagged returns for temporal patterns  
        lag_features = []
        for lag in [1, 2, 3, 5]:
            col_name = f'return_lag_{lag}'
            if col_name in df.columns:
                lag_features.append(col_name)
            elif 'log_return' in df.columns:
                # Create lag features if not present
                df[col_name] = df['log_return'].shift(lag)
                lag_features.append(col_name)
        
        # Rolling volatility (additional temporal feature)
        volatility_features = []
        if 'log_return' in df.columns:
            df['volatility_5d'] = df['log_return'].rolling(window=5, min_periods=1).std()
            df['volatility_20d'] = df['log_return'].rolling(window=20, min_periods=1).std()
            volatility_features = ['volatility_5d', 'volatility_20d']
        
        # Regime features - CRITICAL for regime-aware modeling
        regime_features = []
        if self.use_regime_label and 'regime_label' in df.columns:
            regime_features = ['regime_label']
            print("  ✅ Using hard regime labels")
        else:
            # Use regime probabilities
            for i in range(3):
                prob_col = f'regime_{i}_prob'
                if prob_col in df.columns:
                    regime_features.append(prob_col)
            if regime_features:
                print("  ✅ Using regime probabilities")
        
        # Macro features (if available and not too sparse)
        macro_features = []
        for feature in ['CPI', 'Unemployment', 'FedFunds']:
            if feature in df.columns and df[feature].notna().mean() > 0.7:
                macro_features.append(feature)
        
        # Sentiment (if available)
        sentiment_features = []
        if 'sentiment_score' in df.columns and df['sentiment_score'].notna().mean() > 0.5:
            sentiment_features = ['sentiment_score']
        
        # Combine all feature groups
        all_feature_groups = [
            ('Base', base_features),
            ('Technical', technical_features), 
            ('Lagged', lag_features),
            ('Volatility', volatility_features),
            ('Regime', regime_features),
            ('Macro', macro_features),
            ('Sentiment', sentiment_features)
        ]
        
        # Build final feature list
        self.feature_columns = []
        feature_summary = {}
        
        for group_name, features in all_feature_groups:
            available = [f for f in features if f in df.columns]
            self.feature_columns.extend(available)
            feature_summary[group_name] = len(available)
            
            if available:
                print(f"  {group_name}: {available}")
        
        print(f"\n  Total features: {len(self.feature_columns)}")
        
        return df[self.feature_columns + ['log_return']].copy()
    
    def encode_regime_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode regime labels as integers"""
        if self.use_regime_label and 'regime_label' in df.columns:
            if self.regime_encoder is None:
                self.regime_encoder = LabelEncoder()
                df.loc[:, 'regime_label'] = self.regime_encoder.fit_transform(df['regime_label'])
                print(f"  ✅ Encoded regime labels: {self.regime_encoder.classes_}")
            else:
                df.loc[:, 'regime_label'] = self.regime_encoder.transform(df['regime_label'])
        return df
    
    def scale_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using StandardScaler fitted on training data
        
        Args:
            train_df: Training data
            test_df: Test data
            
        Returns:
            Scaled training and test dataframes
        """
        print("📏 Scaling features...")
        
        # Separate features and target
        feature_cols = [col for col in self.feature_columns if col in train_df.columns]
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        train_features_scaled = self.scalers['features'].fit_transform(train_df[feature_cols])
        test_features_scaled = self.scalers['features'].transform(test_df[feature_cols])
        
        # Scale target (log returns)
        self.scalers['target'] = StandardScaler()
        train_target_scaled = self.scalers['target'].fit_transform(
            train_df[['log_return']]
        ).flatten()
        test_target_scaled = self.scalers['target'].transform(
            test_df[['log_return']]
        ).flatten()
        
        # Create scaled DataFrames
        train_scaled = pd.DataFrame(
            train_features_scaled, 
            columns=feature_cols,
            index=train_df.index
        )
        train_scaled['log_return'] = train_target_scaled
        
        test_scaled = pd.DataFrame(
            test_features_scaled,
            columns=feature_cols,
            index=test_df.index
        )
        test_scaled['log_return'] = test_target_scaled
        
        print(f"  ✅ Scaled {len(feature_cols)} features")
        
        return train_scaled, test_scaled
    
    def create_sequences(self, feature_df: pd.DataFrame, target_series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create LSTM sequences with regime information at each timestep
        
        Creates sequences of format:
        - Input: (n_samples, sequence_length, n_features)
        - Target: (n_samples,) - next day's return
        
        Each timestep in the sequence includes regime information.
        """
        print(f"🔄 Creating sequences (length={self.sequence_length})...")
        
        if len(feature_df) < self.sequence_length + 1:
            raise ValueError(f"Insufficient data: need at least {self.sequence_length + 1} samples")
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(feature_df)):
            # Sequence of features (includes regime info at each timestep)
            sequence = feature_df.iloc[i-self.sequence_length:i].values
            X.append(sequence)
            
            # Target: next day's log return
            y.append(target_series.iloc[i])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        print(f"  ✅ Created {len(X)} sequences")
        print(f"  Input shape: {X.shape} (samples, timesteps, features)")
        print(f"  Target shape: {y.shape}")
        
        return X, y
    
    def prepare_data_for_training(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Complete data preparation pipeline
        
        Steps:
        1. Feature engineering
        2. Data cleaning
        3. Temporal train/test split
        4. Feature scaling
        5. Sequence creation
        
        Returns:
            X_train, y_train, X_test, y_test, test_dates
        """
        print("🚀 Starting data preparation pipeline...")
        print("=" * 50)
        
        # 1. Determine feature columns to use
        self.feature_columns = self.get_feature_columns(df)
        
        # 2. Feature engineering
        df_processed = self.engineer_features(df)
        
        # 3. Encode regime labels if needed
        df_processed = self.encode_regime_labels(df_processed)
        
        # 3. Clean data
        print("🧹 Cleaning data...")
        original_len = len(df_processed)
        df_clean = df_processed.dropna()
        print(f"  Removed {original_len - len(df_clean)} rows with NaN values")
        print(f"  Final shape: {df_clean.shape}")
        
        # 4. Temporal split (crucial for time series!)
        split_idx = int(len(df_clean) * (1 - test_size))
        
        train_df = df_clean.iloc[:split_idx]
        test_df = df_clean.iloc[split_idx:]
        
        print(f"\n📅 Data split:")
        print(f"  Train: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} samples)")
        print(f"  Test:  {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} samples)")
        
        # 5. Scale features
        train_scaled, test_scaled = self.scale_features(train_df, test_df)
        
        # 6. Create sequences
        feature_cols = [col for col in self.feature_columns if col in train_scaled.columns]
        
        X_train, y_train = self.create_sequences(
            train_scaled[feature_cols], 
            train_scaled['log_return']
        )
        X_test, y_test = self.create_sequences(
            test_scaled[feature_cols],
            test_scaled['log_return']
        )
        
        # Get test dates aligned with sequences
        test_dates = test_df.index[self.sequence_length:]
        
        print(f"\n✅ Data preparation complete!")
        print(f"  Training sequences: {X_train.shape}")
        print(f"  Test sequences: {X_test.shape}")
        print(f"  Features per timestep: {X_train.shape[2]}")
        print("=" * 50)
        
        return X_train, y_train, X_test, y_test, test_dates
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names used in sequences"""
        return [col for col in self.feature_columns if col != 'log_return']
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Transform predictions back to original scale"""
        if 'target' not in self.scalers:
            raise ValueError("Target scaler not fitted")
        return self.scalers['target'].inverse_transform(predictions.reshape(-1, 1)).flatten()


def main():
    """Test the data preprocessing pipeline"""
    print("🧪 Testing Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Initialize processor
    processor = RegimeAwareDataProcessor(
        sequence_length=20,
        use_regime_label=True
    )
    
    # Load data
    df = processor.load_data()
    
    # Prepare data
    X_train, y_train, X_test, y_test, test_dates = processor.prepare_data_for_training(df)
    
    print(f"\n📊 Final Results:")
    print(f"  Features: {processor.get_feature_names()}")
    print(f"  Sequence length: {processor.sequence_length}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    return processor, X_train, y_train, X_test, y_test, test_dates


if __name__ == "__main__":
    processor, X_train, y_train, X_test, y_test, test_dates = main()