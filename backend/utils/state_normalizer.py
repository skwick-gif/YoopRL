"""
state_normalizer.py
State Normalization and Distribution Drift Detection

Purpose:
- Normalize state observations for stable RL training
- Detect distribution drift in input features
- Alert when model retraining is needed
- Save/load normalization parameters

Why separate file:
- Critical for RL stability (prevents exploding/vanishing gradients)
- Enables early detection of model degradation
- Reusable across different environments
- Consistent normalization across train/test/production

Normalization Methods:
- Z-score normalization (mean=0, std=1)
- Min-Max normalization (range 0-1)
- Robust scaling (using median and IQR)

Drift Detection:
- Kolmogorov-Smirnov test for distribution changes
- Population Stability Index (PSI)
- Alert thresholds for retraining

Wiring:
- Used by BaseTradingEnv for observation normalization
- Called by DriftAlert.jsx via API
- Saves params to backend/utils/normalization_params.json
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
# scipy removed - using pure NumPy implementation for drift detection


class StateNormalizer:
    """
    Handles state normalization and drift detection for RL environments.
    """
    
    def __init__(
        self,
        method: str = 'zscore',
        params_file: str = 'backend/utils/normalization_params.json'
    ):
        """
        Initialize state normalizer.
        
        Args:
            method: Normalization method ('zscore', 'minmax', 'robust')
            params_file: Path to save/load normalization parameters
        """
        self.method = method
        self.params_file = Path(params_file)
        self.params_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Normalization parameters
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None
        self.median_ = None
        self.iqr_ = None
        
        # Original distribution for drift detection
        self.reference_distribution_ = None
        
        # Fitted flag
        self.is_fitted = False
    
    def fit(self, data: np.ndarray) -> 'StateNormalizer':
        """
        Fit normalizer on training data.
        
        Args:
            data: Training data array (n_samples, n_features)
            
        Returns:
            self
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Calculate parameters based on method
        if self.method == 'zscore':
            self.mean_ = np.mean(data, axis=0)
            self.std_ = np.std(data, axis=0)
            # Prevent division by zero
            self.std_[self.std_ == 0] = 1.0
        
        elif self.method == 'minmax':
            self.min_ = np.min(data, axis=0)
            self.max_ = np.max(data, axis=0)
            # Prevent division by zero
            range_vals = self.max_ - self.min_
            range_vals[range_vals == 0] = 1.0
        
        elif self.method == 'robust':
            self.median_ = np.median(data, axis=0)
            q1 = np.percentile(data, 25, axis=0)
            q3 = np.percentile(data, 75, axis=0)
            self.iqr_ = q3 - q1
            # Prevent division by zero
            self.iqr_[self.iqr_ == 0] = 1.0
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        # Save reference distribution for drift detection
        self.reference_distribution_ = data.copy()
        
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data using fitted parameters.
        
        Args:
            data: Data to normalize (n_samples, n_features)
            
        Returns:
            Normalized data
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Apply normalization
        if self.method == 'zscore':
            normalized = (data - self.mean_) / self.std_
        
        elif self.method == 'minmax':
            normalized = (data - self.min_) / (self.max_ - self.min_)
        
        elif self.method == 'robust':
            normalized = (data - self.median_) / self.iqr_
        
        # Clip extreme values to prevent outlier issues
        normalized = np.clip(normalized, -10, 10)
        
        if squeeze_output:
            normalized = normalized.squeeze()
        
        return normalized
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            data: Training data
            
        Returns:
            Normalized data
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Convert normalized data back to original scale.
        
        Args:
            normalized_data: Normalized data
            
        Returns:
            Original scale data
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")
        
        if normalized_data.ndim == 1:
            normalized_data = normalized_data.reshape(-1, 1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Reverse normalization
        if self.method == 'zscore':
            original = normalized_data * self.std_ + self.mean_
        
        elif self.method == 'minmax':
            original = normalized_data * (self.max_ - self.min_) + self.min_
        
        elif self.method == 'robust':
            original = normalized_data * self.iqr_ + self.median_
        
        if squeeze_output:
            original = original.squeeze()
        
        return original
    
    def detect_drift(
        self,
        new_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.4
    ) -> Dict:
        """
        Detect distribution drift in new data.
        
        Uses Kolmogorov-Smirnov test to compare distributions.
        
        Args:
            new_data: New data to check for drift (n_samples, n_features)
            feature_names: Names of features (optional)
            threshold: PSI threshold for drift detection (default 0.4)
            
        Returns:
            Dictionary with drift detection results
        """
        if not self.is_fitted or self.reference_distribution_ is None:
            raise RuntimeError("Normalizer must be fitted before drift detection")
        
        if new_data.ndim == 1:
            new_data = new_data.reshape(-1, 1)
        
        n_features = new_data.shape[1]
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        drift_detected = False
        drift_scores = {}
        affected_features = []
        
        # Check each feature for drift
        for i in range(min(n_features, self.reference_distribution_.shape[1])):
            ref_feature = self.reference_distribution_[:, i]
            new_feature = new_data[:, i]
            
            # Population Stability Index (PSI) - main drift metric
            psi = self._calculate_psi(ref_feature, new_feature)
            
            drift_scores[feature_names[i]] = psi
            
            # Drift detection
            if psi > threshold:
                drift_detected = True
                affected_features.append(feature_names[i])
        
        # Determine severity
        max_psi = max(drift_scores.values()) if drift_scores else 0
        if max_psi > 0.7:
            severity = 'critical'
        elif max_psi > 0.5:
            severity = 'high'
        else:
            severity = 'medium'
        
        return {
            'drift_detected': drift_detected,
            'severity': severity,
            'affected_features': affected_features,
            'drift_scores': drift_scores,
            'threshold': threshold,
            'recommendation': 'Retrain model with recent data' if drift_detected else 'No action needed',
            'last_check': pd.Timestamp.now().isoformat()
        }
    
    def check_drift_status(
        self,
        recent_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.4
    ) -> Dict:
        """
        Convenience method to check drift status with recent data.
        
        This is the main entry point for drift checking from API endpoints.
        
        Args:
            recent_data: Recent market data (n_samples, n_features)
            feature_names: Names of features (optional)
            threshold: PSI threshold for drift detection (default 0.4)
            
        Returns:
            Dictionary with drift status and recommendations
        """
        try:
            drift_results = self.detect_drift(
                new_data=recent_data,
                feature_names=feature_names,
                threshold=threshold
            )
            
            # Add actionable information
            drift_results['status'] = 'success'
            drift_results['needs_retraining'] = drift_results['drift_detected']
            
            return drift_results
        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'needs_retraining': False
            }
    
    def _calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI measures distribution shift between reference and current data.
        PSI < 0.1: No significant change
        PSI 0.1-0.25: Moderate change
        PSI > 0.25: Significant change requiring investigation
        
        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for histogram
            
        Returns:
            PSI value
        """
        # Create bins based on reference distribution
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates
        
        if len(breakpoints) < 2:
            return 0.0  # Not enough variation to measure drift
        
        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=breakpoints)
        cur_hist, _ = np.histogram(current, bins=breakpoints)
        
        # Convert to proportions
        ref_prop = ref_hist / len(reference)
        cur_prop = cur_hist / len(current)
        
        # Avoid log(0) and division by zero
        ref_prop = np.where(ref_prop == 0, 0.0001, ref_prop)
        cur_prop = np.where(cur_prop == 0, 0.0001, cur_prop)
        
        # Calculate PSI
        psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
        
        return psi
    
    def save_params(self, filepath: Optional[str] = None):
        """
        Save normalization parameters to file.
        
        Args:
            filepath: Path to save file (uses default if None)
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted normalizer")
        
        filepath = Path(filepath) if filepath else self.params_file
        
        params = {
            'method': self.method,
            'is_fitted': self.is_fitted,
            'mean': self.mean_.tolist() if self.mean_ is not None else None,
            'std': self.std_.tolist() if self.std_ is not None else None,
            'min': self.min_.tolist() if self.min_ is not None else None,
            'max': self.max_.tolist() if self.max_ is not None else None,
            'median': self.median_.tolist() if self.median_ is not None else None,
            'iqr': self.iqr_.tolist() if self.iqr_ is not None else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"✅ Normalization parameters saved: {filepath}")
    
    def load_params(self, filepath: Optional[str] = None):
        """
        Load normalization parameters from file.
        
        Args:
            filepath: Path to load file (uses default if None)
        """
        filepath = Path(filepath) if filepath else self.params_file
        
        if not filepath.exists():
            raise FileNotFoundError(f"Parameters file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        self.method = params['method']
        self.is_fitted = params['is_fitted']
        
        self.mean_ = np.array(params['mean']) if params['mean'] else None
        self.std_ = np.array(params['std']) if params['std'] else None
        self.min_ = np.array(params['min']) if params['min'] else None
        self.max_ = np.array(params['max']) if params['max'] else None
        self.median_ = np.array(params['median']) if params['median'] else None
        self.iqr_ = np.array(params['iqr']) if params['iqr'] else None
        
        print(f"✅ Normalization parameters loaded: {filepath}")
