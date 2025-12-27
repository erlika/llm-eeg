"""
Common Spatial Patterns (CSP) Feature Extractor
================================================

This module implements the Common Spatial Patterns algorithm for EEG feature extraction.

CSP is the most widely used and effective feature extraction method for motor imagery
(MI) based Brain-Computer Interfaces (BCI). It learns spatial filters that maximize
the variance difference between two classes.

Algorithm Overview:
------------------
1. Compute class-wise covariance matrices
2. Solve the generalized eigenvalue problem
3. Select top and bottom eigenvectors as spatial filters
4. Project data through filters and compute log-variance features

Mathematical Background:
-----------------------
Given two classes with covariance matrices C1 and C2:
- Solve: C1 * W = (C1 + C2) * W * D
- W contains spatial filters
- D contains eigenvalues (discrimination power)
- Features: log(var(W^T * X))

For Multi-class (One-vs-Rest):
-----------------------------
For K classes, CSP is applied in One-vs-Rest manner:
- Class k vs all other classes
- Combine filters from all binary problems

Usage Example:
    ```python
    from src.features.extractors.csp import CSPExtractor
    
    # Create extractor
    csp = CSPExtractor(n_components=6)
    
    # Fit on training data
    csp.fit(X_train, y_train)  # X: (n_trials, n_channels, n_samples)
    
    # Extract features
    features = csp.extract(X_test)  # Shape: (n_trials, n_components)
    
    # Visualize spatial filters
    filters = csp.get_spatial_filters()
    patterns = csp.get_spatial_patterns()
    ```

Performance Notes:
-----------------
- Best for binary classification, use OVR for multi-class
- Requires at least n_components + 1 trials per class
- Sensitive to noise and artifacts
- Works best with bandpass filtered data (8-30 Hz for MI)

References:
----------
1. Ramoser et al., "Optimal Spatial Filtering of Single Trial EEG During 
   Imagined Hand Movement", IEEE TREHAB, 2000
2. Blankertz et al., "Optimizing Spatial Filters for Robust EEG Single-Trial 
   Analysis", IEEE SPM, 2008

Author: EEG-BCI Framework
Date: 2024
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
from scipy import linalg
import logging

from src.features.base import BaseFeatureExtractor
from src.core.registry import registered

# Configure logging
logger = logging.getLogger(__name__)


@registered('feature_extractor', 'csp')
class CSPExtractor(BaseFeatureExtractor):
    """
    Common Spatial Patterns feature extractor for EEG signals.
    
    CSP learns spatial filters that maximize variance for one class
    while minimizing it for another, making it ideal for motor imagery BCI.
    
    Attributes:
        n_components (int): Number of CSP components (filters) to use.
            For binary classification, this should be even (e.g., 6).
            Uses n_components/2 from each end of the eigenvalue spectrum.
        reg (float): Regularization parameter for covariance estimation.
            Use small values (0.01-0.1) for noisy data.
        log (bool): Whether to apply log transform to features.
            Log transform is standard and recommended.
        norm_trace (bool): Whether to normalize covariance by trace.
            Helps with numerical stability.
        
    Multi-class Support:
        For multi-class problems (>2 classes), CSP uses One-vs-Rest (OVR):
        - Each class is compared against all others
        - Filters from all binary problems are combined
        - Total features = n_components * n_classes
    
    Example:
        >>> csp = CSPExtractor(n_components=6, reg=0.01)
        >>> csp.fit(X_train, y_train)
        >>> features = csp.extract(X_test)
        >>> print(features.shape)  # (n_trials, 6) for binary
    """
    
    def __init__(self,
                 n_components: int = 6,
                 reg: float = 0.0,
                 log: bool = True,
                 norm_trace: bool = True):
        """
        Initialize CSP extractor.
        
        Args:
            n_components: Number of CSP components to extract.
                For binary: uses n_components/2 from each end.
                For multi-class OVR: uses n_components per class pair.
            reg: Regularization parameter (0.0 = no regularization).
                Adds reg * trace(C) * I to covariance matrices.
            log: Apply log transform to variance features.
            norm_trace: Normalize covariance matrices by their trace.
        """
        super().__init__()
        
        # CSP parameters
        self._n_components = n_components
        self._reg = reg
        self._log = log
        self._norm_trace = norm_trace
        
        # Learned parameters (set during fit)
        self._filters: Optional[np.ndarray] = None  # Spatial filters W
        self._patterns: Optional[np.ndarray] = None  # Spatial patterns A
        self._eigenvalues: Optional[np.ndarray] = None
        self._classes: Optional[np.ndarray] = None
        self._n_classes: int = 0
        
        # Multi-class CSP storage
        self._filters_per_class: Dict[int, np.ndarray] = {}
        self._patterns_per_class: Dict[int, np.ndarray] = {}
    
    # =========================================================================
    # PROPERTY IMPLEMENTATIONS
    # =========================================================================
    
    @property
    def name(self) -> str:
        """Extractor name."""
        return "csp"
    
    @property
    def is_trainable(self) -> bool:
        """CSP requires fitting to learn spatial filters."""
        return True
    
    @property
    def n_components(self) -> int:
        """Number of CSP components."""
        return self._n_components
    
    @n_components.setter
    def n_components(self, value: int) -> None:
        """Set number of components."""
        if value < 2:
            raise ValueError("n_components must be at least 2")
        self._n_components = value
        self._is_fitted = False  # Need to refit
    
    # =========================================================================
    # FITTING IMPLEMENTATION
    # =========================================================================
    
    def _fit_implementation(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           **kwargs) -> None:
        """
        Fit CSP spatial filters on training data.
        
        Args:
            X: Training data, shape (n_trials, n_channels, n_samples)
            y: Class labels, shape (n_trials,)
        """
        self._classes = np.unique(y)
        self._n_classes = len(self._classes)
        
        if self._n_classes < 2:
            raise ValueError(
                f"CSP requires at least 2 classes, got {self._n_classes}"
            )
        
        if self._n_classes == 2:
            # Binary CSP
            self._fit_binary(X, y)
        else:
            # Multi-class CSP using One-vs-Rest
            self._fit_multiclass_ovr(X, y)
        
        # Update feature count
        self._n_features_value = self._filters.shape[0]
        
        logger.info(
            f"CSP fitted: {self._n_classes} classes, "
            f"{self._n_features_value} components, "
            f"filters shape {self._filters.shape}"
        )
    
    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit binary CSP.
        
        Args:
            X: Training data, shape (n_trials, n_channels, n_samples)
            y: Binary labels
        """
        class_0, class_1 = self._classes[0], self._classes[1]
        
        # Get data for each class
        X_0 = X[y == class_0]
        X_1 = X[y == class_1]
        
        # Compute covariance matrices
        C_0 = self._compute_class_covariance(X_0)
        C_1 = self._compute_class_covariance(X_1)
        
        # Solve generalized eigenvalue problem
        self._filters, self._eigenvalues = self._solve_csp(C_0, C_1)
        
        # Compute spatial patterns (for visualization)
        self._patterns = self._compute_patterns(C_0 + C_1)
    
    def _fit_multiclass_ovr(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit multi-class CSP using One-vs-Rest approach.
        
        For each class k:
        - Positive class: samples from class k
        - Negative class: samples from all other classes
        
        Args:
            X: Training data, shape (n_trials, n_channels, n_samples)
            y: Multi-class labels
        """
        all_filters = []
        all_patterns = []
        
        # Components per class (distribute evenly)
        n_comp_per_class = max(2, self._n_components // self._n_classes)
        
        for class_k in self._classes:
            # One-vs-Rest: class k vs all others
            y_binary = (y == class_k).astype(int)
            
            X_pos = X[y_binary == 1]  # Class k
            X_neg = X[y_binary == 0]  # All others
            
            # Compute covariance
            C_pos = self._compute_class_covariance(X_pos)
            C_neg = self._compute_class_covariance(X_neg)
            
            # Solve CSP for this binary problem
            filters, eigenvalues = self._solve_csp(
                C_pos, C_neg, n_components=n_comp_per_class
            )
            
            # Compute patterns
            patterns = self._compute_patterns(C_pos + C_neg, filters)
            
            # Store per-class filters
            self._filters_per_class[int(class_k)] = filters
            self._patterns_per_class[int(class_k)] = patterns
            
            all_filters.append(filters)
            all_patterns.append(patterns)
        
        # Combine all filters
        self._filters = np.vstack(all_filters)
        self._patterns = np.vstack(all_patterns)
        
        logger.debug(
            f"Multi-class CSP: {self._n_classes} classes, "
            f"{n_comp_per_class} components each, "
            f"total {self._filters.shape[0]} filters"
        )
    
    def _compute_class_covariance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute average covariance matrix for a class.
        
        Args:
            X: Data for one class, shape (n_trials, n_channels, n_samples)
        
        Returns:
            Covariance matrix, shape (n_channels, n_channels)
        """
        n_trials = X.shape[0]
        
        # Compute covariance for each trial
        covs = []
        for trial in range(n_trials):
            trial_data = X[trial]  # (n_channels, n_samples)
            cov = np.cov(trial_data)  # (n_channels, n_channels)
            
            # Normalize by trace if requested
            if self._norm_trace:
                cov = cov / np.trace(cov)
            
            covs.append(cov)
        
        # Average covariance
        avg_cov = np.mean(covs, axis=0)
        
        # Apply regularization
        if self._reg > 0:
            avg_cov = avg_cov + self._reg * np.trace(avg_cov) * np.eye(avg_cov.shape[0])
        
        return avg_cov
    
    def _solve_csp(self,
                   C_0: np.ndarray,
                   C_1: np.ndarray,
                   n_components: Optional[int] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the CSP generalized eigenvalue problem.
        
        Solves: C_0 * W = (C_0 + C_1) * W * D
        
        Args:
            C_0: Covariance matrix for class 0
            C_1: Covariance matrix for class 1
            n_components: Number of components (uses self._n_components if None)
        
        Returns:
            Tuple of (filters, eigenvalues)
        """
        n_comp = n_components or self._n_components
        
        # Composite covariance
        C_composite = C_0 + C_1
        
        # Solve generalized eigenvalue problem
        # C_0 * W = C_composite * W * D
        try:
            eigenvalues, eigenvectors = linalg.eigh(C_0, C_composite)
        except np.linalg.LinAlgError:
            # If direct solution fails, use pseudo-inverse
            logger.warning("Using pseudo-inverse for CSP due to singular matrix")
            C_composite_inv = np.linalg.pinv(C_composite)
            eigenvalues, eigenvectors = np.linalg.eigh(C_composite_inv @ C_0)
        
        # Sort by eigenvalue (descending)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        # Select components from both ends
        # (highest eigenvalues = best for class 0, lowest = best for class 1)
        n_first = n_comp // 2
        n_last = n_comp - n_first
        
        selected_idx = np.concatenate([
            np.arange(n_first),  # Top eigenvectors (for class 0)
            np.arange(-n_last, 0)  # Bottom eigenvectors (for class 1)
        ])
        
        filters = eigenvectors[:, selected_idx].T  # (n_components, n_channels)
        selected_eigenvalues = eigenvalues[selected_idx]
        
        return filters, selected_eigenvalues
    
    def _compute_patterns(self,
                         C_composite: np.ndarray,
                         filters: Optional[np.ndarray] = None
                         ) -> np.ndarray:
        """
        Compute spatial patterns from filters.
        
        Patterns A = C * W * inv(W^T * C * W)
        These are the "activation patterns" useful for interpretation.
        
        Args:
            C_composite: Composite covariance matrix
            filters: Spatial filters (uses self._filters if None)
        
        Returns:
            Spatial patterns, shape (n_components, n_channels)
        """
        W = filters if filters is not None else self._filters
        
        if W is None:
            return None
        
        # A = C * W * inv(W^T * C * W)
        CW = C_composite @ W.T
        WCW = W @ CW
        
        try:
            patterns = CW @ np.linalg.inv(WCW)
        except np.linalg.LinAlgError:
            patterns = CW @ np.linalg.pinv(WCW)
        
        return patterns.T  # (n_components, n_channels)
    
    # =========================================================================
    # EXTRACTION IMPLEMENTATION
    # =========================================================================
    
    def _extract_implementation(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Extract CSP features from data.
        
        Args:
            X: Input data, shape (n_trials, n_channels, n_samples)
        
        Returns:
            Features, shape (n_trials, n_components)
        """
        n_trials = X.shape[0]
        n_features = self._filters.shape[0]
        
        features = np.zeros((n_trials, n_features))
        
        for trial_idx in range(n_trials):
            trial_data = X[trial_idx]  # (n_channels, n_samples)
            
            # Apply spatial filters
            # Z = W * X, shape (n_components, n_samples)
            Z = self._filters @ trial_data
            
            # Compute variance of each component
            variances = np.var(Z, axis=1)
            
            # Normalize variance (divide by sum)
            variances = variances / np.sum(variances)
            
            # Apply log transform
            if self._log:
                # Add small epsilon to avoid log(0)
                features[trial_idx] = np.log(variances + 1e-10)
            else:
                features[trial_idx] = variances
        
        return features
    
    # =========================================================================
    # FEATURE NAMES
    # =========================================================================
    
    def _generate_feature_names(self) -> List[str]:
        """Generate descriptive feature names."""
        names = []
        
        if self._n_classes == 2:
            # Binary CSP: first half for class 0, second half for class 1
            n_half = self._n_components // 2
            class_0, class_1 = self._classes[0], self._classes[1]
            
            for i in range(n_half):
                names.append(f"csp_class{class_0}_comp{i}")
            for i in range(self._n_components - n_half):
                names.append(f"csp_class{class_1}_comp{i}")
        else:
            # Multi-class OVR
            idx = 0
            for class_k in self._classes:
                n_comp_k = self._filters_per_class.get(int(class_k), np.array([])).shape[0]
                for i in range(n_comp_k):
                    names.append(f"csp_class{class_k}_comp{i}")
                    idx += 1
        
        return names
    
    # =========================================================================
    # PARAMETER HANDLING
    # =========================================================================
    
    def _get_params_implementation(self) -> Dict[str, Any]:
        """Get CSP-specific parameters."""
        return {
            'n_components': self._n_components,
            'reg': self._reg,
            'log': self._log,
            'norm_trace': self._norm_trace,
        }
    
    def _set_params_implementation(self, **params) -> None:
        """Set CSP-specific parameters."""
        if 'n_components' in params:
            self._n_components = params['n_components']
            self._is_fitted = False
        if 'reg' in params:
            self._reg = params['reg']
        if 'log' in params:
            self._log = params['log']
        if 'norm_trace' in params:
            self._norm_trace = params['norm_trace']
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def _get_fitted_state(self) -> Dict[str, Any]:
        """Get fitted state for serialization."""
        state = {
            'filters': self._filters.tolist() if self._filters is not None else None,
            'patterns': self._patterns.tolist() if self._patterns is not None else None,
            'eigenvalues': self._eigenvalues.tolist() if self._eigenvalues is not None else None,
            'classes': self._classes.tolist() if self._classes is not None else None,
            'n_classes': self._n_classes,
            'filters_per_class': {
                str(k): v.tolist() for k, v in self._filters_per_class.items()
            },
            'patterns_per_class': {
                str(k): v.tolist() for k, v in self._patterns_per_class.items()
            },
        }
        return state
    
    def _load_fitted_state(self, state: Dict[str, Any]) -> None:
        """Load fitted state from serialization."""
        self._filters = np.array(state['filters']) if state.get('filters') else None
        self._patterns = np.array(state['patterns']) if state.get('patterns') else None
        self._eigenvalues = np.array(state['eigenvalues']) if state.get('eigenvalues') else None
        self._classes = np.array(state['classes']) if state.get('classes') else None
        self._n_classes = state.get('n_classes', 0)
        
        self._filters_per_class = {
            int(k): np.array(v) for k, v in state.get('filters_per_class', {}).items()
        }
        self._patterns_per_class = {
            int(k): np.array(v) for k, v in state.get('patterns_per_class', {}).items()
        }
    
    # =========================================================================
    # PUBLIC METHODS FOR ANALYSIS
    # =========================================================================
    
    def get_spatial_filters(self) -> Optional[np.ndarray]:
        """
        Get the learned spatial filters.
        
        Returns:
            Spatial filters W, shape (n_components, n_channels)
            or None if not fitted
        
        Note:
            Filters are used for projection: Z = W @ X
        """
        return self._filters.copy() if self._filters is not None else None
    
    def get_spatial_patterns(self) -> Optional[np.ndarray]:
        """
        Get the spatial patterns (activation patterns).
        
        Returns:
            Spatial patterns A, shape (n_components, n_channels)
            or None if not fitted
        
        Note:
            Patterns show which brain regions contribute to each component.
            Use these for topographic visualization, NOT the filters.
        """
        return self._patterns.copy() if self._patterns is not None else None
    
    def get_eigenvalues(self) -> Optional[np.ndarray]:
        """
        Get the eigenvalues (discrimination power).
        
        Returns:
            Eigenvalues, shape (n_components,)
            or None if not fitted
        
        Note:
            Higher eigenvalues = better discrimination for class 0
            Lower eigenvalues = better discrimination for class 1
        """
        return self._eigenvalues.copy() if self._eigenvalues is not None else None
    
    def get_filters_for_class(self, class_label: int) -> Optional[np.ndarray]:
        """
        Get spatial filters for a specific class (multi-class CSP).
        
        Args:
            class_label: Class label
        
        Returns:
            Filters for that class, or None if not found
        """
        return self._filters_per_class.get(class_label, None)
    
    def transform_single_trial(self,
                               trial: np.ndarray,
                               return_projected: bool = False
                               ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Transform a single trial with detailed output.
        
        Args:
            trial: Single trial, shape (n_channels, n_samples)
            return_projected: If True, also return projected signals
        
        Returns:
            Features (and optionally projected signals)
        """
        if self._filters is None:
            raise RuntimeError("CSP not fitted. Call fit() first.")
        
        # Project through filters
        projected = self._filters @ trial  # (n_components, n_samples)
        
        # Compute features
        variances = np.var(projected, axis=1)
        variances = variances / np.sum(variances)
        
        if self._log:
            features = np.log(variances + 1e-10)
        else:
            features = variances
        
        if return_projected:
            return features, projected
        return features


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_csp_extractor(n_components: int = 6,
                         reg: float = 0.0,
                         log: bool = True,
                         norm_trace: bool = True,
                         **kwargs) -> CSPExtractor:
    """
    Factory function to create CSP extractor.
    
    Args:
        n_components: Number of CSP components
        reg: Regularization parameter
        log: Apply log transform
        norm_trace: Normalize covariance by trace
        **kwargs: Additional configuration
    
    Returns:
        Configured CSPExtractor instance
    """
    extractor = CSPExtractor(
        n_components=n_components,
        reg=reg,
        log=log,
        norm_trace=norm_trace
    )
    
    if kwargs:
        extractor.initialize(kwargs)
    
    return extractor
