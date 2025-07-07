"""
Model compatibility checker for fused kernels.

This module provides utilities to determine if a model is compatible with
fused kernels based on weight distribution analysis.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LayerMetrics:
    """Metrics for a single layer."""
    layer_idx: int
    max_k_weight: float
    max_q_weight: float
    median_k_weight: float
    median_q_weight: float
    k_weight_ratio: float
    q_weight_ratio: float
    has_extreme_weights: bool


@dataclass
class CompatibilityResult:
    """Result of compatibility check."""
    status: str  # "COMPATIBLE", "WARNING", "INCOMPATIBLE"
    score: float  # 0.0 to 1.0
    reason: str
    layer_metrics: List[LayerMetrics]
    estimated_amplification: float
    recommendations: List[str]


class FusedKernelCompatibilityChecker:
    """Check if a model is compatible with fused kernels."""
    
    # Thresholds based on Qwen3-0.6B analysis
    MAX_SAFE_WEIGHT_RATIO = 20.0
    EXTREME_WEIGHT_THRESHOLD = 10.0
    MAX_SAFE_AMPLIFICATION = 10.0
    WARNING_WEIGHT_RATIO = 15.0
    
    # Typical fused kernel error
    EXPECTED_KERNEL_ERROR = 0.001
    
    def __init__(self):
        self.cache = {}
    
    def check_model(self, model: nn.Module, model_name: Optional[str] = None) -> CompatibilityResult:
        """
        Check if a model is compatible with fused kernels.
        
        Args:
            model: The model to check
            model_name: Optional model name for caching
            
        Returns:
            CompatibilityResult with detailed analysis
        """
        # Check cache
        if model_name and model_name in self.cache:
            logger.info(f"Using cached compatibility result for {model_name}")
            return self.cache[model_name]
        
        # Analyze model layers
        layer_metrics = self._analyze_layers(model)
        
        # Calculate compatibility score
        score, status = self._calculate_compatibility_score(layer_metrics)
        
        # Estimate error amplification
        amplification = self._estimate_amplification(layer_metrics)
        
        # Generate reason and recommendations
        reason = self._generate_reason(layer_metrics, amplification)
        recommendations = self._generate_recommendations(status, layer_metrics)
        
        result = CompatibilityResult(
            status=status,
            score=score,
            reason=reason,
            layer_metrics=layer_metrics,
            estimated_amplification=amplification,
            recommendations=recommendations
        )
        
        # Cache result
        if model_name:
            self.cache[model_name] = result
        
        return result
    
    def _analyze_layers(self, model: nn.Module) -> List[LayerMetrics]:
        """Analyze normalization weights in model layers."""
        metrics = []
        
        # Find all decoder layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            logger.warning("Could not find model layers for analysis")
            return metrics
        
        for idx, layer in enumerate(layers):
            # Check if layer has K/Q normalization
            if not hasattr(layer, 'self_attn'):
                continue
                
            attn = layer.self_attn
            
            # Analyze K norm weights
            k_metrics = self._analyze_norm_weights(attn, 'k_norm')
            q_metrics = self._analyze_norm_weights(attn, 'q_norm')
            
            if k_metrics is None and q_metrics is None:
                continue
            
            # Default values if one is missing
            if k_metrics is None:
                k_metrics = (1.0, 1.0, 1.0)
            if q_metrics is None:
                q_metrics = (1.0, 1.0, 1.0)
            
            layer_metric = LayerMetrics(
                layer_idx=idx,
                max_k_weight=k_metrics[0],
                median_k_weight=k_metrics[1],
                k_weight_ratio=k_metrics[2],
                max_q_weight=q_metrics[0],
                median_q_weight=q_metrics[1],
                q_weight_ratio=q_metrics[2],
                has_extreme_weights=(
                    k_metrics[0] > self.EXTREME_WEIGHT_THRESHOLD or
                    q_metrics[0] > self.EXTREME_WEIGHT_THRESHOLD
                )
            )
            
            metrics.append(layer_metric)
        
        return metrics
    
    def _analyze_norm_weights(self, module: nn.Module, norm_name: str) -> Optional[Tuple[float, float, float]]:
        """Analyze normalization weights for a specific norm layer."""
        if not hasattr(module, norm_name):
            return None
            
        norm_layer = getattr(module, norm_name)
        if not hasattr(norm_layer, 'weight'):
            return None
            
        weights = norm_layer.weight.detach().cpu()
        
        max_weight = weights.max().item()
        median_weight = weights.median().item()
        
        # Avoid division by zero
        if median_weight == 0:
            ratio = float('inf') if max_weight != 0 else 1.0
        else:
            ratio = max_weight / median_weight
        
        return max_weight, median_weight, ratio
    
    def _calculate_compatibility_score(self, metrics: List[LayerMetrics]) -> Tuple[float, str]:
        """Calculate overall compatibility score and status."""
        if not metrics:
            return 1.0, "COMPATIBLE"
        
        # Calculate component scores
        weight_ratio_score = self._calculate_weight_ratio_score(metrics)
        extreme_layer_score = self._calculate_extreme_layer_score(metrics)
        max_weight_score = self._calculate_max_weight_score(metrics)
        
        # Weighted combination
        score = (
            weight_ratio_score * 0.4 +
            extreme_layer_score * 0.3 +
            max_weight_score * 0.3
        )
        
        # Determine status
        if score < 0.5:
            status = "INCOMPATIBLE"
        elif score < 0.8:
            status = "WARNING"
        else:
            status = "COMPATIBLE"
        
        return score, status
    
    def _calculate_weight_ratio_score(self, metrics: List[LayerMetrics]) -> float:
        """Score based on weight ratios."""
        max_ratio = max(m.k_weight_ratio for m in metrics)
        
        if max_ratio > self.MAX_SAFE_WEIGHT_RATIO:
            return 0.0
        elif max_ratio > self.WARNING_WEIGHT_RATIO:
            return 0.5 * (self.MAX_SAFE_WEIGHT_RATIO - max_ratio) / (
                self.MAX_SAFE_WEIGHT_RATIO - self.WARNING_WEIGHT_RATIO
            )
        else:
            return 1.0
    
    def _calculate_extreme_layer_score(self, metrics: List[LayerMetrics]) -> float:
        """Score based on number of layers with extreme weights."""
        extreme_count = sum(1 for m in metrics if m.has_extreme_weights)
        extreme_ratio = extreme_count / len(metrics)
        
        if extreme_ratio > 0.5:
            return 0.0
        elif extreme_ratio > 0.3:
            return 1.0 - 2 * (extreme_ratio - 0.3)
        else:
            return 1.0
    
    def _calculate_max_weight_score(self, metrics: List[LayerMetrics]) -> float:
        """Score based on maximum weight values."""
        max_weight = max(m.max_k_weight for m in metrics)
        
        if max_weight > 50.0:
            return 0.0
        elif max_weight > 20.0:
            return (50.0 - max_weight) / 30.0
        else:
            return 1.0
    
    def _estimate_amplification(self, metrics: List[LayerMetrics]) -> float:
        """Estimate cumulative error amplification through layers."""
        amplification = 1.0
        error = self.EXPECTED_KERNEL_ERROR
        
        for metric in metrics:
            if metric.has_extreme_weights:
                # Use geometric mean of amplification
                layer_amp = (metric.max_k_weight + metric.max_q_weight) / 2.0
                amplification *= (1.0 + error * layer_amp)
        
        return amplification
    
    def _generate_reason(self, metrics: List[LayerMetrics], amplification: float) -> str:
        """Generate human-readable reason for compatibility result."""
        extreme_layers = [m for m in metrics if m.has_extreme_weights]
        
        if not extreme_layers:
            return "Model weights are within normal ranges."
        
        max_weight = max(m.max_k_weight for m in metrics)
        
        parts = []
        parts.append(f"Found {len(extreme_layers)}/{len(metrics)} layers with extreme weights")
        parts.append(f"Maximum weight: {max_weight:.1f}")
        parts.append(f"Estimated error amplification: {amplification:.1f}x")
        
        return "; ".join(parts)
    
    def _generate_recommendations(self, status: str, metrics: List[LayerMetrics]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if status == "INCOMPATIBLE":
            recommendations.append("Use standard kernels only")
            recommendations.append("Consider model quantization or pruning to reduce weight magnitudes")
        elif status == "WARNING":
            recommendations.append("Test thoroughly before using fused kernels in production")
            recommendations.append("Monitor generation quality for anomalies")
        else:
            recommendations.append("Model is compatible with fused kernels")
        
        # Specific layer recommendations
        extreme_layers = [m for m in metrics if m.has_extreme_weights]
        if extreme_layers:
            layer_indices = [m.layer_idx for m in extreme_layers[:3]]
            recommendations.append(
                f"Layers {layer_indices} have particularly extreme weights"
            )
        
        return recommendations


def check_model_compatibility(
    model: nn.Module,
    model_name: Optional[str] = None,
    use_custom_kernels: bool = True
) -> Tuple[bool, Optional[CompatibilityResult]]:
    """
    Quick compatibility check for model loading.
    
    Args:
        model: The model to check
        model_name: Optional model name for caching
        use_custom_kernels: Whether custom kernels are requested
        
    Returns:
        (can_use_custom_kernels, compatibility_result)
    """
    if not use_custom_kernels:
        return False, None
    
    checker = FusedKernelCompatibilityChecker()
    result = checker.check_model(model, model_name)
    
    if result.status == "INCOMPATIBLE":
        logger.error(
            f"Model incompatible with fused kernels: {result.reason}\n"
            f"Recommendations: {'; '.join(result.recommendations)}"
        )
        return False, result
    elif result.status == "WARNING":
        logger.warning(
            f"Fused kernels may cause issues: {result.reason}\n"
            f"Recommendations: {'; '.join(result.recommendations)}"
        )
        return True, result
    else:
        logger.info("Model is compatible with fused kernels")
        return True, result


def generate_compatibility_report(result: CompatibilityResult, model_name: str = "Unknown") -> str:
    """Generate a detailed compatibility report."""
    lines = []
    lines.append("Model Compatibility Report")
    lines.append("=" * 50)
    lines.append(f"Model: {model_name}")
    lines.append(f"Status: {result.status}")
    lines.append(f"Score: {result.score:.2f}")
    lines.append("")
    
    lines.append("Issues Found:")
    for metric in result.layer_metrics:
        if metric.has_extreme_weights:
            lines.append(
                f"- Layer {metric.layer_idx}: "
                f"K norm max = {metric.max_k_weight:.1f}, "
                f"Q norm max = {metric.max_q_weight:.1f}"
            )
    
    lines.append("")
    lines.append(f"Estimated error amplification: {result.estimated_amplification:.1f}x")
    lines.append("")
    
    lines.append("Recommendations:")
    for rec in result.recommendations:
        lines.append(f"- {rec}")
    
    return "\n".join(lines)