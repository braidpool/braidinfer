"""
Metrics and monitoring for nano-vllm.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import threading


@dataclass
class InferenceMetrics:
    """Metrics for a single inference request."""
    request_id: str
    is_prefill: bool
    num_sequences: int
    num_tokens: int
    start_time: float
    end_time: Optional[float] = None
    prefill_time: Optional[float] = None
    decode_time: Optional[float] = None
    tokens_per_second: Optional[float] = None
    error: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        if self.end_time:
            return self.end_time - self.start_time
        return None


class MetricsCollector:
    """Collects and aggregates inference metrics."""
    
    def __init__(self):
        self._metrics: List[InferenceMetrics] = []
        self._current_requests: Dict[str, InferenceMetrics] = {}
        self._counters = defaultdict(int)
        self._lock = threading.Lock()
        
        # Performance tracking
        self._prefill_times: List[float] = []
        self._decode_times: List[float] = []
        self._tokens_per_second: List[float] = []
    
    def start_request(self, request_id: str, is_prefill: bool, 
                     num_sequences: int, num_tokens: int) -> InferenceMetrics:
        """Start tracking a new inference request."""
        metrics = InferenceMetrics(
            request_id=request_id,
            is_prefill=is_prefill,
            num_sequences=num_sequences,
            num_tokens=num_tokens,
            start_time=time.time()
        )
        
        with self._lock:
            self._current_requests[request_id] = metrics
            self._counters['total_requests'] += 1
            if is_prefill:
                self._counters['prefill_requests'] += 1
            else:
                self._counters['decode_requests'] += 1
                
        return metrics
    
    def end_request(self, request_id: str, error: Optional[str] = None):
        """End tracking for an inference request."""
        with self._lock:
            if request_id not in self._current_requests:
                return
                
            metrics = self._current_requests.pop(request_id)
            metrics.end_time = time.time()
            metrics.error = error
            
            if error:
                self._counters['failed_requests'] += 1
            else:
                self._counters['successful_requests'] += 1
                
                # Calculate performance metrics
                duration = metrics.duration
                if duration and duration > 0:
                    if metrics.is_prefill:
                        metrics.prefill_time = duration
                        self._prefill_times.append(duration)
                    else:
                        metrics.decode_time = duration
                        self._decode_times.append(duration)
                    
                    if metrics.num_tokens > 0:
                        metrics.tokens_per_second = metrics.num_tokens / duration
                        self._tokens_per_second.append(metrics.tokens_per_second)
            
            self._metrics.append(metrics)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self._lock:
            summary = {
                'total_requests': self._counters['total_requests'],
                'successful_requests': self._counters['successful_requests'],
                'failed_requests': self._counters['failed_requests'],
                'prefill_requests': self._counters['prefill_requests'],
                'decode_requests': self._counters['decode_requests'],
                'active_requests': len(self._current_requests)
            }
            
            # Performance statistics
            if self._prefill_times:
                summary['avg_prefill_time'] = sum(self._prefill_times) / len(self._prefill_times)
                summary['min_prefill_time'] = min(self._prefill_times)
                summary['max_prefill_time'] = max(self._prefill_times)
                
            if self._decode_times:
                summary['avg_decode_time'] = sum(self._decode_times) / len(self._decode_times)
                summary['min_decode_time'] = min(self._decode_times)
                summary['max_decode_time'] = max(self._decode_times)
                
            if self._tokens_per_second:
                summary['avg_tokens_per_second'] = sum(self._tokens_per_second) / len(self._tokens_per_second)
                summary['max_tokens_per_second'] = max(self._tokens_per_second)
                
            return summary
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._current_requests.clear()
            self._counters.clear()
            self._prefill_times.clear()
            self._decode_times.clear()
            self._tokens_per_second.clear()


# Global metrics collector instance
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics_collector


class MetricsContext:
    """Context manager for tracking inference metrics."""
    
    def __init__(self, request_id: str, is_prefill: bool, 
                 num_sequences: int, num_tokens: int):
        self.request_id = request_id
        self.is_prefill = is_prefill
        self.num_sequences = num_sequences
        self.num_tokens = num_tokens
        self.metrics = None
        
    def __enter__(self):
        self.metrics = get_metrics_collector().start_request(
            self.request_id, self.is_prefill, 
            self.num_sequences, self.num_tokens
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        error = str(exc_val) if exc_val else None
        get_metrics_collector().end_request(self.request_id, error)
        return False