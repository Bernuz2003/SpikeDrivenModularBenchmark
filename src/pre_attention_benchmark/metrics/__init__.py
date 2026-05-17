# Re-export intenzionale: gli script usano questo modulo come facciata leggera.
from .collector import MetricsCollector, summarize_layer_metrics
from .pareto import pareto_front
