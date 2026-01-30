"""
Soccer Analytics Hackathon - Source Module
==========================================

Network-based lineup cohesion metric for soccer analytics.

Usage:
    from src.cohesion_metric import PassingNetworkBuilder, CohesionCalculator
"""

from .cohesion_metric import (
    PassingNetworkBuilder,
    CohesionCalculator,
    CohesionScore,
    analyze_match,
    compute_network_metrics,
)

__all__ = [
    'PassingNetworkBuilder',
    'CohesionCalculator', 
    'CohesionScore',
    'analyze_match',
    'compute_network_metrics',
]

__version__ = '0.1.0'
