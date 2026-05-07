"""
Model3: EGO-Planner + TD3 hybrid for UAV navigation.

Combines EGO-Planner (ESDF-free gradient-based trajectory optimization)
with TD3 reinforcement learning for urban drone logistics path planning.
"""

from .ego_planner import EGOPlanner, BSplineTrajectory
from .env import EGOUAVEnv
from .network import EGOFeatureExtractor, CustomCombinedExtractor

__all__ = [
    'EGOPlanner',
    'BSplineTrajectory',
    'EGOUAVEnv',
    'EGOFeatureExtractor',
    'CustomCombinedExtractor',
]
