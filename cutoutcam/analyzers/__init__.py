from typing import TYPE_CHECKING

from .face_analyzer import FaceAnalyzer
from .pose_analyzer import PoseAnalyzer

try:
    from .hand_analyzer import HandAnalyzer
except ImportError:
    if TYPE_CHECKING:
        from .hand_analyzer import HandAnalyzer as HandAnalyzer
    else:
        HandAnalyzer = None

__all__ = ["FaceAnalyzer", "HandAnalyzer", "PoseAnalyzer"]
