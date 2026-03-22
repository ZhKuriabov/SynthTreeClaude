import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from .estimators import SynthTreeClassifier, SynthTreeRegressor

__all__ = ["SynthTreeClassifier", "SynthTreeRegressor"]
