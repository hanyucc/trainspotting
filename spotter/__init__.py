"""Spotter: object-centric 3D reconstruction from found photos (Trainspotting)."""

from .load import load_spotter
from .models import Spotter

__version__ = "1.0.0"

__all__ = ["Spotter", "load_spotter", "__version__"]
