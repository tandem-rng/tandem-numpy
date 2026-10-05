"""NumPy BitGenerator and Generator for Tandem8x32."""

from ._generator import TandemGenerator
from ._tandem import ChoiceTable, Tandem

__all__ = ["ChoiceTable", "Tandem", "TandemGenerator"]
