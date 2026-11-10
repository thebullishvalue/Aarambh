"""Aarambh (आरंभ) — walk-forward fair value breadth.

Layout:
    config   every tunable constant
    data     loading, date parsing, numeric coercion, cleaning (no look-ahead)
    stats    mathematical primitives (OU, leaky accumulator, DFA, calibrated z, HAC)
    engine   FairValueEngine: fair value → gap → breadth → conviction → signal → validation

``correl.py`` at the repository root is the Streamlit app built on top of these.
"""

from .config import VERSION

__all__ = ["VERSION"]
