"""Round 1 typed-contract package.

Only data contracts, canonical serialization, validation and provenance are
allowed in this package at this stage. Numerical optimization is intentionally
absent.
"""

from ._version import CONTRACT_VERSION, __version__

__all__ = ["CONTRACT_VERSION", "__version__"]
