"""
Shared configuration constants for the ISPS alignment system.

Centralises thresholds and defaults that are used across multiple modules
so they are defined in exactly one place.

Author : shalindri20@gmail.com
Created: 2025-01
"""

# ---------------------------------------------------------------------------
# Alignment classification thresholds
# ---------------------------------------------------------------------------
# Calibrated against the all-MiniLM-L6-v2 model's typical similarity range
# for hospital-domain documents (unrelated pairs: 0.05–0.20, strongly
# related pairs: 0.45–0.65).

THRESHOLD_EXCELLENT = 0.75   # Near-direct operationalisation of strategy
THRESHOLD_GOOD = 0.60        # Clear strategic support
THRESHOLD_FAIR = 0.45        # Partial or indirect alignment
ORPHAN_THRESHOLD = 0.45      # Below this for ALL objectives → orphan

# ---------------------------------------------------------------------------
# Default hospital name (used when no name is provided in data)
# ---------------------------------------------------------------------------
DEFAULT_HOSPITAL_NAME = "Hospital"
