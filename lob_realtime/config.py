"""Configuration for the real-time LOB feature engine."""

EPSILON = 1e-10
N_LEVELS = 10
WINDOW_SIZE = 5  # Match existing pipeline's buffer_size
STANDARD_ORDER_SIZE = 1000.0  # For price impact estimation

# OI weights: 1/i for level i
OI_WEIGHTS = tuple(1.0 / i for i in range(1, N_LEVELS + 1))
OI_WEIGHT_SUM = sum(OI_WEIGHTS)
