"""Default configuration and CV pipeline constants for BioPalm Auth."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TEMPLATES_DIR = DATA_DIR / "templates"
USERS_FILE = DATA_DIR / "users.json"
STATIC_DIR = BASE_DIR / "static"

# Stage 2: Normalization
IMAGE_SIZE = 256
DEFAULT_CENTER_CROP_RATIO = 0.75

# Stage 3: Contrast enhancement & smoothing
DEFAULT_USE_BLACKHAT = True
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID = (8, 8)
MEDIAN_KSIZE = 5
GAUSSIAN_KSIZE = 5
BLACKHAT_KSIZE = 15

# Stage 4: Adaptive segmentation
ADAPTIVE_BLOCK_SIZE = 25
ADAPTIVE_C = 9

# Stage 5: Morphology & component cleanup
MORPH_OPEN_KSIZE = 3
MORPH_CLOSE_KSIZE = 3
MIN_COMPONENT_PIXELS = 25

# Storage & Enrollment
MAX_ENROLL_SAMPLES = 3

# Stage 6: Feature matching
DEFAULT_THRESHOLD = 0.80
DEFAULT_DILATION_KSIZE = 9
DEFAULT_SHIFT_RANGE = 12
DEFAULT_SHIFT_STEP = 3
MIN_SKELETON_RATIO = 0.004

# Lighting quality thresholds (0-255 scale)
LIGHTING_DARK_MAX = 55.0
LIGHTING_BRIGHT_MIN = 215.0
