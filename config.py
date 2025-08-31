import os

# Create database directory
DB_DIR = os.path.abspath('face_database')
os.makedirs(DB_DIR, exist_ok=True)

# Database path
DB_PATH = os.path.join(DB_DIR, 'faces.sqlite')

# Configuration
MIN_FACE_SIZE = 60  # Minimum size of face to consider
TRACKING_THRESHOLD = 0.5  # IOU threshold for tracking
DETECTION_SIZE = (640, 640)  # InsightFace detection size
RECOGNITION_THRESHOLD = 0.85  # Similarity threshold for face matching
EVALUATION_FRAMES = 10  # Number of frames to evaluate before determining identity
DEBUG_MODE = True  # Enable debug output
DEFAULT_DISPLAY_CONFIDENCE = 0.01  # Default confidence to display when there's no match

# Quality thresholds
MIN_BRIGHTNESS = 65  # Minimum brightness for face quality
MAX_BRIGHTNESS = 165  # Maximum brightness for face quality
MIN_BLUR_SCORE = 20  # Minimum blur score for face quality
MIN_FACE_DIMENSION = 80  # Minimum face dimension for registration
MIN_ASPECT_RATIO = 0.5  # Minimum aspect ratio for face proportions
MAX_ASPECT_RATIO = 1.5  # Maximum aspect ratio for face proportions

# Recognition thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.90  # High confidence threshold for matches
REGISTRATION_SIMILARITY_THRESHOLD = 0.70  # Threshold for preventing duplicate registrations
SIMILARITY_CHANGE_THRESHOLD = 0.01  # Threshold for significant similarity changes

# Timing and intervals
FPS_UPDATE_INTERVAL = 30  # Update FPS every N frames
REGISTRATION_COOLDOWN = 2.0  # Cooldown between registrations (seconds)
ERROR_DISPLAY_DURATION = 1500  # Duration to display error messages (ms)

# Feature extraction parameters
MAX_EMBEDDINGS_PER_PERSON = 30  # Maximum embeddings stored per person
GLOBAL_HIST_BINS = 32  # Number of bins for global histogram
REGIONAL_HIST_BINS = 16  # Number of bins for regional histograms
EDGE_HIST_BINS = 32  # Number of bins for edge histogram
CANNY_LOW_THRESHOLD = 100  # Canny edge detection low threshold
CANNY_HIGH_THRESHOLD = 200  # Canny edge detection high threshold
LBP_RADIUS = 1  # Local Binary Pattern radius
LBP_POINTS = 8  # Local Binary Pattern points
FEATURE_CACHE_SIZE = 1000  # Maximum number of cached features

# Detection parameters
MIN_DETECTION_CONFIDENCE = 0.5  # Minimum confidence for face detection

# Pose checking parameters
MAX_EYE_ANGLE = 10  # Maximum eye angle for face tilt (degrees)
MIN_EYE_RATIO = 0.25  # Minimum eye distance ratio for frontal face
MAX_NOSE_OFFSET = 0.08  # Maximum nose offset for frontal face
MIN_VERTICAL_RATIO = 0.8  # Minimum vertical ratio (looking up)
MAX_VERTICAL_RATIO = 1.5  # Maximum vertical ratio (looking down)

# Adaptive threshold parameters
FEW_EMBEDDINGS_THRESHOLD = 0.35  # Threshold for few embeddings (< 5)
MEDIUM_EMBEDDINGS_THRESHOLD = 0.20  # Standard threshold (5-15 embeddings)
MANY_EMBEDDINGS_THRESHOLD = 0.15  # Threshold for many embeddings (> 15)
DEFAULT_ADAPTIVE_THRESHOLD = 0.20  # Default adaptive threshold 