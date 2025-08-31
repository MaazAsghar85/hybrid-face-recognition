import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.model_zoo import get_model
from skimage.feature import local_binary_pattern
from config import (
    DETECTION_SIZE, GLOBAL_HIST_BINS, REGIONAL_HIST_BINS, EDGE_HIST_BINS,
    CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD, LBP_RADIUS, LBP_POINTS,
    FEATURE_CACHE_SIZE, MIN_DETECTION_CONFIDENCE
)
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def init_face_detector():
    """Initialize the InsightFace detector (CPU only)"""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Initializing InsightFace detector (attempt {attempt + 1}/{max_retries})")
            face_analyzer = FaceAnalysis(
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection', 'recognition'],  # Enable both detection and recognition
                name='buffalo_l'  # Use the larger model for better accuracy
            )
            face_analyzer.prepare(ctx_id=-1, det_size=DETECTION_SIZE)
            logger.info("InsightFace detector initialized successfully")
            return face_analyzer
        except Exception as e:
            logger.error(f"Error initializing InsightFace (attempt {attempt + 1}): {str(e)}", exc_info=True)
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.critical("Failed to initialize InsightFace after multiple attempts")
                return None

# Cache for feature calculations
_feature_cache = {}
_cache_size = FEATURE_CACHE_SIZE  # Maximum number of cached features

def _clear_old_cache():
    """Clear old entries from the feature cache"""
    if len(_feature_cache) > _cache_size:
        # Remove oldest entries
        oldest_keys = sorted(_feature_cache.keys(), key=lambda k: _feature_cache[k]['timestamp'])[:len(_feature_cache) - _cache_size]
        for key in oldest_keys:
            del _feature_cache[key]

def calculate_face_features(face_img):
    """Calculate feature vector from a face image with recovery mechanism"""
    if face_img is None or face_img.size == 0:
        logger.error("Invalid face image provided for feature calculation")
        return None
        
    try:
        # Create cache key from image hash
        img_hash = hash(face_img.tobytes())
        if img_hash in _feature_cache:
            logger.debug("Retrieved features from cache")
            return _feature_cache[img_hash]['features']
        
        # Resize to standard size
        face_resized = cv2.resize(face_img, (64, 64))
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Normalize lighting
        gray = cv2.equalizeHist(gray)
        
        # Calculate features in parallel where possible
        features = []
        
        # Global histogram (faster calculation with fewer bins)
        global_hist = cv2.calcHist([gray], [0], None, [GLOBAL_HIST_BINS], [0, 256])
        global_hist = cv2.normalize(global_hist, global_hist).flatten()
        features.extend(global_hist)
        
        # Regional histograms (optimized calculation)
        h, w = gray.shape
        cell_h, cell_w = h // 3, w // 3
        for i in range(3):
            for j in range(3):
                roi = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                hist = cv2.calcHist([roi], [0], None, [REGIONAL_HIST_BINS], [0, 256])  # Reduced bins for speed
                hist = cv2.normalize(hist, hist).flatten()
                features.extend(hist)
        
        # Edge features (optimized calculation)
        edges = cv2.Canny(gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
        edge_hist = cv2.calcHist([edges], [0], None, [EDGE_HIST_BINS], [0, 256])  # Reduced bins
        edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
        features.extend(edge_hist)
        
        # Texture features using Local Binary Patterns (optimized)
        radius = LBP_RADIUS
        n_points = LBP_POINTS
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2), density=True)
        features.extend(lbp_hist)
        
        # Convert to numpy array and normalize
        features = np.array(features)
        features = features / np.linalg.norm(features)
        
        # Cache the result
        _feature_cache[img_hash] = {
            'features': features,
            'timestamp': time.time()
        }
        _clear_old_cache()
        logger.debug("Calculated and cached new features")
        
        return features
    except Exception as e:
        logger.error(f"Error calculating face features: {str(e)}", exc_info=True)
        logger.info("Attempting to recover by clearing feature cache")
        _feature_cache.clear()
        try:
            features = calculate_face_features(face_img)
            logger.info("Successfully recovered feature calculation")
            return features
        except Exception as e:
            logger.error(f"Recovery failed: {str(e)}", exc_info=True)
            return None

def detect_faces(face_analyzer, frame):
    """Detect faces in a frame using InsightFace with recovery mechanism"""
    if face_analyzer is None:
        logger.warning("Face analyzer not initialized, attempting to reinitialize")
        face_analyzer = init_face_detector()
        if face_analyzer is None:
            logger.error("Failed to reinitialize face analyzer")
            return []
    
    try:
        # Ensure frame is in correct format and size
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame received")
            return []
            
        # Detect faces
        faces = face_analyzer.get(frame)
        num_faces = len(faces) if faces is not None else 0
        logger.debug(f"Detected {num_faces} faces in frame")
        
        if faces is None:
            return []
            
        # Filter out low confidence detections
        valid_faces = []
        for face in faces:
            if hasattr(face, 'det_score') and face.det_score > MIN_DETECTION_CONFIDENCE:  # Adjust threshold if needed
                valid_faces.append(face)
        
        logger.debug(f"Found {len(valid_faces)} valid faces after filtering")
        return valid_faces
        
    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}", exc_info=True)
        logger.info("Attempting to recover face analyzer")
        face_analyzer = init_face_detector()
        if face_analyzer is not None:
            try:
                faces = face_analyzer.get(frame)
                logger.info("Successfully recovered face detection")
                return faces if faces is not None else []
            except Exception as e:
                logger.error(f"Recovery failed: {str(e)}", exc_info=True)
        return [] 