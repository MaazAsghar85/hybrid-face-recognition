import cv2
import numpy as np
from config import (
    MIN_BRIGHTNESS, MAX_BRIGHTNESS, MIN_BLUR_SCORE, MIN_FACE_DIMENSION,
    MIN_ASPECT_RATIO, MAX_ASPECT_RATIO, FEW_EMBEDDINGS_THRESHOLD,
    MEDIUM_EMBEDDINGS_THRESHOLD, MANY_EMBEDDINGS_THRESHOLD, DEFAULT_ADAPTIVE_THRESHOLD,
    MAX_EYE_ANGLE, MIN_EYE_RATIO, MAX_NOSE_OFFSET, MIN_VERTICAL_RATIO, MAX_VERTICAL_RATIO
)

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    # box format: (x1, y1, x2, y2)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union

def calculate_brightness(img):
    """Calculate the average brightness of an image"""
    if len(img.shape) == 3:
        # Convert to grayscale if the image is in color
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Calculate mean brightness
    mean_brightness = np.mean(gray)
    return mean_brightness

def check_face_pose(face_img, landmarks):
    """
    Check if face is frontal/straight using facial landmarks
    Returns (is_straight, reason)
    """
    if landmarks is None:
        return False, "No landmarks detected"
        
    # Get key landmarks for pose estimation
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]
    
    # Calculate eye angle to check if face is tilted - make more strict
    eye_angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    if abs(eye_angle) > MAX_EYE_ANGLE:  # More strict: 10 degrees instead of 15
        return False, "Face is tilted"
    
    # Check if face is turned sideways using eye distance ratio - make more strict
    eye_distance = np.linalg.norm(right_eye - left_eye)
    face_width = face_img.shape[1]
    eye_ratio = eye_distance / face_width
    if eye_ratio < MIN_EYE_RATIO:  # More strict: 0.25 instead of 0.2
        return False, "Face is not frontal"
    
    # Check nose position relative to eyes (horizontal center) - make more strict
    eye_center = (left_eye + right_eye) / 2
    nose_offset = abs(nose[0] - eye_center[0]) / face_width
    if nose_offset > MAX_NOSE_OFFSET:  # More strict: 0.08 instead of 0.1
        return False, "Face is turned sideways"
        
    # Check vertical angle (looking up/down) - make more strict
    eye_level = (left_eye[1] + right_eye[1]) / 2
    mouth_level = (left_mouth[1] + right_mouth[1]) / 2
    nose_to_eye = nose[1] - eye_level
    nose_to_mouth = mouth_level - nose[1]
    vertical_ratio = nose_to_eye / nose_to_mouth if nose_to_mouth != 0 else float('inf')
    
    # More strict ratios for vertical angle
    if vertical_ratio < MIN_VERTICAL_RATIO:  # Looking up (was 0.7)
        return False, "Face is looking up"
    if vertical_ratio > MAX_VERTICAL_RATIO:  # Looking down (was 1.5)
        return False, "Face is looking down"
    
    return True, None

def check_face_quality(face_img, landmarks=None, for_registration=False):
    """Check various quality metrics of a face image"""
    try:
        # Check brightness (existing check)
        brightness = calculate_brightness(face_img)
        if brightness < MIN_BRIGHTNESS:
            return False, "Too dark"
        if brightness > MAX_BRIGHTNESS:
            return False, "Too bright"
        
        # Check blur - make it stricter for registration
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if blur_value < MIN_BLUR_SCORE:  # Basic check for recognition
            return False, "Too blurry"
            
        # Check face size - stricter for registration
        height, width = face_img.shape[:2]
        min_dimension = min(height, width)
        
        if min_dimension < MIN_FACE_DIMENSION:  # Stricter size requirement for registration
            return False, "Face too small for registration"
            
        # Check face aspect ratio (to detect partial faces)
        aspect_ratio = height / width
        if for_registration:
            if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):  # Stricter aspect ratio for registration
                return False, "Invalid face proportions for registration"
        else:
            if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):  # Basic check for recognition
                return False, "Invalid face proportions"
        
        # Face pose checks only for registration
        if for_registration and landmarks is not None:
            is_straight, pose_reason = check_face_pose(face_img, landmarks)
            if not is_straight:
                return False, f"Poor face pose: {pose_reason}"
        
        return True, None
        
    except Exception as e:
        return False, f"Quality check failed: {str(e)}"

def get_embedding_based_threshold(embedding_count):
    """
    Adjust threshold based on number of stored embeddings:
    - Few embeddings (< 5): Require higher match percentage (35%)
    - Medium embeddings (5-15): Standard threshold (20%)
    - Many embeddings (> 15): Lower threshold acceptable (15%)
    """
    if embedding_count < 5:
        return FEW_EMBEDDINGS_THRESHOLD  # Require higher match for few embeddings
    elif embedding_count < 15:
        return MEDIUM_EMBEDDINGS_THRESHOLD  # Standard threshold
    else:
        return MANY_EMBEDDINGS_THRESHOLD  # Can be more lenient with many embeddings

def get_quality_based_threshold(matches):
    """
    Adjust threshold based on the quality of matching embeddings:
    - High quality matches can use lower threshold
    - Low quality matches require higher threshold
    """
    if not matches:
        return DEFAULT_ADAPTIVE_THRESHOLD  # Default threshold if no matches
        
    # Extract quality scores if available, otherwise use similarity as proxy
    quality_scores = []
    for match in matches:
        if 'quality_score' in match:
            quality_scores.append(match['quality_score'])
        else:
            # Use similarity as a proxy for quality
            quality_scores.append(match['avg_similarity'])
    
    avg_quality = sum(quality_scores) / len(quality_scores)
    
    # Base threshold adjusted by quality (0.5-1.5 range)
    base_threshold = DEFAULT_ADAPTIVE_THRESHOLD
    quality_factor = 1.0 + (0.5 - avg_quality)
    
    return base_threshold * quality_factor

def get_confidence_based_threshold(similarities):
    """
    Adjust threshold based on:
    - Similarity score distribution
    - Gap between best and second-best matches
    - Consistency of matches
    """
    if len(similarities) < 2:
        return DEFAULT_ADAPTIVE_THRESHOLD  # Default threshold if not enough data
    
    # Sort similarities in descending order
    sorted_sims = sorted(similarities, reverse=True)
    best_sim = sorted_sims[0]
    second_best = sorted_sims[1]
    
    # Calculate confidence metrics
    similarity_gap = best_sim - second_best
    similarity_std = np.std(similarities)
    
    # Base threshold
    threshold = 0.20
    
    # Adjust based on separation between best and second-best match
    if similarity_gap > 0.15:  # Clear separation
        threshold *= 0.8  # Lower threshold
    elif similarity_gap < 0.05:  # Close matches
        threshold *= 1.2  # Raise threshold
    
    # Adjust based on consistency
    if similarity_std < 0.1:  # Consistent matches
        threshold *= 0.9
    else:  # Inconsistent matches
        threshold *= 1.1
        
    return threshold

def get_adaptive_threshold(matches, embedding_count, similarities):
    """
    Combine all factors for a comprehensive adaptive threshold
    """
    # Get individual thresholds
    embedding_threshold = get_embedding_based_threshold(embedding_count)
    quality_threshold = get_quality_based_threshold(matches)
    confidence_threshold = get_confidence_based_threshold(similarities)
    
    # Weighted combination
    weights = {
        'embedding': 0.3,
        'quality': 0.4,
        'confidence': 0.3
    }
    
    final_threshold = (
        weights['embedding'] * embedding_threshold +
        weights['quality'] * quality_threshold +
        weights['confidence'] * confidence_threshold
    )
    
    # Ensure threshold stays within reasonable bounds
    return max(0.15, min(0.40, final_threshold)) 