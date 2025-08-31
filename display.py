import cv2
from utils import calculate_brightness
from config import EVALUATION_FRAMES, MAX_EMBEDDINGS_PER_PERSON

def draw_face_box(frame, bbox, name, similarity, brightness, insight_face, track_info):
    """Draw face detection box with name and info"""
    x1, y1, x2, y2 = bbox
    
    # Determine color and display text based on match status and phase
    if track_info and track_info.get('matched', False):
        # Already matched - show green box with person name
        color = (0, 255, 0)  # Green
        show_confidence = True
    elif track_info and track_info.get('phase') == 'recognition' and not track_info.get('matched', False):
        # In recognition phase
        color = (255, 0, 0)  # Blue
        name = "Recognizing"
        show_confidence = False
    elif track_info and track_info.get('phase') == 'registration' and not track_info.get('matched', False):
        # In registration phase
        color = (0, 0, 255)  # Red
        name = "Registering"
        show_confidence = False
    else:
        # Default case
        color = (0, 0, 255)  # Red
        show_confidence = False
    
    # Draw face box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw name and confidence
    text_y = y1 - 35 if y1 > 50 else y2 + 35
    
    # Format name text
    if show_confidence and similarity > 0:
        name_text = f"{name} ({similarity:.2f})"
    else:
        name_text = name
        
    cv2.putText(frame, name_text,
                (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)
    
    # Draw brightness value
    brightness_text = f"Brightness: {brightness:.1f}"
    cv2.putText(frame, brightness_text,
                (x1, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)
    
    # Draw landmarks if available
    if insight_face and insight_face.kps is not None:
        for point in insight_face.kps:
            x, y = point.astype(int)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    # Draw collection progress if in collection phase and not matched
    if track_info and not track_info.get('matched', False):
        phase = track_info.get('phase', '')
        embeddings = track_info.get(f'{phase}_embeddings', [])
        if embeddings:
            progress = len(embeddings)
            total = EVALUATION_FRAMES  # Total frames needed for current phase
            progress_text = f"{progress}/{total}"
            cv2.putText(frame, progress_text,
                       (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, color, 2)

def draw_status(display, face_tracking, last_face_id, fps):
    """Draw status information on the display"""
    status_y = display.shape[0] - 10
    
    # Show track count
    cv2.putText(display, f"Tracking {len(face_tracking)} faces", 
               (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, (255, 255, 0), 2)
    
    # Show database info
    db_status = f"Database: {last_face_id} faces"
    cv2.putText(display, db_status, 
               (10, status_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, (255, 255, 255), 2)
    
    # Show FPS
    cv2.putText(display, f"FPS: {fps:.1f}", 
               (display.shape[1] - 120, status_y), cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, (255, 255, 255), 2) 