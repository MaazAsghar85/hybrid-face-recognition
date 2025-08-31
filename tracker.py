import time
import numpy as np
from utils import calculate_iou
from config import (
    TRACKING_THRESHOLD,
    EVALUATION_FRAMES,
    DEBUG_MODE,
    DEFAULT_DISPLAY_CONFIDENCE
)
import logging
from database import update_active_person

logger = logging.getLogger(__name__)

class FaceTracker:
    def __init__(self, db_cursor=None):
        self.face_tracking = {}  # For tracking faces between frames
        self.last_face_id = 0  # For person IDs (Person_1, Person_2, etc.)
        self.last_track_id = 0  # For track IDs (track_1, track_2, etc.)
        self.last_registration_time = 0
        self.db_cursor = db_cursor
        self.EMBEDDING_COLLECTION_FRAMES = EVALUATION_FRAMES  # Number of frames to collect embeddings

    def init_track_info(self, track_id):
        """Initialize a new track with all necessary fields"""
        if track_id not in self.face_tracking:
            self.face_tracking[track_id] = {
                'bbox': None,
                'frames_seen': 1,
                'last_seen': 0,
                'track_start_time': 0,
                'face_img': None,
                'name': "Unknown",
                'face_id': None,  # Initialize face_id
                'similarity': 0.0,
                'matched': False,
                'similarity_history': [],
                'match_history': [],
                'evaluated': False,
                'can_register': False,
                'embeddings': [],
                'embedding_qualities': [],
                'recognition_embeddings': [],  # For recognition phase
                'registration_embeddings': [],  # For registration phase
                'phase': 'recognition',  # Current phase: 'recognition' or 'registration'
                'post_registration_embeddings': 0,  # Counter for embeddings after registration
                'needs_more_embeddings': False  # Flag to indicate if we need more embeddings
            }
        return self.face_tracking[track_id]

    def update_tracks(self, face_images, current_time):
        """Update face tracking information"""
        matched_tracks = set()
        unmatched_detections = list(range(len(face_images)))
        
        # First pass: Try to match each track with its closest detection
        for track_id, track_info in self.face_tracking.items():
            best_detection_idx = -1
            best_iou = TRACKING_THRESHOLD
            
            # Find best matching detection for this track
            for idx in unmatched_detections:
                face_img, bbox, insight_face = face_images[idx]
                iou = calculate_iou(bbox, track_info['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_detection_idx = idx
            
            # If we found a match, update the track
            if best_detection_idx >= 0:
                face_img, bbox, insight_face = face_images[best_detection_idx]
                track_info['bbox'] = bbox
                track_info['frames_seen'] += 1
                track_info['last_seen'] = current_time
                track_info['face_img'] = face_img
                matched_tracks.add(track_id)
                unmatched_detections.remove(best_detection_idx)
                logger.debug(f"Updated track {track_id} with detection {best_detection_idx}")
        
        # Second pass: Create new tracks for unmatched detections
        for idx in unmatched_detections:
            face_img, bbox, insight_face = face_images[idx]
            self.last_track_id += 1
            track_id = f"track_{self.last_track_id}"
            
            self.face_tracking[track_id] = {
                'bbox': bbox,
                'frames_seen': 1,
                'last_seen': current_time,
                'track_start_time': current_time,
                'face_img': face_img,
                'name': "Unknown",
                'face_id': None,  # Initialize face_id
                'similarity': 0.0,
                'matched': False,
                'similarity_history': [],
                'match_history': [],
                'evaluated': False,
                'can_register': False,
                'embeddings': [],
                'embedding_qualities': [],
                'recognition_embeddings': [],
                'registration_embeddings': [],
                'phase': 'recognition'
            }
            matched_tracks.add(track_id)
            logger.debug(f"Created new track {track_id} for unmatched detection {idx}")
        
        # Remove old tracks
        self._remove_old_tracks(matched_tracks, current_time)
        
        # Log tracking status
        logger.debug(f"Active tracks: {len(self.face_tracking)}, Matched this frame: {len(matched_tracks)}")
        return matched_tracks

    def _remove_old_tracks(self, matched_tracks, current_time):
        """Remove tracks that haven't been seen recently"""
        tracks_to_remove = []
        for track_id, track_info in self.face_tracking.items():
            if track_id not in matched_tracks:
                # If track wasn't matched and hasn't been seen for a while, remove it
                if current_time - track_info['last_seen'] > 1.0:  # 1 second threshold
                    tracks_to_remove.append(track_id)
                    # Don't clear active person when face leaves frame
                    if DEBUG_MODE and track_info.get('face_id') is not None:
                        print(f"Person left frame (was ID: {track_info['face_id']})")
        
        # Remove old tracks
        for track_id in tracks_to_remove:
            if DEBUG_MODE:
                track_info = self.face_tracking[track_id]
                track_age = current_time - track_info.get('track_start_time', current_time)
                logger.debug(f"Removing track {track_id} (age: {track_age:.1f}s, frames: {track_info['frames_seen']})")
            del self.face_tracking[track_id]

    def update_track_identity(self, track_id, match_result, similarity):
        """Update track identity based on recognition results"""
        track_info = self.face_tracking[track_id]
        
        # Store results in history
        track_info['similarity_history'].append(similarity)
        track_info['match_history'].append(match_result)
        
        # Keep only the last EVALUATION_FRAMES results
        if len(track_info['similarity_history']) > EVALUATION_FRAMES:
            track_info['similarity_history'] = track_info['similarity_history'][-EVALUATION_FRAMES:]
            track_info['match_history'] = track_info['match_history'][-EVALUATION_FRAMES:]
        
        # If we get a very high similarity match, immediately mark as matched
        HIGH_SIMILARITY_THRESHOLD = 0.99
        if similarity > HIGH_SIMILARITY_THRESHOLD and match_result is not None:
            face_id, name = match_result
            track_info['name'] = name
            track_info['face_id'] = face_id
            track_info['similarity'] = similarity
            track_info['matched'] = True
            track_info['evaluated'] = True
            track_info['can_register'] = False
            
            # Update active person in database for high confidence matches
            if self.db_cursor and face_id is not None:
                if DEBUG_MODE:
                    print(f"High confidence match, updating active person to ID: {face_id}")
                update_active_person(self.db_cursor, face_id)
            
            if DEBUG_MODE:
                print(f"Immediate match due to high similarity: {name} ({similarity:.4f})")
            return
        
        # Make identity decision after collecting enough frames
        if len(track_info['similarity_history']) >= EVALUATION_FRAMES and not track_info['evaluated']:
            self._evaluate_track_identity(track_id)

    def _evaluate_track_identity(self, track_id):
        """Evaluate track identity based on collected history"""
        track_info = self.face_tracking[track_id]
        
        # Count how many frames resulted in the same match
        match_counts = {}
        max_similarity = 0.0  # Track highest similarity
        high_similarity_matches = []  # Track all high similarity matches
        consistent_matches = []  # Track all matches above recognition threshold
        
        HIGH_SIMILARITY_THRESHOLD = 0.99
        RECOGNITION_THRESHOLD = 0.85
        
        for i, m in enumerate(track_info['match_history']):
            similarity = track_info['similarity_history'][i]
            # Update max similarity
            max_similarity = max(max_similarity, similarity)
            
            if m is not None:
                face_id, name = m
                # Track high similarity matches
                if similarity > HIGH_SIMILARITY_THRESHOLD:
                    high_similarity_matches.append((name, similarity))
                # Track all matches above recognition threshold
                if similarity > RECOGNITION_THRESHOLD:
                    consistent_matches.append((name, similarity))
                # Count matches for normal evaluation
                if name in match_counts:
                    match_counts[name] += 1
                else:
                    match_counts[name] = 1
        
        # Calculate how many frames had ANY match
        frames_with_match = sum(1 for m in track_info['match_history'] if m is not None)
        
        if DEBUG_MODE:
            print("\nIdentity Evaluation Results:")
            print(f"  Frames with any match: {frames_with_match}/{EVALUATION_FRAMES}")
            if match_counts:
                print(f"  Match counts: {match_counts}")
            print(f"  Max similarity: {max_similarity:.4f}")
            if high_similarity_matches:
                print(f"  High similarity matches: {high_similarity_matches}")
            if consistent_matches:
                print(f"  Consistent matches above threshold: {consistent_matches}")
        
        # If ANY frame had high similarity, prevent registration and match to that person
        if high_similarity_matches:
            # Use the most frequent high similarity match
            high_sim_names = [name for name, _ in high_similarity_matches]
            most_common_high_sim = max(set(high_sim_names), key=high_sim_names.count)
            track_info['name'] = most_common_high_sim
            track_info['similarity'] = max(sim for name, sim in high_similarity_matches if name == most_common_high_sim)
            track_info['matched'] = True
            track_info['evaluated'] = True
            track_info['can_register'] = False
            if DEBUG_MODE:
                print(f"  Decision: Matched to {most_common_high_sim} due to high similarity frames")
            return
        
        # If ANY frame consistently matched above recognition threshold, prevent registration
        if consistent_matches:
            # Use the most frequent match above threshold
            consistent_names = [name for name, _ in consistent_matches]
            most_common_consistent = max(set(consistent_names), key=consistent_names.count)
            track_info['name'] = most_common_consistent
            track_info['similarity'] = max(sim for name, sim in consistent_matches if name == most_common_consistent)
            track_info['matched'] = True
            track_info['evaluated'] = True
            track_info['can_register'] = False
            if DEBUG_MODE:
                print(f"  Decision: Matched to {most_common_consistent} due to consistent matches above threshold")
            return
        
        # Find the most frequent match (mode) for normal evaluation
        most_common_match = None
        max_count = 0
        for name, count in match_counts.items():
            if count > max_count:
                max_count = count
                most_common_match = name
        
        if most_common_match is not None and max_count >= (EVALUATION_FRAMES // 2):
            # Consistent match found
            self._set_consistent_match(track_info, most_common_match)
        else:
            # Handle inconsistent or no matches
            self._handle_inconsistent_match(track_info, frames_with_match)

    def _set_consistent_match(self, track_info, most_common_match):
        """Set track info for consistently matched face"""
        face_id, name = most_common_match
        track_info['name'] = name
        track_info['face_id'] = face_id
        # Calculate average similarity for the most common match
        matched_similarities = [s for i, s in enumerate(track_info['similarity_history']) 
                             if track_info['match_history'][i] is not None and 
                                track_info['match_history'][i][1] == name]
        avg_similarity = sum(matched_similarities) / len(matched_similarities) if matched_similarities else 0
        track_info['similarity'] = avg_similarity
        track_info['matched'] = True
        track_info['evaluated'] = True
        
        # Update active person in database
        if self.db_cursor and face_id is not None:
            update_active_person(self.db_cursor, face_id)
        
        if DEBUG_MODE:
            print(f"  Decision: {name} (mode with {len(matched_similarities)}/{EVALUATION_FRAMES} frames)")

    def _handle_inconsistent_match(self, track_info, frames_with_match):
        """Handle cases where there isn't a consistent match"""
        if frames_with_match == 0:
            # No matches at all - mark as unknown
            track_info['name'] = "Unknown"
            avg_similarity = sum(track_info['similarity_history']) / len(track_info['similarity_history']) if track_info['similarity_history'] else DEFAULT_DISPLAY_CONFIDENCE
            track_info['similarity'] = max(avg_similarity, DEFAULT_DISPLAY_CONFIDENCE)
            track_info['matched'] = False
            track_info['evaluated'] = True
            track_info['can_register'] = True
            
            if DEBUG_MODE:
                print(f"  Decision: Unknown (no matches in any frame) - can register as new")
        else:
            # Some matches but not consistent - mark as ambiguous
            track_info['name'] = "Ambiguous"
            max_similarity = max(track_info['similarity_history']) if track_info['similarity_history'] else DEFAULT_DISPLAY_CONFIDENCE
            track_info['similarity'] = max(max_similarity, DEFAULT_DISPLAY_CONFIDENCE)
            track_info['matched'] = False
            track_info['evaluated'] = True
            track_info['can_register'] = False
            
            if DEBUG_MODE:
                print(f"  Decision: Ambiguous ({frames_with_match}/{EVALUATION_FRAMES} frames had matches) - cannot register")

    def add_embedding(self, track_id, features, quality_score):
        """Add embedding to collection based on current phase"""
        if track_id not in self.face_tracking:
            return
            
        track_info = self.face_tracking[track_id]
        
        # Initialize collections if they don't exist
        if 'recognition_embeddings' not in track_info:
            track_info['recognition_embeddings'] = []
        if 'registration_embeddings' not in track_info:
            track_info['registration_embeddings'] = []
            
        # Normalize features before storing
        features = features / np.linalg.norm(features)
        
        if track_info['phase'] == 'recognition':
            # Collecting embeddings for recognition
            if len(track_info['recognition_embeddings']) < self.EMBEDDING_COLLECTION_FRAMES:
                track_info['recognition_embeddings'].append(features)
                if len(track_info['recognition_embeddings']) >= self.EMBEDDING_COLLECTION_FRAMES:
                    track_info['evaluated'] = True
                    
        elif track_info['phase'] == 'registration' and track_info.get('can_register', False):
            # Collecting embeddings for registration
            if len(track_info['registration_embeddings']) < self.EMBEDDING_COLLECTION_FRAMES:
                track_info['registration_embeddings'].append(features)
                if DEBUG_MODE:
                    print(f"Added registration embedding: {len(track_info['registration_embeddings'])}/{self.EMBEDDING_COLLECTION_FRAMES}")
                
        # Store the embedding for the current active collection
        track_info['embeddings'] = (track_info['registration_embeddings'] 
                                  if track_info['phase'] == 'registration' 
                                  else track_info['recognition_embeddings'])
        
        # Store quality score
        if 'embedding_qualities' not in track_info:
            track_info['embedding_qualities'] = []
        track_info['embedding_qualities'].append(quality_score)

    def has_enough_embeddings(self, track_id):
        """Check if we have enough embeddings for the current phase"""
        track_info = self.face_tracking.get(track_id)
        if not track_info:
            return False
            
        if track_info['phase'] == 'recognition':
            count = len(track_info.get('recognition_embeddings', []))
            if DEBUG_MODE:
                print(f"Recognition embeddings: {count}/{self.EMBEDDING_COLLECTION_FRAMES}")
            return count >= self.EMBEDDING_COLLECTION_FRAMES
        else:
            count = len(track_info.get('registration_embeddings', []))
            if DEBUG_MODE:
                print(f"Registration embeddings: {count}/{self.EMBEDDING_COLLECTION_FRAMES}")
            return count >= self.EMBEDDING_COLLECTION_FRAMES

    def start_registration_phase(self, track_id):
        """Switch to registration phase for a track"""
        if track_id in self.face_tracking:
            track_info = self.face_tracking[track_id]
            track_info['phase'] = 'registration'
            track_info['registration_embeddings'] = []  # Clear any old registration embeddings
            track_info['evaluated'] = False  # Reset evaluation for new phase

    def get_average_embedding(self, track_id):
        """Get average embedding for current phase"""
        track_info = self.face_tracking[track_id]
        embeddings = (track_info['registration_embeddings'] 
                     if track_info['phase'] == 'registration' 
                     else track_info['recognition_embeddings'])
        
        if not embeddings:
            return None
            
        # Convert list of embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Calculate simple average without weights
        avg_embedding = np.mean(embeddings_array, axis=0)
        
        # Normalize the averaged embedding
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        return avg_embedding

    def start_post_registration_collection(self, track_id):
        """Start collecting additional embeddings after registration"""
        if track_id in self.face_tracking:
            track_info = self.face_tracking[track_id]
            track_info['post_registration_embeddings'] = 0
            track_info['needs_more_embeddings'] = True
            track_info['recognition_embeddings'] = []  # Clear existing embeddings
            track_info['embedding_qualities'] = []
            if DEBUG_MODE:
                print(f"Starting post-registration embedding collection for {track_info['name']}")

    def needs_more_embeddings(self, track_id):
        """Check if we need to collect more embeddings for this track"""
        if track_id in self.face_tracking:
            track_info = self.face_tracking[track_id]
            return track_info.get('needs_more_embeddings', False)
        return False

    def increment_post_registration_embeddings(self, track_id):
        """Increment the counter for post-registration embeddings"""
        if track_id in self.face_tracking:
            track_info = self.face_tracking[track_id]
            track_info['post_registration_embeddings'] += 1
            if track_info['post_registration_embeddings'] >= 5:
                track_info['needs_more_embeddings'] = False
            if DEBUG_MODE:
                print(f"Post-registration embeddings for {track_info['name']}: {track_info['post_registration_embeddings']}/5")
            return track_info['post_registration_embeddings']
        return 0 