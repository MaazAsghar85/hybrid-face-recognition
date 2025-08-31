import sqlite3
import pickle
from datetime import datetime
import cv2
import numpy as np
from config import DB_PATH, DEBUG_MODE, MAX_EMBEDDINGS_PER_PERSON

def init_database():
    """Initialize the database and return connection and cursor"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Enable WAL mode for better concurrent access
        cursor.execute("PRAGMA journal_mode=WAL")
        
        # Create faces table (main table for person info)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image BLOB,
            features BLOB,
            timestamp TEXT
        )
        ''')
        
        # Create face_embeddings table (stores averaged embeddings per face)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id INTEGER NOT NULL,
            features BLOB NOT NULL,
            quality_score FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (face_id) REFERENCES faces(id)
        )
        ''')
        
        # Create active_person table (stores currently active person)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS active_person (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES faces(id)
        )
        ''')
        
        # Ensure only one row exists in active_person table
        cursor.execute("SELECT COUNT(*) FROM active_person")
        count = cursor.fetchone()[0]
        if count == 0:
            cursor.execute("INSERT INTO active_person (person_id) VALUES (NULL)")
        elif count > 1:
            cursor.execute("DELETE FROM active_person")
            cursor.execute("INSERT INTO active_person (person_id) VALUES (NULL)")
        
        # Create indices for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_face_id ON face_embeddings(face_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name)')
        
        conn.commit()
        
        # Get face count
        cursor.execute("SELECT COUNT(*) FROM faces")
        face_count = cursor.fetchone()[0]
        if DEBUG_MODE:
            print(f"Database contains {face_count} faces")
        
        return conn, cursor
    except Exception as e:
        print(f"Database error: {e}")
        return None, None

def _prepare_face_data(face_img, features=None):
    """Helper function to prepare face data for database storage"""
    # Convert image to bytes
    _, img_encoded = cv2.imencode('.jpg', face_img)
    img_bytes = img_encoded.tobytes()
    
    # Convert features to bytes if provided
    features_bytes = features.tobytes() if features is not None else None
    
    # Get timestamp
    timestamp = datetime.now().isoformat()
    
    return img_bytes, features_bytes, timestamp

def save_faces_batch(cursor, conn, faces_data):
    """Save multiple faces to the database in a single transaction"""
    try:
        if not faces_data:
            return []
        
        if DEBUG_MODE:
            print(f"Saving {len(faces_data)} faces in batch...")
        
        face_ids = []
        cursor.execute("BEGIN TRANSACTION")
        
        try:
            for face_img, name, features in faces_data:
                img_bytes, features_bytes, timestamp = _prepare_face_data(face_img, features)
                
                # Insert into database
                cursor.execute(
                    "INSERT INTO faces (name, image, features, timestamp) VALUES (?, ?, ?, ?)",
                    (name, img_bytes, features_bytes, timestamp)
                )
                face_ids.append(cursor.lastrowid)
            
            conn.commit()
            
            if DEBUG_MODE:
                print(f"✓ Successfully saved {len(faces_data)} faces")
            return face_ids
            
        except Exception as e:
            conn.rollback()
            raise e
            
    except Exception as e:
        print(f"Error in batch save: {e}")
        return []

def save_face(cursor, conn, face_img, name, features=None, quality_score=None):
    """Save a face to the database with optional features and quality score"""
    try:
        if DEBUG_MODE:
            print(f"Saving face for '{name}'...")
        
        cursor.execute("BEGIN TRANSACTION")
        try:
            # Check if face already exists
            cursor.execute("SELECT id FROM faces WHERE name = ?", (name,))
            result = cursor.fetchone()
            
            if result is None:
                # New face - save image
                img_bytes, _, timestamp = _prepare_face_data(face_img)
                cursor.execute('INSERT INTO faces (name, image, timestamp) VALUES (?, ?, ?)',
                          (name, img_bytes, timestamp))
                face_id = cursor.lastrowid
            else:
                face_id = result[0]
            
            # Handle features and quality score if provided
            if features is not None and quality_score is not None:
                # Check how many embeddings this face already has
                cursor.execute("SELECT COUNT(*) FROM face_embeddings WHERE face_id = ?", (face_id,))
                embedding_count = cursor.fetchone()[0]
                
                if embedding_count < MAX_EMBEDDINGS_PER_PERSON:  # Only store up to N averaged embeddings per face
                    # Save the averaged embedding
                    embedding_bytes = features.tobytes()
                    cursor.execute('''
                    INSERT INTO face_embeddings (face_id, features, quality_score)
                    VALUES (?, ?, ?)
                    ''', (face_id, embedding_bytes, float(quality_score)))
            
            conn.commit()
            
            if DEBUG_MODE:
                print(f"✓ Saved face {face_id} as '{name}'")
            return face_id
            
        except Exception as e:
            conn.rollback()
            raise e
            
    except Exception as e:
        print(f"Error saving face: {e}")
        return None

def clear_database(cursor, conn):
    """Clear all faces from the database"""
    try:
        cursor.execute("BEGIN TRANSACTION")
        try:
            # Clear face_embeddings first (due to foreign key constraint)
            cursor.execute("DELETE FROM face_embeddings")
            # Then clear faces
            cursor.execute("DELETE FROM faces")
            conn.commit()
            print("Database cleared")
            return True
        except Exception as e:
            conn.rollback()
            raise e
    except Exception as e:
        print(f"Error clearing database: {e}")
        return False

def get_face_count(cursor):
    """Get the number of faces in the database"""
    try:
        cursor.execute("SELECT COUNT(*) FROM faces")
        return cursor.fetchone()[0]
    except Exception as e:
        print(f"Error getting face count: {e}")
        return 0

def get_all_faces(cursor):
    """Get all faces from the database"""
    try:
        cursor.execute("SELECT id, name FROM faces")
        return cursor.fetchall()
    except Exception as e:
        print(f"Error getting faces: {e}")
        return []

def load_face_embeddings(cursor):
    """Load all averaged face embeddings from database"""
    try:
        cursor.execute('''
        SELECT f.id, f.name, fe.features, fe.quality_score
        FROM faces f
        JOIN face_embeddings fe ON f.id = fe.face_id
        ORDER BY f.id, fe.quality_score DESC
        ''')
        rows = cursor.fetchall()
        
        face_data = {}
        for face_id, name, feat_blob, quality in rows:
            if feat_blob is not None:
                if face_id not in face_data:
                    face_data[face_id] = {
                        'name': name,
                        'embeddings': [],
                        'qualities': []
                    }
                            # Only store up to N embeddings per face
            if len(face_data[face_id]['embeddings']) < MAX_EMBEDDINGS_PER_PERSON:
                    face_data[face_id]['embeddings'].append(np.frombuffer(feat_blob))
                    face_data[face_id]['qualities'].append(quality)
        
        return face_data
    except Exception as e:
        print(f"Error loading face embeddings: {e}")
        return {}

def update_active_person(cursor, person_id):
    """Update the active person in the database"""
    try:
        # Don't update if person_id is None
        if person_id is None:
            return True
            
        cursor.execute("""
            UPDATE active_person 
            SET person_id = ?, last_seen = CURRENT_TIMESTAMP 
            WHERE id = (SELECT id FROM active_person LIMIT 1)
        """, (person_id,))
        cursor.connection.commit()
        if DEBUG_MODE:
            print(f"Updated active person to ID: {person_id}")
        return True
    except Exception as e:
        print(f"Error updating active person: {e}")
        return False

def get_active_person(cursor):
    """Get the currently active person from the database"""
    try:
        cursor.execute("""
            SELECT f.id, f.name, a.last_seen 
            FROM active_person a 
            LEFT JOIN faces f ON a.person_id = f.id 
            LIMIT 1
        """)
        result = cursor.fetchone()
        if result and result[0] is not None:
            person_id, name, last_seen = result
            return person_id, name, last_seen
        return None, "Unknown", None
    except Exception as e:
        print(f"Error getting active person: {e}")
        return None, "Unknown", None