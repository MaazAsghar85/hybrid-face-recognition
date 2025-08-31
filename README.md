# Hybrid Face Recognition System

A real-time face recognition system that combines InsightFace (ML-based) with traditional computer vision techniques for robust face detection, tracking, and recognition.

## Features

### Core Functionality
- **Real-time face detection** using InsightFace with CPU-only processing
- **Hybrid feature extraction** combining InsightFace embeddings with traditional CV techniques (histograms, edge detection, LBP patterns)
- **IoU-based face tracking** with identity persistence across frames
- **Two-phase recognition system**: Recognition → Registration for unknown faces
- **Adaptive recognition thresholds** based on embedding count and quality
- **Continuous learning** - improves recognition by adding new embeddings for known faces
- **Quality-based filtering** for optimal face registration and recognition

### Technical Features
- **Single-threaded processing** for real-time performance
- **SQLite database** with WAL mode for persistent face storage
- **Configurable parameters** centralized in `config.py`
- **Comprehensive debugging** with detailed logging and status display
- **Pose and quality assessment** for reliable face registration
- **Active person tracking** with center-based selection

## System Architecture

### Components
1. **Face Detector** (`detector.py`) - InsightFace-based detection and hybrid feature extraction
2. **Face Tracker** (`tracker.py`) - IoU-based tracking and identity management
3. **Database Manager** (`database.py`) - SQLite operations with WAL mode and face storage
4. **Display Manager** (`display.py`) - Real-time visualization and status display
5. **Utility Functions** (`utils.py`) - Quality checks, pose assessment, and adaptive thresholds
6. **Configuration** (`config.py`) - Centralized system parameters

### Recognition Flow
1. **Detection Phase**: InsightFace detects faces in each frame
2. **Tracking Phase**: IoU matching maintains face identity across frames
3. **Quality Assessment**: Face quality and pose are evaluated
4. **Embedding Collection**: 10 frames are collected for analysis
5. **Recognition Phase**: Attempts to match against known faces using cosine similarity
6. **Registration Phase**: If no match found, registers as new person
7. **Continuous Learning**: Adds new embeddings for known faces

## Installation

### Prerequisites
- Python 3.8+
- OpenCV
- InsightFace
- NumPy
- SQLite3

### Setup
```bash
# Install required packages
pip install opencv-python insightface numpy

# Clone the repository
git clone <repository-url>
cd face

# Run the system
python main.py
```

## Configuration

All system parameters are centralized in `config.py`:

### Recognition Parameters
- `RECOGNITION_THRESHOLD = 0.85` - Base similarity threshold
- `HIGH_CONFIDENCE_THRESHOLD = 0.90` - High confidence threshold
- `EVALUATION_FRAMES = 10` - Frames collected for analysis

### Quality Parameters
- `MIN_BRIGHTNESS = 65` - Minimum brightness for face quality
- `MAX_BRIGHTNESS = 165` - Maximum brightness for face quality
- `MIN_BLUR_SCORE = 20` - Minimum blur score for face quality
- `MIN_FACE_DIMENSION = 80` - Minimum face size for registration

### Pose Parameters
- `MAX_EYE_ANGLE = 10` - Maximum eye angle for face tilt (degrees)
- `MIN_EYE_RATIO = 0.25` - Minimum eye distance ratio for frontal face
- `MAX_NOSE_OFFSET = 0.08` - Maximum nose offset for frontal face
- `MIN_VERTICAL_RATIO = 0.8` - Minimum vertical ratio (looking up)
- `MAX_VERTICAL_RATIO = 1.5` - Maximum vertical ratio (looking down)

### Registration Parameters
- `REGISTRATION_SIMILARITY_THRESHOLD = 0.70` - Prevents duplicate registrations
- `REGISTRATION_COOLDOWN = 2.0` - Cooldown between registrations (seconds)
- `MAX_EMBEDDINGS_PER_PERSON = 30` - Maximum embeddings per person

### Feature Extraction Parameters
- `GLOBAL_HIST_BINS = 32` - Number of bins for global histogram
- `REGIONAL_HIST_BINS = 16` - Number of bins for regional histograms
- `EDGE_HIST_BINS = 32` - Number of bins for edge histogram
- `CANNY_LOW_THRESHOLD = 100` - Canny edge detection low threshold
- `CANNY_HIGH_THRESHOLD = 200` - Canny edge detection high threshold
- `LBP_RADIUS = 1` - Local Binary Pattern radius
- `LBP_POINTS = 8` - Local Binary Pattern points
- `FEATURE_CACHE_SIZE = 1000` - Maximum number of cached features

## Usage

### Controls
- **q**: Quit the application
- **r**: Manually register current face
- **c**: Clear the database
- **s**: Show all saved faces

### Operation Modes

#### Recognition Mode
- System attempts to recognize detected faces
- Collects 10 frames for analysis
- Uses adaptive thresholds based on embedding count
- Displays confidence scores and match results

#### Registration Mode
- Activates when no match is found
- Collects 10 high-quality frames
- Performs pose and quality checks
- Prevents duplicate registrations
- Automatically assigns Person_N naming

#### Continuous Learning
- Adds new embeddings for known faces
- Improves recognition accuracy over time
- Maintains embedding quality standards
- Updates active person tracking

## Database Structure

### Tables
- **faces**: Stores face metadata and basic information
- **face_embeddings**: Stores averaged embeddings per face with quality scores
- **active_person**: Tracks currently active person

### Face Storage
- **Embeddings**: Feature vectors combining InsightFace and traditional CV features
- **Quality Scores**: Brightness and blur metrics
- **Metadata**: Name, registration time, face ID

## Performance Features

### Adaptive Thresholds
- **Few embeddings** (< 5): 0.35 threshold
- **Medium embeddings** (5-15): 0.20 threshold  
- **Many embeddings** (> 15): 0.15 threshold

### Quality Filtering
- **Brightness checks**: Ensures proper lighting (65-165 range)
- **Blur detection**: Filters out blurry frames using Laplacian variance
- **Pose assessment**: Validates face orientation using facial landmarks
- **Size validation**: Ensures adequate face size (minimum 80px)

### Tracking Optimization
- **IoU-based matching**: Maintains face identity across frames
- **Frame persistence**: Handles temporary occlusions
- **Center-based selection**: Prioritizes centered faces for active person
- **Automatic cleanup**: Removes stale tracks after 1 second

### Feature Caching
- **Feature cache**: Stores calculated features to avoid recomputation
- **Cache size limit**: Maximum 1000 cached features
- **Automatic cleanup**: Removes oldest entries when limit exceeded

## Debugging and Monitoring

### Debug Mode
Enable `DEBUG_MODE = True` in `config.py` for detailed logging:
- Recognition attempts and results
- Quality check failures
- Registration progress
- Track management events
- Feature calculation details

### Status Display
Real-time information shown on video feed:
- Face bounding boxes with confidence
- Person names and similarity scores
- Quality metrics (brightness, blur)
- System status and frame counts
- Collection progress (X/10 frames)

## File Structure

```
face/
├── main.py              # Main application entry point
├── config.py            # Centralized configuration
├── detector.py          # Face detection and hybrid feature extraction
├── tracker.py           # Face tracking and identity management
├── database.py          # Database operations with WAL mode
├── display.py           # Visualization and status display
├── utils.py             # Utility functions and quality checks
├── face_database/       # SQLite database storage
│   └── faces.sqlite
└── README.md           # This file
```

## Technical Implementation Details

### Feature Extraction
- **InsightFace embeddings**: 512-dimensional ML-based features
- **Traditional features**: 
  - Global histogram (32 bins)
  - Regional histograms (9 regions, 16 bins each)
  - Edge histogram (32 bins) using Canny edge detection
  - Local Binary Patterns (LBP) texture features
- **Hybrid combination**: Concatenated feature vectors
- **Normalization**: L2 normalization for consistent similarity calculation

### Similarity Calculation
- **Cosine similarity**: Primary similarity metric using normalized vectors
- **Adaptive thresholds**: Dynamic based on embedding count
- **Multi-embedding matching**: Compares against all stored embeddings per person
- **Quality weighting**: Considers face quality in matching decisions

### Memory Management
- **Feature caching**: Efficient storage and retrieval with timestamp-based cleanup
- **Track cleanup**: Automatic removal of stale tracks
- **Database optimization**: Indexed queries for fast matching
- **Embedding limits**: Prevents excessive memory usage (30 per person)

### Error Handling
- **Recovery mechanisms**: Automatic reinitialization of face detector
- **Feature calculation recovery**: Cache clearing on errors
- **Database transaction rollback**: Ensures data integrity
- **Graceful degradation**: Continues operation on non-critical errors

## Current Limitations

### Performance
- **CPU-only processing**: No GPU acceleration implemented
- **Single-threaded**: No multi-threading for parallel processing
- **Single camera**: No multi-camera support
- **Basic optimization**: No advanced performance tuning

### Features
- **No age/gender detection**: Basic face recognition only
- **No emotion recognition**: No facial expression analysis
- **No liveness detection**: No anti-spoofing measures
- **No face clustering**: No automatic grouping of similar faces
- **No API interface**: Command-line application only

### Scalability
- **Local database**: No cloud synchronization
- **Single instance**: No distributed processing
- **Limited storage**: No advanced database optimization
- **No batch processing**: Processes one frame at a time

## Recent Improvements

### Version 1.0 Enhancements
- **Fixed registration logic**: Proper phase transitions and embedding collection
- **Separated track/person IDs**: Prevents ID conflicts between tracking and registration
- **Enhanced quality checks**: Comprehensive pose and quality assessment
- **Improved debugging**: Comprehensive logging for troubleshooting
- **Database initialization**: Proper face ID management across sessions
- **Duplicate prevention**: Smart similarity checking during registration

### Bug Fixes
- **Registration completion**: Fixed stuck registration after 10 frames
- **Person ID numbering**: Corrected first person registration as Person_1
- **Quality threshold tuning**: Adjusted pose parameters for better acceptance
- **Logic flow optimization**: Streamlined recognition and registration phases
- **Feature extraction recovery**: Added error handling and cache management

## Troubleshooting

### Common Issues
1. **"Stuck in recognition"**: Check quality thresholds in `config.py`
2. **Poor recognition**: Ensure adequate lighting and face size
3. **Database errors**: Verify SQLite permissions and file integrity
4. **Performance issues**: Adjust `EVALUATION_FRAMES` and quality thresholds

### Performance Tuning
- **Lower `EVALUATION_FRAMES`**: Faster response, less accuracy
- **Adjust quality thresholds**: Balance between acceptance and quality
- **Modify recognition thresholds**: Tune for your use case
- **Enable/disable features**: Customize based on requirements

## Future Enhancements (Planned)

### Performance Optimizations
- **GPU acceleration**: CUDA support for faster processing
- **Model quantization**: Reduced memory footprint
- **Streaming optimization**: Improved real-time performance
- **Database optimization**: Advanced indexing and caching

### Feature Additions
- **Multi-camera support**: Distributed face recognition
- **Age/gender detection**: Additional demographic information
- **Emotion recognition**: Facial expression analysis
- **Access control integration**: Door lock and security systems
- **Cloud synchronization**: Multi-device face database sync

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Support

For support and questions:
- Check the debugging section above
- Review configuration parameters
- Enable debug mode for detailed logging
- Check the troubleshooting section for common issues 