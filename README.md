# AI Personal Trainer

An intelligent fitness analysis application that uses computer vision (MediaPipe & OpenCV) to track exercise reps, analyze form, and provide real-time feedback.

## Features

- **Real-time Pose Estimation**: Tracks 33 3D body landmarks.
- **Exercise Support**:
  - **Dumbbell Curls**:
    - Independent Left/Right arm tracking.
    - Counts full repetitions (< 80° flexion to > 140° extension).
    - Auto-detects active arm.
  - **Squats**:
    - Depth analysis.
    - Counts repetitions based on knee angles (< 100° down, > 160° up).
- **Repetition Counting**: Robust logic to prevent false positives and support partial reps.
- **Feedback System**:
  - Displays repetition count.
  - Visual overlays for joint angles and skeleton.
  - Directional feedback (e.g., "Up (Right)", "Good Depth!").
- **GPU Acceleration**: Experimental support for hardware acceleration via MediaPipe Tasks API.

## Installation

### Prerequisites
- Python 3.10 or higher.
- [uv](https://github.com/astral-sh/uv) (Recommended package manager) or pip.

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-personal-trainer
   ```

2. **Install dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```
   *Or with standard pip:*
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application using the command line.

### Basic Usage (Webcam)
```bash
uv run python main.py
```
*Defaults to Dumbbell Curl exercise.*

### Select Exercise
Analyze Squats:
```bash
uv run python main.py --exercise squat
```

### Use a Video File
Analyze a pre-recorded video instead of webcam:
```bash
uv run python main.py --video path/to/video.mp4
```

### Enable GPU Acceleration (Experimental)
Use the modern MediaPipe Task API with GPU support:
```bash
uv run python main.py --gpu
```

## Project Structure

- `main.py`: Application entry point. Handles video capture, UI drawing, and loop.
- `pose_estimator.py`:
  - `PoseDetector`: Legacy CPU-based detector wrapper.
  - `GPUPoseDetector`: GPU-accelerated detector using Tasks API.
- `exercises.py`: Contains logic for specific exercises (`DumbbellCurl`, `Squat`).
- `tests/`: Unit tests for exercise logic.

## Technology Stack

- **Python**: Core language.
- **MediaPipe**: For pose detection and landmark extraction.
- **OpenCV**: For image processing and display.
- **NumPy**: For angle calculations.

## Contributing

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License

Distributed under the MIT License. See `LICENSE` for more information.
