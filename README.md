# Tello DeepSync: Autonomous AI Drone System

Tello DeepSync is a sophisticated autonomous flight system for the DJI Tello drone, powered by YOLOv8 object detection. The system enables the drone to navigate complex environments, recognize directional signs, detect hazards (fire/smoke), and perform maneuvers autonomously.

## 🚀 Key Features

- **Autonomous Navigation**: Intelligent logic for searching, tracking, and executing commands based on visual cues.
- **YOLOv8 Integration**: High-speed object detection for directional arrows, maneuver signs, and fire/smoke detection.
- **Advanced HUD**: A professional fighter-pilot style Heads-Up Display showing:
    - Real-time Telemetry (Altitude, Speed, Temperature)
    - Battery Status & Failsafe Alerts
    - AI Processing FPS & Target Lock Status
    - Live TOF (Time of Flight) Sensor Data
- **Multi-Threaded Architecture**: Separate threads for AI processing, telemetry, and flight logic to ensure smooth performance.
- **Robust Control Logic**: 
    - Automatic centering and approach based on object size and TOF data.
    - Mathematical orientation correction for arrow signs using OpenCV image processing.
- **Safety Systems**: 
    - Critical battery failsafe (automatic landing at <10%).
    - Connection monitoring and recovery.
    - Temperature-based warnings.

## 🛠 Project Structure

- `fly.py`: The core application containing the autonomous logic, AI worker, and HUD system.
- `train.py`: Script to train the YOLOv8 model for fire/smoke or custom object detection.
- `export_onnx.py`: Utility to export trained `.pt` models to ONNX format for optimization.
- `data.yaml`: YOLOv8 dataset configuration file.
- `best.pt` / `fire.pt`: Pre-trained YOLOv8 weights for navigation and hazard detection.

## 📋 Requirements

- Python 3.8+
- OpenCV
- Ultralytics (YOLOv8)
- DJITelloPy
- NumPy

Install dependencies via pip:
```bash
pip install opencv-python ultralytics djitellopy numpy
```

## 🎮 How to Use

1. **Connect**: Power on your DJI Tello and connect your PC to its Wi-Fi network.
2. **Run**: Execute the main script:
   ```bash
   python fly.py
   ```
3. **Controls**:
   - `T`: Takeoff (Automatic search mode begins)
   - `L`: Land
   - `Q`: Quit and Emergency Shutdown
   - `C`: Reconnect to Drone

## 🧠 Autonomous Logic Flow

1. **Searching**: The drone moves forward or rotates to find a recognized target.
2. **Tracking**: Once a target (e.g., "up arrow") is detected, the drone centers it in its field of view.
3. **Approaching**: The drone moves closer to the target until it reaches the "trigger width" or ideal TOF distance.
4. **Execution**: The drone performs the command associated with the target (Rotate, Move Up/Down, Flip, etc.).
5. **Hazard Detection**: If fire or smoke is detected, the drone prioritizes safety, issues a warning, and maintains a safe distance.

## 🛡️ Safety & Optimization

- **Socket Protection**: The system manages socket communication to prevent "WinError 6" and other common Tello SDK lockups.
- **Spam Filtering**: Suppresses redundant logging to maintain console clarity.
- **Morphological Processing**: Uses OpenCV to verify arrow directions, preventing AI misclassification in low-light or skewed angles.

---
*Developed for advanced autonomous drone research and search & rescue simulation.*
