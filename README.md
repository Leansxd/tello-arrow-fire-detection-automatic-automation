<h1 align="center">🛸 Tello DeepSync: Autonomous AI Drone Mission System</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" alt="Python Badge"/>
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge&logo=pytorch" alt="YOLOv8 Badge"/>
  <img src="https://img.shields.io/badge/OpenCV-Computer_Vision-green?style=for-the-badge&logo=opencv" alt="OpenCV Badge"/>
  <img src="https://img.shields.io/badge/DJI_Tello-Robotics-orange?style=for-the-badge&logo=dji" alt="DJI Tello Badge"/>
  <img src="https://img.shields.io/badge/Status-Production_Ready-brightgreen?style=for-the-badge" alt="Status Badge"/>
</p>

<p align="center">
  A premium, professional-grade educational SDK and autonomous flight controller for the <b>DJI Tello / Tello Talent</b> drone.
  <br />
  Leverages dual-model <b>YOLOv8 AI object detection</b> (navigation signs + fire/smoke detection), PID-controlled physical tracking, telemetry logging, and a glassmorphism HUD interface.
</p>

---

## 📂 Project Architecture

The repository workspace has been optimized to keep student work separate from core library code, preventing directory clutter:

```text
tello-arrow-fire-detection/
├── core/                         # Core SDK, libraries, and AI weights
│   ├── best.pt                   # YOLOv8 Navigation sign detection model
│   ├── fire.pt                   # YOLOv8 Fire & Smoke detection model
│   ├── tello_otonom.py           # Core flight controller & PID logic
│   ├── drone_config.py           # System-wide hardware variables
│   └── bridge.py                 # Simulation / communication bridge
├── logs/                         # Automatically created CSV flight data & AVI video captures
└── ogrenci_gorev_1.py            # Student-editable task file (Main entry point)
```

---

## 🚀 Installation & Setup

1. **Clone or Download** the repository to your workspace.
2. **Install system dependencies** by running:
   ```bash
   pip install opencv-python ultralytics djitellopy numpy pyttsx3 comtypes pypiwin32 pywin32
   ```
3. **Turn on your Tello drone**, connect your PC to the drone's Wi-Fi network.
4. **Execute the mission script**:
   ```bash
   python ogrenci_gorev_1.py
   ```

---

## 🎮 Interactive HUD Controls

During flight, an interactive HUD screen is rendered. You can use the following keyboard keys to command the drone manually in case of emergency:

*   **`T`**: Connects and issues a manual **Takeoff** command.
*   **`L`**: Instantly issues a manual **Landing** command (Emergency override).
*   **`Q`**: **Safely terminates** the program, lands the drone, and turns off all LEDs.
*   **`C`**: Triggers custom disconnection.

---

## 🛠️ Decorator-Based Autonomous Mission API

Students configure autonomous flight paths using a clean, Python decorator-based API in `ogrenci_gorev_1.py`. When the drone's front camera detects and centers on a sign, the corresponding decorator function is executed.

<h3 align="center">Sign Mapping Example</h3>

<p align="center">
  Each direction has its own target string mapped to YOLOv8 classes.
</p>

```python
@drone.hedefte("sol")
def move_left(tello):
    print("[TASK] Left sign detected. Moving left.")
    tello.move_left(drone_config.LEFT_RIGHT_DISTANCE)

@drone.hedefte("sag")
def move_right(tello):
    print("[TASK] Right sign detected. Moving right.")
    tello.move_right(drone_config.LEFT_RIGHT_DISTANCE)

@drone.hedefte("yukari")
def move_up(tello):
    print("[TASK] Up sign detected. Climbing up.")
    tello.move_up(drone_config.UPWARD_DISTANCE)

@drone.hedefte("asagi")
def move_down(tello):
    print("[TASK] Down sign detected. Descending.")
    tello.move_down(drone_config.DOWNWARD_DISTANCE)
```

---

## ⚙️ Adjustable Mission Parameters

You can easily adjust the behavior of the drone directly at the top of `ogrenci_gorev_1.py`:

```python
# Flight Distance (cm) and Rotation Angle Settings
drone_config.LEFT_RIGHT_DISTANCE      = 80  # Distance to move left/right (cm)
drone_config.FORWARD_DISTANCE         = 50  # Distance to move forward (cm)
drone_config.BACKWARD_DISTANCE        = 80  # Distance to move backward (cm)
drone_config.TARGET_IDEAL_DISTANCE_CM = 40  # Ideal distance to stop before the sign (cm)
drone_config.UPWARD_DISTANCE          = 80  # Distance to climb vertically (cm)
drone_config.DOWNWARD_DISTANCE        = 50  # Distance to descend vertically (cm)
drone_config.ROTATION_ANGLE           = 90  # Degrees to turn on rotation signs

# Drone Speed Settings
drone_config.SEARCH_SPEED             = 20  # Forward speed during search mode (cm/s)
drone_config.ALIGNMENT_SPEED_LIMIT    = 20  # Max adjustment speed limit during PID centering (cm/s)

# Centering Precision & Locking Settings
drone_config.HORIZONTAL_SENSITIVITY   = 150 # Horizontal target centering tolerance (pixels)
drone_config.VERTICAL_SENSITIVITY     = 150 # Vertical target centering tolerance (pixels)
drone_config.LOCK_DURATION            = 0.5 # Wait duration at center before executing the task (seconds)
```

---

## 📈 System Features

*   👤 **Face Tracking (`face`)**: Real-time human face tracking and locking.
*   🖐️ **Hand Gestures (`fist`, `open_hand`)**: Gesture-based remote flight controls.
*   🔥 **Fire & Smoke Detection**: Runs a secondary YOLOv8 model in parallel, rendering warnings on screen and providing voice notifications if smoke/fire is spotted.
*   📦 **Telemetry Black-Box**: Records second-by-second telemetry data to `logs/flight_*.csv` and captures flight video feed to `logs/video_*.avi`.
*   🛡️ **Smart Failsafe**: Auto-lands the drone immediately if battery falls below `15%`.
