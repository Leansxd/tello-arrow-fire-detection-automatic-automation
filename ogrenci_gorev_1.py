from core import drone_config
from core.tello_otonom import OtonomSistem

# =====================================================================
# --- 8x8 LED MATRIX (SCREEN) LIBRARY AND PRESETS ---
# =====================================================================
# Color Codes (64-pixel solid colors)
COLOR_OFF    = "0" * 64
COLOR_RED    = "r" * 64
COLOR_GREEN  = "g" * 64
COLOR_BLUE   = "b" * 64
COLOR_PURPLE = "p" * 64

# Pre-defined Patterns / Emojis (8x8 Pixel Matrix)
EMOJI_SMILE    = "00rrrr000r0000r0r0r00r0rr000000rr0r00r0rr00rr00r0r0000r000rrrr00"
EMOJI_HEART    = "000000000rr00rr0rrrrrrrrrrrrrrrr0rrrrrr000rrrr00000rr00000000000"
EMOJI_SAD      = "00rrrr000r0000r0r0r00r0rr000000rr00rr00rr0r00r0r0r0000r000rrrr00"
EMOJI_ARROW_UP = "000rr00000rrrr000rrrrrr0rr0rr0rr000rr000000rr000000rr000000rr000"

EMOJI = "000rr00000r00r0000r00r0000r00r000r0000r00r0000r000r00r00000rr000"

# =====================================================================
# --- ADJUSTABLE CONFIGURATIONS (YOU CAN EDIT THESE) ---
# =====================================================================
# Flight Distance (cm) and Rotation Angle Settings
drone_config.LEFT_RIGHT_DISTANCE      = 180
drone_config.FORWARD_DISTANCE    = 50
drone_config.BACKWARD_DISTANCE     = 80
drone_config.TARGET_IDEAL_DISTANCE_CM = 40
drone_config.UPWARD_DISTANCE   = 50
drone_config.DOWNWARD_DISTANCE    = 60
drone_config.ROTATION_ANGLE           = 90

# Drone Speed Settings (Search speed & Centering/PID speed limit)
drone_config.SEARCH_SPEED            = 30
drone_config.ALIGNMENT_SPEED_LIMIT   = 30

# Centering Precision & Locking Settings (Lower tolerance = centers more precisely before executing action)
drone_config.HORIZONTAL_SENSITIVITY      = 110  # Horizontal centering tolerance (pixels)
drone_config.VERTICAL_SENSITIVITY      = 110  # Vertical centering tolerance (pixels)
drone_config.LOCK_DURATION     = 0.8  # Seconds to wait/stay aligned before executing action

# =====================================================================

drone = OtonomSistem()
tello = drone.tello

# Scroll the text "EMRULLAH ARSLANTAS" on the screen (unconditional/no function)
# tello.send_control_command("ext mled s l r 3.0 EMRULLAH ARSLANTAS {EMOJI_SMILE} AMTAL")

tello.send_control_command("ext mled s l r 3.0 TOBIAS")

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

@drone.hedefte("ileri")
def move_forward(tello):
    print("[TASK] Forward sign detected. Moving forward.")
    tello.move_forward(drone_config.FORWARD_DISTANCE)

@drone.hedefte("geri")
def move_back(tello):
    print("[TASK] Back sign detected. Moving back.")
    tello.move_back(drone_config.BACKWARD_DISTANCE)

@drone.hedefte("soladon")
def turn_left(tello):
    print("[TASK] Turn left arrow detected. Rotating counter-clockwise.")
    tello.rotate_counter_clockwise(drone_config.ROTATION_ANGLE)

@drone.hedefte("sagadon")
def turn_right(tello):
    print("[TASK] Turn right arrow detected. Rotating clockwise.")
    tello.rotate_clockwise(drone_config.ROTATION_ANGLE)

@drone.hedefte("don180")
def turn_180(tello):
    print("[TASK] Turn 180 sign detected. Rotating 180 degrees.")
    tello.rotate_clockwise(180)

@drone.hedefte("takla")
def do_flip(tello):
    print("[TASK] Flip sign detected! Starting the show.")
    tello.flip_back()

@drone.hedefte("fire")
def fire_protocol(tello):
    print("[ALARM] Fire detected! Retracting to a safe distance.")
    tello.move_up(30)
    tello.move_back(100)

@drone.hedefte("smoke")
def smoke_protocol(tello):
    print("[ALARM] Smoke detected! Inspecting the area.")
    tello.move_up(40)

@drone.hedefte("parkurson")
def land_drone(tello):
    print("[TASK] Course completed. Landing. Congratulations!")
    tello.land()

if __name__ == "__main__":
    drone.baslat()
