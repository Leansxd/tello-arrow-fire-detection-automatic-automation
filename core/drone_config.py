import os

FORCE_SIMULATION_MODE = False 
LANG = 'TR' # 'TR', 'EN', 'DE'

if FORCE_SIMULATION_MODE:
    try:
        from bridge import Tello
        SIMULATION = True
    except ImportError:
        from djitellopy import Tello
        SIMULATION = False
else:
    try:
        from djitellopy import Tello
        SIMULATION = False
    except ImportError:
        from bridge import Tello
        SIMULATION = True

LEFT_RIGHT_DISTANCE      = 70
FORWARD_DISTANCE    = 50
BACKWARD_DISTANCE     = 80
TARGET_IDEAL_DISTANCE_CM = 40
UPWARD_DISTANCE   = 80
DOWNWARD_DISTANCE    = 50
ROTATION_ANGLE           = 90

AI_CONF_THRESHOLD        = 0.55
FIRE_CONF             = 0.60
SMOKE_CONF            = 0.45
AI_IMG_SIZE           = 640

BATTERY_FAILSAFE      = 15
PROXIMITY_THRESHOLD_CM = 50
SCREEN_WIDTH        = 960
SCREEN_HEIGHT       = 720

SEARCH_SPEED            = 10 if SIMULATION else 20
ALIGNMENT_SPEED_LIMIT   = 12
SCAN_SPEED           = 25
SCAN_WAIT        = 2.0
TRIGGER_WIDTH    = 320
MAX_APPROACH_SPEED     = 15
MIN_APPROACH_SPEED     = 8

LOCK_DURATION     = 0.5
HORIZONTAL_SENSITIVITY      = 150
VERTICAL_SENSITIVITY      = 150
MIN_BOX_WIDTH     = 0
MAX_BOX_WIDTH     = 900

CENTER_X = SCREEN_WIDTH // 2
CENTER_Y = SCREEN_HEIGHT // 2

# --- Çalışma Modu Konfigürasyonu ---
# Ön mesafe sensörüne göre yakınken ve uzakken gösterilecek desenler/renkler
# Bunları ogrenci_gorev_1.py dosyasında ezerek özelleştirebilirsiniz.
MLED_NEAR_MODE = "r" * 64
MLED_FAR_MODE  = "g" * 64

# Sadece batarya yüzdesini gösterme modu (True ise batarya gösterilir, False ise mesafe sensörü)
MLED_BATTERY_ONLY = True


