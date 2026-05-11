import os

SIMULASYON_MODU_ZORLA = False 

if SIMULASYON_MODU_ZORLA:
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

SAGA_SOLA_MESAFE      = 70
ILERI_GITME_MESAFE    = 50
GERI_GITME_MESAFE     = 80
HEDEF_IDEAL_MESAFE_CM = 40
YUKARI_GITME_MESAFE   = 80
ASAGI_GITME_MESAFE    = 50
DONUS_ACISI           = 90

AI_GUVEN_ESIGI        = 0.55
FIRE_CONF             = 0.60
SMOKE_CONF            = 0.45
AI_IMG_SIZE           = 640

BATTERY_FAILSAFE      = 15
EKRAN_GENISLIK        = 960
EKRAN_YUKSEKLIK       = 720

ARAMA_HIZI            = 10 if SIMULATION else 20
TARAMA_HIZI           = 25
TARAMA_BEKLEME        = 2.0
TETIKLEME_GENISLIK    = 320
MAX_YAKLASMA_HIZI     = 15
MIN_YAKLASMA_HIZI     = 8

KILITLENME_SURESI     = 0.4
YATAY_HASSASIYET      = 200
DIKEY_HASSASIYET      = 210
MIN_KUTU_GENISLIK     = 0
MAX_KUTU_GENISLIK     = 900

MERKEZ_X = EKRAN_GENISLIK // 2
MERKEZ_Y = EKRAN_YUKSEKLIK // 2
