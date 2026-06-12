import cv2
import time
import math
import threading
import numpy as np
import os
import sys
import logging
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import drone_config
from drone_config import Tello, SIMULATION
from ultralytics import YOLO

try:
    import pyttsx3
    HAS_TTS = True
except Exception:
    HAS_TTS = False

logging.getLogger('djitellopy').setLevel(logging.ERROR)
DroneConfig = drone_config

FONT = {
    'A': [0x18, 0x24, 0x42, 0x7E, 0x42, 0x42, 0x42, 0x00],
    'B': [0x7C, 0x42, 0x42, 0x7C, 0x42, 0x42, 0x7C, 0x00],
    'C': [0x3C, 0x42, 0x40, 0x40, 0x40, 0x42, 0x3C, 0x00],
    'D': [0x78, 0x44, 0x42, 0x42, 0x42, 0x44, 0x78, 0x00],
    'E': [0x7E, 0x40, 0x40, 0x78, 0x40, 0x40, 0x7E, 0x00],
    'F': [0x7E, 0x40, 0x40, 0x78, 0x40, 0x40, 0x40, 0x00],
    'G': [0x3C, 0x42, 0x40, 0x4E, 0x42, 0x42, 0x3C, 0x00],
    'H': [0x42, 0x42, 0x42, 0x7E, 0x42, 0x42, 0x42, 0x00],
    'I': [0x3E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x3E, 0x00],
    'J': [0x1F, 0x04, 0x04, 0x04, 0x04, 0x24, 0x18, 0x00],
    'K': [0x42, 0x44, 0x48, 0x70, 0x48, 0x44, 0x42, 0x00],
    'L': [0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x7E, 0x00],
    'M': [0x42, 0x66, 0x5A, 0x42, 0x42, 0x42, 0x42, 0x00],
    'N': [0x42, 0x62, 0x52, 0x4A, 0x46, 0x42, 0x42, 0x00],
    'O': [0x3C, 0x42, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00],
    'P': [0x7C, 0x42, 0x42, 0x7C, 0x40, 0x40, 0x40, 0x00],
    'Q': [0x3C, 0x42, 0x42, 0x42, 0x4A, 0x44, 0x3A, 0x00],
    'R': [0x7C, 0x42, 0x42, 0x7C, 0x48, 0x44, 0x42, 0x00],
    'S': [0x3C, 0x40, 0x40, 0x3C, 0x02, 0x02, 0x3C, 0x00],
    'T': [0x7E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x00],
    'U': [0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00],
    'V': [0x42, 0x42, 0x42, 0x42, 0x24, 0x24, 0x18, 0x00],
    'W': [0x42, 0x42, 0x42, 0x42, 0x5A, 0x66, 0x42, 0x00],
    'X': [0x42, 0x24, 0x18, 0x18, 0x24, 0x42, 0x42, 0x00],
    'Y': [0x42, 0x42, 0x24, 0x18, 0x08, 0x08, 0x08, 0x00],
    'Z': [0x7E, 0x02, 0x04, 0x08, 0x16, 0x32, 0x7E, 0x00],
    '0': [0x3C, 0x42, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00],
    '1': [0x08, 0x18, 0x28, 0x08, 0x08, 0x08, 0x3E, 0x00],
    '2': [0x3C, 0x42, 0x02, 0x3C, 0x40, 0x40, 0x7E, 0x00],
    '3': [0x3C, 0x02, 0x02, 0x3C, 0x02, 0x02, 0x3C, 0x00],
    '4': [0x08, 0x18, 0x28, 0x48, 0x7E, 0x08, 0x08, 0x00],
    '5': [0x7E, 0x40, 0x40, 0x7C, 0x02, 0x02, 0x7C, 0x00],
    '6': [0x3C, 0x40, 0x40, 0x7C, 0x42, 0x42, 0x3C, 0x00],
    '7': [0x7E, 0x02, 0x04, 0x08, 0x10, 0x10, 0x10, 0x00],
    '8': [0x3C, 0x42, 0x42, 0x3C, 0x42, 0x42, 0x3C, 0x00],
    '9': [0x3C, 0x42, 0x42, 0x3E, 0x02, 0x02, 0x3C, 0x00],
    '♥': [0x00, 0x66, 0xFF, 0xFF, 0x7E, 0x3C, 0x18, 0x00],
    '☺': [0x3C, 0x42, 0xA5, 0x81, 0xA5, 0x99, 0x42, 0x3C],
    '☹': [0x3C, 0x42, 0xA5, 0x81, 0x99, 0xA5, 0x42, 0x3C],
    '↑': [0x18, 0x3C, 0x7E, 0xDB, 0x18, 0x18, 0x18, 0x18],
    ' ': [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
}

class AIWorker(threading.Thread):
    def __init__(self, main_path="best.pt", fire_path="fire.pt"):
        super().__init__()
        self.main_path = main_path
        self.fire_path = fire_path
        self.daemon = True
        self.running = True
        self.is_loaded = False
        self.gesture_objs = []
        self.frame = None
        self.result = None
        self.fire_objs = []
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.fps = 0

    def set_frame(self, frame):
        if not self.is_loaded: return
        with self.lock:
            self.frame = frame.copy()
            self.new_frame_event.set()

    def get_results(self):
        with self.lock:
            return self.result, self.fire_objs, self.fps, self.is_loaded

    def run(self):
        print(f"[AI] ANA MODEL YUKLENIYOR: {self.main_path}")
        try:
            self.model = YOLO(self.main_path, task='detect')
            print("[AI] ANA MODEL OK. YANGIN MODELI YUKLENIYOR...")
            self.fire_model = None
            if os.path.exists(self.fire_path):
                self.fire_model = YOLO(self.fire_path, task='detect')
                print("[AI] YANGIN MODELI OK.")
            self.is_loaded = True
            print("[AI] SİSTEM HAZIR. Tüm modeller basariyla yuklendi.")
        except Exception as e:
            print(f"[AI] Yukleme hatasi: {e}")

        last_time = time.time()
        frame_count = 0
        while self.running:
            self.new_frame_event.wait(timeout=0.1)
            if not self.new_frame_event.is_set(): continue
            with self.lock:
                img = self.frame
                self.new_frame_event.clear()
            if img is None: continue
            img = cv2.resize(img, (960, 720))
            res = self.model.predict(img, verbose=False, conf=DroneConfig.AI_CONF_THRESHOLD, imgsz=DroneConfig.AI_IMG_SIZE)
            fires = None
            if self.fire_model and frame_count % 5 == 0:
                fires = []
                f_res = self.fire_model.predict(img, verbose=False, conf=DroneConfig.SMOKE_CONF, imgsz=640)
                for fr in f_res:
                    for b in fr.boxes:
                        cls = int(b.cls[0])
                        conf = float(b.conf[0])
                        if cls == 0 and conf < DroneConfig.FIRE_CONF: continue
                        if cls == 1 and conf < DroneConfig.SMOKE_CONF: continue
                        xyxy = list(map(int, b.xyxy[0].cpu().numpy()))
                        fires.append((cls, xyxy))
            with self.lock:
                self.result = res
                if fires is not None:
                    self.fire_objs = fires
            
            frame_count += 1
            if time.time() - last_time >= 1.0:
                self.fps = frame_count
                frame_count = 0
                last_time = time.time()

    def detect_gestures(self, img):
        return []

class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.prev_err = 0
        self.integral = 0
    def update(self, error):
        self.integral += error
        derivative = error - self.prev_err
        self.prev_err = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class HUDSystem:
    LANGUAGES = {
        'TR': {'STATUS': 'DURUM', 'BAT': 'BAT', 'ALT': 'YUK', 'AI': 'YZ', 'TARGET': 'HEDEF', 'TEMP': 'SIC', 'MSG_READY': 'HAZIR', 'MSG_LOADING': 'YUKLENIYOR', 'ACQUIRED': 'HEDEF KILITLENDI'},
        'EN': {'STATUS': 'STATUS', 'BAT': 'BAT', 'ALT': 'ALT', 'AI': 'AI', 'TARGET': 'TARGET', 'TEMP': 'TEMP', 'MSG_READY': 'READY', 'MSG_LOADING': 'LOADING', 'ACQUIRED': 'TARGET ACQUIRED'},
        'DE': {'STATUS': 'STATUS', 'BAT': 'BAT', 'ALT': 'HÖHE', 'AI': 'KI', 'TARGET': 'ZIEL', 'TEMP': 'TEMP', 'MSG_READY': 'BEREIT', 'MSG_LOADING': 'LADEN', 'ACQUIRED': 'ZIEL ERFASST'}
    }
    
    @staticmethod
    def get_str(key, lang='TR'):
        return HUDSystem.LANGUAGES.get(lang, HUDSystem.LANGUAGES['TR']).get(key, key)

    @staticmethod
    def draw_rounded_rect(img, pt1, pt2, color, thickness, r):
        x1, y1 = pt1; x2, y2 = pt2
        if thickness == -1:
            cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, -1)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, -1)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, -1)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, -1)
        else:
            cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)
            cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)
            cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)
            cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)
            for center, angle in [((x1+r, y1+r), 180), ((x2-r, y1+r), 270), ((x1+r, y2-r), 90), ((x2-r, y2-r), 0)]:
                cv2.ellipse(img, center, (r, r), angle, 0, 90, color, thickness)

    @staticmethod
    def draw_fighter_hud(frame, config, ds, ai_fps, ai_loaded):
        cx, cy = 480, 360
        cv2.circle(frame, (cx, cy), 15, (0, 255, 0), 1)
        overlay = frame.copy()
        HUDSystem.draw_rounded_rect(overlay, (12, 12), (240, 160), (0, 0, 0), -1, 10)
        HUDSystem.draw_rounded_rect(overlay, (750, 12), (948, 160), (0, 0, 0), -1, 10)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        font, sf, white, neon = cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), (0, 255, 0)
        lang = getattr(config, 'LANG', 'TR')
        
        cv2.putText(frame, f"{HUDSystem.get_str('STATUS', lang)}: {ds['msg']}", (22, 35), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 255), 1)
        bat = ds['bat']
        cv2.rectangle(frame, (23, 50), (120, 55), (50, 50, 50), -1)
        cv2.rectangle(frame, (23, 50), (23 + int(bat * 0.97), 55), neon if bat > 15 else (0,0,255), -1)
        cv2.putText(frame, f"{HUDSystem.get_str('BAT', lang)}: %{bat}", (125, 55), font, sf, white, 1)
        cv2.putText(frame, f"{HUDSystem.get_str('ALT', lang)}: {ds['h']}cm", (22, 80), font, sf, white, 1)
        cv2.putText(frame, f"B-TOF: {ds.get('tof', 0)}cm", (22, 90), font, sf, white, 1)
        


        ai_status = HUDSystem.get_str('MSG_READY' if ai_loaded else 'MSG_LOADING', lang)
        cv2.putText(frame, f"{HUDSystem.get_str('AI', lang)}: {ai_status}", (22, 100), font, sf, neon if ai_loaded else (0, 165, 255), 1)
        cv2.putText(frame, f"FPS: {ai_fps}", (22, 120), font, sf, (0, 200, 255), 1)
        cv2.putText(frame, f"{HUDSystem.get_str('TARGET', lang)}: {str(ds['target']).upper()}", (22, 145), font, sf, (255, 165, 0), 1)
        
        target_name = str(ds['target']).upper()
        if target_name != "NONE":
            t_color = (0, 255, 0) if "LOCK" in ds['msg'] else (0, 200, 255)
            cv2.putText(frame, HUDSystem.get_str('ACQUIRED', lang), (380, 650), font, 0.6, t_color, 1)
            cv2.putText(frame, target_name, (360, 690), cv2.FONT_HERSHEY_DUPLEX, 1.2, t_color, 2)
            
        cv2.putText(frame, "TELEMETRY", (760, 35), cv2.FONT_HERSHEY_DUPLEX, 0.45, white, 1)
        cv2.putText(frame, f"VX: {ds['vx']}", (760, 60), font, sf, white, 1)
        cv2.putText(frame, f"VY: {ds['vy']}", (760, 85), font, sf, white, 1)
        cv2.putText(frame, f"VZ: {ds['vz']}", (760, 110), font, sf, white, 1)
        ext_tof_cm = ds.get('ext_tof', 0) / 10.0
        cv2.putText(frame, f"F-TOF: {ext_tof_cm:.1f}cm", (760, 135), font, sf, neon if ext_tof_cm > 1 else (0,0,255), 1)
        cv2.putText(frame, f"{HUDSystem.get_str('TEMP', lang)}: {ds['temp']}C", (760, 160), font, sf, neon if ds['temp'] < 85 else (0,0,255), 1)

    @staticmethod
    def draw_fire_warning(frame, warning_type="FIRE"):
        h, w = frame.shape[:2]
        color = (0, 0, 200) if warning_type == "FIRE" else (0, 140, 255)
        cv2.rectangle(frame, (w//2-180, h//2-40), (w//2+180, h//2+40), color, -1)
        cv2.putText(frame, f"!!! {warning_type} !!!", (w//2-150, h//2+15), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

class OtonomSistem:
    def __init__(self):
        self.gorevler = {}
        self.ogrenci_mled_dongusu = None
        self.mled_yazi_str = "FURKAN"
        self.mled_yazi_color = "b"
        self.cfg = drone_config
        base_dir = os.path.dirname(os.path.abspath(__file__))
        main_path = os.path.join(base_dir, "best.pt")
        fire_path = os.path.join(base_dir, "fire.pt")
        self.ai_worker = AIWorker(main_path=main_path, fire_path=fire_path)
        self.tello = Tello()
        
        # Tello send_control_command metodunu soket hatalarına karşı koru ve başlangıçta tamponla
        orig_send_control_command = self.tello.send_control_command
        def safe_send_control_command(command, *args, **kwargs):
            command = self._normalize_mled_command(command)
            if "mled" in command.lower():
                self.tello.buffered_control_command = command
                # Eğer bağlı değilsek sadece tamponda tutalım, hata vermeyelim
                if not self.is_connected:
                    print(f"[SYS] Matrix komutu bağlantı kurulana kadar tamponlandı: {command}")
                    return "ok"
            try:
                with self.cmd_lock:
                    return orig_send_control_command(command, *args, **kwargs)
            except Exception as e:
                print(f"[SYS-CMD-ERROR] Komut hatası: {e}")
                return "error"
        self.tello.send_control_command = safe_send_control_command
        self.tello.buffered_control_command = None

        self.running = True
        self.is_flying = False
        self.is_busy = False
        self.is_connected = False
        self.is_stream_ok = False
        self.is_moving = False
        self.frame_read = None
        self.bbox_history = deque(maxlen=3)
        self.class_history = deque(maxlen=3)
        self.pid_x = PID(0.04, 0.0, 0.02)
        self.pid_y = PID(0.05, 0.0, 0.02)
        self.last_seen_time = time.time()
        self.wait_start_time = 0
        self.cmd_lock = threading.RLock()
        self.data_lock = threading.Lock() 
        self.state = "SEARCHING"
        self.telemetry = {'bat': 0, 'h': 0, 'vx': 0, 'vy': 0, 'vz': 0, 'temp': 0, 'target': 'NONE', 'msg': 'INIT...', 'ext_tof': 0, 'tof': 0, 'mled_color': 'OFF'}
        self.fire_detected = False
        
        # Camera Watchdog variables
        self.last_frame_hash = None
        self.last_frame_change_time = time.time()
        self.last_stream_restart_time = 0
        
        # Log Ayarları
        self.log_file = None
        if not os.path.exists("logs"): os.makedirs("logs")
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.log_path = f"logs/flight_{timestamp}.csv"
        self.vid_path = f"logs/video_{timestamp}.avi"
        self.video_writer = None
        
        # TTS Hazirlik
        self.tts_engine = None
        if HAS_TTS:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
            except: 
                pass
    def hedefte(self, hedef_ismi):
        """Kullanıcıların hedeflere görev atamasını sağlayan Decorator."""
        def decorator(fonk):
            self.gorevler[hedef_ismi.lower()] = fonk
            return fonk
        return decorator

    def mled_dongusu(self, fonk):
        """Öğrencilerin 8x8 Dot Matrix LED döngüsünü kontrol etmesini sağlayan Decorator."""
        self.ogrenci_mled_dongusu = fonk
        return fonk

    def ekrana_yaz(self, yazi, renk="b"):
        """Ekranda kayacak yazıyı belirler. Örneğin: drone.ekrana_yaz("FURKAN")"""
        self.mled_yazi_str = str(yazi)
        self.mled_yazi_color = str(renk)

    def yazdir(self, yazi, renk="b"):
        """ekrana_yaz ile aynı işlevi gören Türkçe alternatif."""
        self.ekrana_yaz(yazi, renk)

    def mled_yazi(self, yazi, renk="b"):
        """ekrana_yaz ile aynı işlevi gören alternatif."""
        self.ekrana_yaz(yazi, renk)

    def baslat(self):
        """Öğrencilerin sistemi başlatması için Türkçe metod."""
        self.start()

    def start(self):
        print("[SYS] Sistem Baslatiliyor...")
        self.ai_worker.start()
        threading.Thread(target=self.connection_worker, daemon=True).start()
        threading.Thread(target=self.tof_worker, daemon=True).start()
        threading.Thread(target=self.logic_loop, daemon=True).start()
        threading.Thread(target=self.logging_worker, daemon=True).start()
        threading.Thread(target=self.matrix_led_worker, daemon=True).start()
        threading.Thread(target=self.mled_cycle_worker, daemon=True).start()
        self.ui_loop()

    def matrix_led_worker(self):
        print("[SYS] Matrix LED Worker Baslatildi.")
        son_durum = None # "YAKIN" veya "UZAK"
        while self.running:
            if self.is_connected:
                try:
                    with self.data_lock:
                        tof = self.telemetry.get('tof', 0)
                    
                    if tof < self.cfg.PROXIMITY_THRESHOLD_CM:
                        durum = "YAKIN"
                    else:
                        durum = "UZAK"
                    
                    if durum != son_durum:
                        son_durum = durum
                        with self.data_lock:
                            self.telemetry['mled_color'] = "RED" if durum == "YAKIN" else "GREEN"
                        
                        if durum == "YAKIN":
                            print(f"[LED] Mesafe Yere Yakın ({tof} cm) -> LED: KIRMIZI")
                            with self.cmd_lock:
                                self.tello.send_command_without_return("EXT led 255 0 0")
                        else:
                            print(f"[LED] Mesafe Yere Uzak ({tof} cm) -> LED: YEŞİL")
                            with self.cmd_lock:
                                self.tello.send_command_without_return("EXT led 0 255 0")
                except Exception as e:
                    print(f"[LED] Hata: {e}")
            time.sleep(0.5)

    def mled_cycle_worker(self):
        print("[SYS] Matrix LED Döngü Çalışanı Başlatıldı.")
        while self.running:
            if self.is_connected:
                try:
                    # Eğer djitellopy Tello nesnesinde tamponlanmış komut varsa onu öncelikli gönder
                    buffered_cmd = getattr(self.tello, 'buffered_control_command', None)
                    if buffered_cmd:
                        parts = buffered_cmd.split(' ')
                        if len(parts) >= 6 and parts[1] == "mled" and parts[2] in ['l', 'r', 'u', 'd']:
                            color = parts[3]
                            try:
                                speed = float(parts[4])
                            except:
                                speed = 2.0
                            text = " ".join(parts[5:])
                            
                            # Yazıyı sütunlara dönüştür
                            all_cols = []
                            for char in text:
                                if char == 'Ş': char = 'S'
                                elif char == 'Ç': char = 'C'
                                elif char == 'Ğ': char = 'G'
                                elif char == 'Ö': char = 'O'
                                elif char == 'Ü': char = 'U'
                                elif char == 'İ': char = 'I'
                                elif char == 'ı': char = 'I'
                                
                                pattern = FONT.get(char, FONT[' '])
                                for col in range(8):
                                    col_bits = [(pattern[r] >> (7 - col)) & 1 for r in range(8)]
                                    all_cols.append(col_bits)
                                all_cols.append([0]*8)
                                
                            padding = [[0]*8 for _ in range(8)]
                            all_cols = padding + all_cols + padding
                            
                            delay = 0.2 / speed
                            i = 0
                            while i <= len(all_cols) - 8 and self.running:
                                current_buffered = getattr(self.tello, 'buffered_control_command', None)
                                if current_buffered != buffered_cmd:
                                    break
                                    
                                window = all_cols[i : i+8]
                                grid_chars = []
                                for r in range(8):
                                    for c in range(8):
                                        bit = window[c][r]
                                        grid_chars.append(color if bit else '0')
                                grid_string = "".join(grid_chars)
                                
                                try:
                                    self.tello.send_command_without_return(f"EXT mled g {grid_string}")
                                except:
                                    pass
                                i += 1
                                time.sleep(delay)
                            continue
                        else:
                            self.send_command(buffered_cmd)
                            time.sleep(2.0)
                            continue

                    # Eğer öğrencinin tanımladığı özel bir döngü fonksiyonu varsa onu çalıştır
                    if getattr(self, 'ogrenci_mled_dongusu', None) is not None:
                        with self.data_lock:
                            ext_tof_cm = self.telemetry.get('ext_tof', 0) / 10.0
                            bat = self.telemetry.get('bat', 0)
                        try:
                            self.ogrenci_mled_dongusu(self.tello, ext_tof_cm, bat)
                        except Exception as e:
                            print(f"[MLED-OGRENCI] Hata: {e}")
                        time.sleep(2.0)
                        continue

                    yazi = getattr(self, 'mled_yazi_str', "F")
                    color = getattr(self, 'mled_yazi_color', "b")
                    if len(yazi) == 1:
                        # Tek karakter ise statik olarak göster
                        self.send_command(f"EXT mled s {color} {yazi}")
                    else:
                        # Çoklu karakter ise kaydırarak göster
                        text = yazi
                        all_cols = []
                        for char in text:
                            if char == 'Ş': char = 'S'
                            elif char == 'Ç': char = 'C'
                            elif char == 'Ğ': char = 'G'
                            elif char == 'Ö': char = 'O'
                            elif char == 'Ü': char = 'U'
                            elif char == 'İ': char = 'I'
                            elif char == 'ı': char = 'I'
                            
                            pattern = FONT.get(char, FONT[' '])
                            for col in range(8):
                                col_bits = [(pattern[r] >> (7 - col)) & 1 for r in range(8)]
                                all_cols.append(col_bits)
                            all_cols.append([0]*8)
                            
                        padding = [[0]*8 for _ in range(8)]
                        all_cols = padding + all_cols + padding
                        
                        delay = 0.2 / 2.0
                        i = 0
                        while i <= len(all_cols) - 8 and self.running:
                            current_buffered = getattr(self.tello, 'buffered_control_command', None)
                            if current_buffered:
                                break
                            if getattr(self, 'mled_yazi_str', "F") != yazi:
                                break
                                
                            window = all_cols[i : i+8]
                            grid_chars = []
                            for r in range(8):
                                for c in range(8):
                                    bit = window[c][r]
                                    grid_chars.append(color if bit else '0')
                            grid_string = "".join(grid_chars)
                            
                            try:
                                self.tello.send_command_without_return(f"EXT mled g {grid_string}")
                            except:
                                pass
                            i += 1
                            time.sleep(delay)
                        continue
                except Exception as e:
                    print(f"[MLED-CYCLE] Hata: {e}")
            time.sleep(2.0)

    def tof_worker(self):
        while self.running:
            if self.is_connected and not self.is_busy and not self.is_moving:
                try:
                    with self.cmd_lock:
                        tof_str = self.tello.send_read_command("EXT tof?")
                    if tof_str and "error" not in tof_str.lower() and "ok" not in tof_str.lower():
                        digit_str = "".join([c for c in tof_str if c.isdigit() or c == '-'])
                        if digit_str:
                            val = int(digit_str)
                            if 0 < val < 3000:
                                with self.data_lock: self.telemetry['ext_tof'] = val
                except: 
                    pass
            time.sleep(1.0)

    def connection_worker(self):
        while self.running:
            if not self.is_connected:
                try:
                    self.telemetry['msg'] = "BAGLANILIYOR..."
                    self.tello.connect()
                    self.is_connected = True
                    self.tello.streamon()
                    self.frame_read = self.tello.get_frame_read()
                    print("[CONN] Drone baglandi.")
                except:
                    self.telemetry['msg'] = "WI-FI KONTROL ET!"
                    time.sleep(2.0); continue
            if self.is_connected and self.frame_read:
                raw = self.frame_read.frame
                if raw is not None and raw.size > 0:
                    if not self.is_stream_ok:
                        self.is_stream_ok = True
                        self.telemetry['msg'] = "AKTIF"
                else: 
                    self.is_stream_ok = False
            time.sleep(0.5)

    def restart_stream(self):
        print("[SYS-WARN] Video stream frozen! Restarting Tello camera stream...")
        try:
            with self.cmd_lock:
                self.tello.streamoff()
                time.sleep(0.5)
                self.tello.streamon()
                time.sleep(0.5)
            self.frame_read = self.tello.get_frame_read()
            self.last_frame_change_time = time.time()
            self.last_frame_hash = None
            print("[SYS] Video stream restarted successfully.")
        except Exception as e:
            print(f"[SYS-ERROR] Failed to restart stream: {e}")

    def get_corrected_direction(self, frame, xyxy, name):
        if name in ['sol', 'sag']: return name
        if name not in ['soladon', 'sagadon']: return name
        try:
            x1, y1, x2, y2 = map(int, xyxy)
            W, H = x2 - x1, y2 - y1
            px, py = int(W * 0.20), int(H * 0.20)
            crop = frame[max(0, y1+py):min(720, y2-py), max(0, x1+px):min(960, x2-px)]
            if crop.size < 50: return name
            _, mask = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            h, w = mask.shape
            if 'don' in name:
                bottom_strip = mask[int(h*0.75):, :]
                l_sum = np.sum(bottom_strip[:, :w//2])
                r_sum = np.sum(bottom_strip[:, w//2:])
                return 'soladon' if r_sum > l_sum else 'sagadon'
            else:
                left_strip = mask[:, :int(w*0.20)]
                right_strip = mask[:, -int(w*0.20):]
                l_sum = np.sum(left_strip)
                r_sum = np.sum(right_strip)
                return 'sol' if l_sum > r_sum else 'sag'
        except:
            return name

    def logging_worker(self):
        """Uçuş verilerini CSV dosyasına kaydeder."""
        print(f"[LOG] Veri kaydi basladi: {self.log_path}")
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("Time,Battery,Height,Temp,VX,VY,VZ,Target,State\n")
            
        while self.running:
            if self.is_connected:
                with self.data_lock:
                    t = self.telemetry
                    log_line = f"{time.strftime('%H:%M:%S')},{t['bat']},{t['h']},{t['temp']},{t['vx']},{t['vy']},{t['vz']},{t['target']},{self.state}\n"
                
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(log_line)
            time.sleep(1.0)

    def speak(self, text):
        """Sesli geri bildirim verir."""
        if HAS_TTS and self.tts_engine:
            def run():
                try: self.tts_engine.say(text); self.tts_engine.runAndWait()
                except: pass
            threading.Thread(target=run, daemon=True).start()

    def logic_loop(self):
        while self.running:
            if hasattr(self.tello, 'state') and hasattr(self.tello.state, 'takeoff_received'):
                if not self.is_flying and self.tello.state.takeoff_received:
                    print("[AI] BAŞLAT Komutu Alındı, Kalkış Yapılıyor...")
                    self.tello.takeoff()
                    self.is_flying = True
                    self.tello.state.takeoff_received = False
            ai_res, fire_objs, ai_fps, ai_loaded = self.ai_worker.get_results()
            if not self.is_stream_ok or not ai_loaded:
                time.sleep(0.1); continue
            frame_rgb = self.frame_read.frame
            if frame_rgb is None: continue
            self.ai_worker.set_frame(frame_rgb)
            
            # Watchdog to detect frozen frame
            sample = frame_rgb[::20, ::20, 0].tobytes()
            f_hash = hash(sample)
            if self.last_frame_hash is None or f_hash != self.last_frame_hash:
                self.last_frame_hash = f_hash
                self.last_frame_change_time = time.time()
                
            is_frozen = (time.time() - self.last_frame_change_time) > 2.5
            if is_frozen:
                self.telemetry['msg'] = "CAMERA FROZEN"
                if time.time() - self.last_stream_restart_time > 8.0:
                    self.last_stream_restart_time = time.time()
                    threading.Thread(target=self.restart_stream, daemon=True).start()
            
            ai_res, fire_objs, ai_fps, ai_loaded = self.ai_worker.get_results()
            gestures = self.ai_worker.gesture_objs
            
            # If camera is frozen and we are already locking/waiting, bypass detection update to keep target
            if is_frozen and self.state == "WAITING":
                pass
            else:
                best_det = None
                if is_frozen:
                    pass
                elif fire_objs:
                    has_fire = any(f[0] == 0 for f in fire_objs)
                    has_smoke = any(f[0] == 1 for f in fire_objs)
                    fire_objs.sort(key=lambda x: (x[1][2]-x[1][0])*(x[1][3]-x[1][1]), reverse=True)
                    _, f_box = fire_objs[0]
                    if has_fire and has_smoke:
                        f_name = "fire & smoke"
                    else:
                        f_name = "fire" if fire_objs[0][0] == 0 else "smoke"
                    best_det = (f_name, f_box, 0.99) 
                elif gestures:
                    g_name, g_box = gestures[0][2], gestures[0][1]
                    best_det = (g_name, g_box, 0.95)
                elif ai_res and len(ai_res[0].boxes) > 0:
                    dets = []
                    for box in ai_res[0].boxes:
                        name = self.ai_worker.model.names[int(box.cls[0])]
                        if name == 'assagi': name = 'asagi'
                        if name == 'takla' and float(box.conf[0]) < 0.92: continue
                        dets.append((name, box.xyxy[0].cpu().numpy(), float(box.conf[0])))
                    if dets:
                        dets.sort(key=lambda x: (x[1][2]-x[1][0])*(x[1][3]-x[1][1]), reverse=True)
                        target_name, target_box, target_conf = dets[0][0], dets[0][1], dets[0][2]
                        c_name = self.get_corrected_direction(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY), target_box, target_name)
                        best_det = (c_name, target_box, target_conf)    
                if best_det:
                    self.last_seen_time = time.time()
                    if best_det[0] in ['fire', 'smoke', 'face']:
                        self.telemetry['target'] = best_det[0]
                        self.bbox_history.append(best_det[1])
                        if time.time() - self.last_seen_time > 5: # 5 saniyede bir sesli uyar
                            self.speak(f"{best_det[0]} detected")
                    else:
                        self.class_history.append(best_det[0])
                        if self.class_history.count(best_det[0]) >= 2:
                            self.telemetry['target'] = best_det[0]
                            self.bbox_history.append(best_det[1])
                        else:
                            self.state = "EXAMINING"
                            if self.is_flying and not self.is_busy:
                                with self.cmd_lock: self.tello.send_rc_control(0,0,0,0)
                            time.sleep(0.04)
                            continue
                else:
                    self.class_history.append(None)
                    if all(x is None for x in self.class_history): 
                        self.telemetry['target'] = "NONE"
                        self.bbox_history.clear()
            if not self.is_flying or self.is_busy:
                time.sleep(0.05); continue
            with self.data_lock:
                current_bat = self.telemetry.get('bat', 100)
            if current_bat < self.cfg.BATTERY_FAILSAFE and self.is_flying:
                self.telemetry['msg'] = "CRITICAL BATTERY! LANDING"
                self.tello.land()
                self.is_flying = False
                continue
            target = self.telemetry['target']
            with self.data_lock:
                bbox_len = len(self.bbox_history)
                if target != "NONE" and bbox_len > 0:
                    box = np.mean(self.bbox_history, axis=0)
                else:
                    box = None
            if box is not None:
                cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                err_x, err_y = cx - self.cfg.CENTER_X, cy - self.cfg.CENTER_Y
                is_aligned_xy = abs(err_x) < self.cfg.HORIZONTAL_SENSITIVITY and abs(err_y) < self.cfg.VERTICAL_SENSITIVITY
                
                with self.data_lock:
                    dist_cm = self.telemetry.get('ext_tof', 0) / 10.0
                
                bw = box[2]-box[0]
                is_close_enough = False
                
                IDEAL_BW_MIN = 180
                IDEAL_BW_MAX = 450
                
                if not SIMULATION and (10 <= dist_cm < 300):
                    err_dist = dist_cm - self.cfg.TARGET_IDEAL_DISTANCE_CM
                    if abs(err_dist) <= 15:
                        is_close_enough = True
                else:
                    if IDEAL_BW_MIN <= bw <= IDEAL_BW_MAX:
                        is_close_enough = True
                        
                is_aligned = is_aligned_xy and is_close_enough

                if self.state == "WAITING":
                    prog = min(1.0, (time.time() - self.wait_start_time) / self.cfg.LOCK_DURATION)
                    self.telemetry['msg'] = f"LOCK: {int(prog*100)}%"
                    if prog >= 1.0: 
                        self.execute_command(target)
                        self.state = "SEARCHING"
                        with self.data_lock:
                            self.telemetry['target'] = "NONE"
                            self.bbox_history.clear()
                            self.class_history.clear()
                    with self.cmd_lock: self.tello.send_rc_control(0,0,0,0)
                else:
                    if is_aligned:
                        self.state, self.wait_start_time = "WAITING", time.time()
                        self.telemetry['msg'] = "LOCKING..."
                        with self.cmd_lock: self.tello.send_rc_control(0,0,0,0)
                    else:
                        self.state = "ALIGNING"
                        # PID Kontrollü Hizalama
                        lr = int(self.pid_x.update(err_x))
                        ud = int(self.pid_y.update(-err_y))
                        limit = self.cfg.ALIGNMENT_SPEED_LIMIT
                        lr = max(-limit, min(limit, lr))
                        ud = max(-limit, min(limit, ud))
                        
                        fb = 0
                        if not SIMULATION and (10 <= dist_cm < 300):
                            err_dist = dist_cm - self.cfg.TARGET_IDEAL_DISTANCE_CM
                            if err_dist > 15:
                                fb = max(6, min(limit, int(err_dist * 0.25)))
                            elif err_dist < -15:
                                fb = max(-limit, min(-6, int(err_dist * 0.25)))
                        else:
                            if bw < IDEAL_BW_MIN: fb = int(limit * 0.8)
                            elif bw > IDEAL_BW_MAX: fb = -int(limit * 0.8)
                        with self.cmd_lock: self.tello.send_rc_control(int(lr), int(fb), int(ud), 0)
            else:
                if time.time() - self.last_seen_time > self.cfg.SCAN_WAIT:
                    self.state = "HOVERING"
                    with self.cmd_lock: self.tello.send_rc_control(0,0,0, 0)
                else:
                    self.state = "SEARCHING"
                    with self.cmd_lock: self.tello.send_rc_control(0, self.cfg.SEARCH_SPEED, 0, 0)
            time.sleep(0.02)

    def ui_loop(self):
        cv2.namedWindow("Tello DeepSync Otonom - AMTAL")
        while self.running:
            raw = None
            if self.frame_read: raw = self.frame_read.frame
            if raw is not None and raw.size > 0:
                frame = raw.copy()
                frame = cv2.resize(frame, (960, 720))
                # Video Kaydı
                if self.is_flying:
                    if self.video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        self.video_writer = cv2.VideoWriter(self.vid_path, fourcc, 20.0, (960, 720))
                    self.video_writer.write(frame)
            else:
                frame = np.full((720, 960, 3), 30, dtype=np.uint8)
                cv2.putText(frame, "Asenkron YZ Baglaniliyor...", (300, 340), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(frame, "Drone Kamerasi Bekleniyor...", (300, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            if self.is_connected:
                try:
                    s = self.tello.get_current_state()
                    if s: 
                        self.telemetry.update({
                            'h': s['h'], 'vx': s['vgx'], 'vy': s['vgy'], 'vz': s['vgz'], 
                            'temp': s['temph'], 'bat': s.get('bat', self.telemetry['bat']),
                            'tof': s.get('tof', s['h'])
                        })
                except: pass
            ai_res, fire_objs, ai_fps, ai_loaded = self.ai_worker.get_results()
            has_fire = any(f[0] == 0 for f in fire_objs)
            has_smoke = any(f[0] == 1 for f in fire_objs)
            self.fire_detected = has_fire or has_smoke
            if ai_loaded and raw is not None:
                if ai_res and len(ai_res[0].boxes) > 0:
                    for box in ai_res[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        b_name = self.ai_worker.model.names[int(box.cls[0])]
                        b_conf = float(box.conf[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        cv2.putText(frame, f"{b_name.upper()} {b_conf:.2f}", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                with self.data_lock:
                    target = self.telemetry.get('target', 'NONE')
                    bbox_len = len(self.bbox_history)
                    if target != "NONE" and bbox_len > 0:
                        box = np.mean(self.bbox_history, axis=0).astype(int)
                    else: box = None
                if box is not None:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
            
            for f_cls, f_coord in fire_objs:
                if f_cls == 0:
                    f_color, label = (0, 0, 255), "FIRE"
                elif f_cls == 1:
                    f_color, label = (0, 165, 255), "SMOKE"
                else:
                    continue
                
                cv2.rectangle(frame, (f_coord[0], f_coord[1]), (f_coord[2], f_coord[3]), f_color, 2)
                cv2.putText(frame, label, (f_coord[0], f_coord[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, f_color, 1)
            
            HUDSystem.draw_fighter_hud(frame, self.cfg, self.telemetry, ai_fps, ai_loaded)
            if has_fire: HUDSystem.draw_fire_warning(frame, "FIRE")
            elif has_smoke: HUDSystem.draw_fire_warning(frame, "SMOKE")
            cv2.imshow("Tello DeepSync Otonom - AMTAL YZ", frame)
            if hasattr(self.tello, 'send_processed_frame'):
                self.tello.send_processed_frame(frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): self.terminate(); break
            elif key == ord('c'): self.is_connected = False
            elif key == ord('t') and self.is_connected and not self.is_flying:
                self.tello.takeoff(); self.is_flying = True
                self.last_seen_time = time.time() + 10
                with self.cmd_lock: self.tello.send_rc_control(0,0,20,0); time.sleep(1); self.tello.send_rc_control(0,0,0,0)
            elif key == ord('l') and self.is_flying:
                self.tello.land(); self.is_flying = False

    def execute_command(self, cmd):
        self.is_busy = True
        self.is_moving = True
        cmd_lower = cmd.lower()
        self.telemetry['msg'] = f"DOING: {cmd.upper()}"
        print(f"\n==============================================")
        print(f"[HEDEF KİLİTLENDİ] İşlem yapılıyor: {cmd.upper()}")
        print(f"==============================================\n")
        
        with self.cmd_lock:
            try: self.tello.send_rc_control(0,0,0,0); time.sleep(0.5)
            except: pass
            
            if cmd_lower in self.gorevler:
                try:
                    # Kullanıcının tanımladığı fonksiyonu çalıştır!
                    self.gorevler[cmd_lower](self.tello)
                except Exception as e:
                    print(f"[HATA] Görev çalıştırılırken hata: {e}")
            else:
                # Varsayılan modül komutlarını çalıştır!
                self.varsayilan_gorev_uygula(cmd_lower)
            
            try: 
                time.sleep(1.0)
                self.tello.send_rc_control(0,0,0,0)
            except: pass
        with self.data_lock:
            self.telemetry['target'] = "NONE"
            self.bbox_history.clear()
            self.class_history.clear()
        self.state = "SEARCHING"
        self.last_seen_time = time.time() + 1.0; self.is_busy = False; self.is_moving = False

    def varsayilan_gorev_uygula(self, cmd):
        """Modülün içine gömülü varsayılan hareket mantığı."""
        c = cmd.lower()
        cfg = self.cfg
        t = self.tello
        
        if c == 'sol':
            print(f"[MODÜL] Varsayılan: Sola gidiliyor ({cfg.LEFT_RIGHT_DISTANCE}cm)")
            t.move_left(cfg.LEFT_RIGHT_DISTANCE)
        elif c == 'sag':
            print(f"[MODÜL] Varsayılan: Sağa gidiliyor ({cfg.LEFT_RIGHT_DISTANCE}cm)")
            t.move_right(cfg.LEFT_RIGHT_DISTANCE)
        elif c == 'yukari':
            print(f"[MODÜL] Varsayılan: Yukarı çıkılıyor ({cfg.UPWARD_DISTANCE}cm)")
            t.move_up(cfg.UPWARD_DISTANCE)
        elif c == 'asagi':
            print(f"[MODÜL] Varsayılan: Aşağı iniliyor ({cfg.DOWNWARD_DISTANCE}cm)")
            t.move_down(cfg.DOWNWARD_DISTANCE)
        elif c == 'ileri':
            print(f"[MODÜL] Varsayılan: İleri gidiliyor ({cfg.FORWARD_DISTANCE}cm)")
            t.move_forward(cfg.FORWARD_DISTANCE)
        elif c == 'geri':
            print(f"[MODÜL] Varsayılan: Geri gidiliyor ({cfg.BACKWARD_DISTANCE}cm)")
            t.move_back(cfg.BACKWARD_DISTANCE)
        elif c == 'soladon':
            print(f"[MODÜL] Varsayılan: Sola dönülüyor ({cfg.ROTATION_ANGLE} derece)")
            t.rotate_counter_clockwise(cfg.ROTATION_ANGLE)
        elif c == 'sagadon':
            print(f"[MODÜL] Varsayılan: Sağa dönülüyor ({cfg.ROTATION_ANGLE} derece)")
            t.rotate_clockwise(cfg.ROTATION_ANGLE)
        elif c == 'don180':
            print("[MODÜL] Varsayılan: 180 derece dönülüyor...")
            t.rotate_clockwise(180)
        elif c == 'takla':
            print(f"[MODÜL] Varsayılan: Takla atılıyor!")
            t.flip_back()
        elif c == 'parkurson':
            print(f"[MODÜL] Varsayılan: Parkur bitti, iniş yapılıyor...")
            t.land()
        else:
            print(f"[UYARI] '{cmd}' hedefi için tanımlanmış bir görev bulunamadı!")

    def send_command(self, cmd):
        """Drona doğrudan Tello/RoboMaster SDK komutu gönderir."""
        cmd = self._normalize_mled_command(cmd)

        # Matrix LED komutlarında gereksiz tekrarları önleyerek ekranın sıfırlanmasını engelle
        # Ancak UDP paket kayıplarına karşı her 5 saniyede bir yeniden gönderilmesine izin ver
        # Önbellek kaydını sadece drone bağlıyken yapalım (bağlı değilken önbellek kirlenmesin)
        # Matrix LED komutlarında gereksiz tekrarları önleyerek ekranın sıfırlanmasını engelle.
        # Dron komutu aldığında kayan yazıyı zaten kendi donanımında sonsuz döngüde oynatır.
        # Bu yüzden aynı komutu tekrar tekrar gönderip yazıyı yarıda kesmeyelim.
        if self.is_connected and cmd.startswith("EXT mled"):
            last_cmd = getattr(self, '_son_mled_komut', None)
            if last_cmd == cmd:
                return "ok"
            self._son_mled_komut = cmd
            
        print(f"[SDK-CMD] Gonderiliyor: {cmd}")
        try:
            with self.cmd_lock:
                return self.tello.send_command_without_return(cmd)
        except Exception as e:
            print(f"[SDK-CMD] Hata (Gonderilemedi): {e}")
            if cmd.startswith("EXT mled"):
                self._son_mled_komut = None
            return "error"

    def _normalize_mled_command(self, cmd):
        # Küçük harfle girildiyse veya hatalı "s l/r/u/d" yazıldıysa otomatik düzelt
        if cmd.lower().startswith("ext"):
            cmd = "EXT" + cmd[3:]
            for yon in ['l', 'r', 'u', 'd']:
                if f"mled s {yon}" in cmd.lower():
                    idx = cmd.lower().find(f"mled s {yon}")
                    cmd = cmd[:idx] + f"mled {yon}" + cmd[idx + len(f"mled s {yon}"):]

        # Hız ve {EMOJI_HEART} gibi yer tutucuları düzelt
        parts = cmd.split(' ')
        if len(parts) >= 6 and parts[0] == "EXT" and parts[1] == "mled" and parts[2] in ['l', 'r', 'u', 'd']:
            try:
                speed = float(parts[4])
                if speed > 10.0:
                    parts[4] = "10.0"
                elif speed < 0.1:
                    parts[4] = "0.1"
            except ValueError:
                pass
            
            text = " ".join(parts[5:])
            text = text.replace("{EMOJI_HEART}", "♥")
            text = text.replace("{EMOJI_SMILE}", "☺")
            text = text.replace("{EMOJI_SAD}", "☹")
            text = text.replace("{EMOJI_ARROW_UP}", "↑")
            cmd = " ".join(parts[:5]) + " " + text
            
        return cmd


    def terminate(self):
        print("[SYS] Sistem Kapaniyor...")
        self.running = False
        if hasattr(self, 'ai_worker'): self.ai_worker.running = False
        try:
            self.tello.send_command_without_return("EXT mled g " + "0" * 64)
            self.tello.send_command_without_return("EXT led 0 0 0")
        except: pass
        try:
            if self.is_flying: self.tello.land()
            self.tello.streamoff()
        except: pass
        cv2.destroyAllWindows(); sys.exit(0)

# Kütüphane (Modül) Sonu
