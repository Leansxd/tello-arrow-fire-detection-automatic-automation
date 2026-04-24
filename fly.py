import cv2
import time
import threading
import numpy as np
import os
import sys
import logging
from collections import deque
from djitellopy import Tello
from ultralytics import YOLO

logging.getLogger('djitellopy').setLevel(logging.ERROR)

class DroneConfig:
    SAGA_SOLA_MESAFE    = 40
    ILERI_GITME_MESAFE  = 30
    GERI_GITME_MESAFE   = 50
    HEDEF_IDEAL_MESAFE_CM = 70
    YUKARI_GITME_MESAFE = 30
    ASAGI_GITME_MESAFE  = 40
    DONUS_ACISI         = 90
    
    BATTERY_FAILSAFE    = 10
    
    EKRAN_GENISLIK  = 960
    EKRAN_YUKSEKLIK = 720
    
    AI_GUVEN_ESIGI    = 0.45
    FIRE_CONF         = 0.55
    SMOKE_CONF        = 0.40
    AI_IMG_SIZE       = 640
    ARAMA_HIZI        = 22
    TARAMA_HIZI       = 25
    TARAMA_BEKLEME    = 2.5
    TETIKLEME_GENISLIK = 320
    MAX_YAKLASMA_HIZI = 18
    MIN_YAKLASMA_HIZI = 8
    
    YATAY_HASSASIYET = 220
    DIKEY_HASSASIYET = 230
    
    MIN_KUTU_GENISLIK = 0
    MAX_KUTU_GENISLIK = 900
    
    MERKEZ_X = EKRAN_GENISLIK // 2
    MERKEZ_Y = EKRAN_YUKSEKLIK // 2

class AIWorker(threading.Thread):
    def __init__(self, main_path="best.pt", fire_path="fire.pt"):
        super().__init__()
        self.daemon = True
        self.running = True
        self.is_loaded = False
        
        self.main_path = main_path
        self.fire_path = fire_path
        
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
        print(f"[AI] Modeller yukleniyor: {self.main_path}")
        try:
            self.model = YOLO(self.main_path, task='detect')
            self.fire_model = None
            if os.path.exists(self.fire_path):
                self.fire_model = YOLO(self.fire_path, task='detect')
            self.is_loaded = True
            print("[AI] SİSTEM HAZIR. Modeller basariyla yuklendi.")
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
            
            res = self.model.predict(img, verbose=False, conf=DroneConfig.AI_GUVEN_ESIGI, imgsz=DroneConfig.AI_IMG_SIZE)
            
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

class HUDSystem:
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
        cv2.putText(frame, f"STATUS: {ds['msg']}", (22, 35), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 255), 1)
        
        bat = ds['bat']
        cv2.rectangle(frame, (23, 50), (120, 55), (50, 50, 50), -1)
        cv2.rectangle(frame, (23, 50), (23 + int(bat * 0.97), 55), neon if bat > 15 else (0,0,255), -1)
        cv2.putText(frame, f"BAT: %{bat}", (125, 55), font, sf, white, 1)
        
        cv2.putText(frame, f"ALT: {ds['h']}cm", (22, 80), font, sf, white, 1)
        cv2.putText(frame, f"AI: {'READY' if ai_loaded else 'LOADING...'}", (22, 100), font, sf, neon if ai_loaded else (0, 165, 255), 1)
        cv2.putText(frame, f"FPS: {ai_fps}", (22, 120), font, sf, (0, 200, 255), 1)
        cv2.putText(frame, f"TARGET: {str(ds['target']).upper()}", (22, 145), font, sf, (255, 165, 0), 1)

        target_name = str(ds['target']).upper()
        if target_name != "NONE":
            t_color = (0, 255, 0) if "LOCK" in ds['msg'] else (0, 200, 255)
            cv2.putText(frame, "TARGET ACQUIRED", (380, 650), font, 0.6, t_color, 1)
            cv2.putText(frame, target_name, (360, 690), cv2.FONT_HERSHEY_DUPLEX, 1.2, t_color, 2)

        cv2.putText(frame, "TELEMETRY", (760, 35), cv2.FONT_HERSHEY_DUPLEX, 0.45, white, 1)
        cv2.putText(frame, f"VX: {ds['vx']}", (760, 60), font, sf, white, 1)
        cv2.putText(frame, f"VY: {ds['vy']}", (760, 85), font, sf, white, 1)
        cv2.putText(frame, f"VZ: {ds['vz']}", (760, 110), font, sf, white, 1)
        ext_tof_cm = ds.get('ext_tof', 0) / 10.0
        cv2.putText(frame, f"F-TOF: {ext_tof_cm:.1f}cm", (760, 135), font, sf, neon if ext_tof_cm > 1 else (0,0,255), 1)
        cv2.putText(frame, f"TEMP: {ds['temp']}C", (760, 160), font, sf, neon if ds['temp'] < 85 else (0,0,255), 1)

    @staticmethod
    def draw_fire_warning(frame, warning_type="FIRE"):
        h, w = frame.shape[:2]
        color = (0, 0, 200) if warning_type == "FIRE" else (0, 140, 255)
        cv2.rectangle(frame, (w//2-180, h//2-40), (w//2+180, h//2+40), color, -1)
        cv2.putText(frame, f"!!! {warning_type} !!!", (w//2-150, h//2+15), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

class TelloAutonomousApp:
    def __init__(self):
        self.cfg = DroneConfig()
        self.ai_worker = AIWorker()
        self.tello = Tello()
        self.running = True
        self.is_flying = False
        self.is_busy = False
        self.is_connected = False
        self.is_stream_ok = False
        self.is_moving = False
        
        self.frame_read = None
        self.bbox_history = deque(maxlen=3)
        self.class_history = deque(maxlen=3)
        self.last_seen_time = time.time()
        self.wait_start_time = 0
        self.cmd_lock = threading.Lock()
        self.data_lock = threading.Lock() 
        
        self.state = "SEARCHING"
        self.telemetry = {'bat': 0, 'h': 0, 'vx': 0, 'vy': 0, 'vz': 0, 'temp': 0, 'target': 'NONE', 'msg': 'INIT...', 'ext_tof': 0}
        self.fire_detected = False

    def start(self):
        print("[SYS] Sistem Baslatiliyor...")
        self.ai_worker.start()
        threading.Thread(target=self.connection_worker, daemon=True).start()
        threading.Thread(target=self.tof_worker, daemon=True).start()
        threading.Thread(target=self.logic_loop, daemon=True).start()
        self.ui_loop()

    def tof_worker(self):
        while self.running:
            if self.is_connected and self.is_flying and not self.is_busy and not self.is_moving:
                try:
                    with self.cmd_lock:
                        tof_str = self.tello.send_read_command("EXT tof?")
                    if tof_str and "error" not in tof_str.lower() and "ok" not in tof_str.lower():
                        digit_str = "".join([c for c in tof_str if c.isdigit() or c == '-'])
                        if digit_str:
                            val = int(digit_str)
                            if 0 < val < 3000:
                                with self.data_lock: self.telemetry['ext_tof'] = val
                except Exception as e: 
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
                except Exception as e:
                    print(f"[CONN] Baglanti Hatasi: Drone WIFI'a bagli misiniz?")
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

    def get_corrected_direction(self, frame, xyxy, name):
        if name not in ['sol', 'sag', 'soladon', 'sagadon']: return name
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
                
                return 'sol' if r_sum > l_sum else 'sag'
        except Exception as e:
            return name

    def logic_loop(self):
        while self.running:
            ai_res, fire_objs, ai_fps, ai_loaded = self.ai_worker.get_results()
            if not self.is_stream_ok or not ai_loaded:
                time.sleep(0.1); continue
                
            frame_rgb = self.frame_read.frame
            if frame_rgb is None: continue
            
            self.ai_worker.set_frame(frame_rgb)
            
            best_det = None
            
            if fire_objs:
                has_fire = any(f[0] == 0 for f in fire_objs)
                has_smoke = any(f[0] == 1 for f in fire_objs)
                
                fire_objs.sort(key=lambda x: (x[1][2]-x[1][0])*(x[1][3]-x[1][1]), reverse=True)
                _, f_box = fire_objs[0]
                
                if has_fire and has_smoke:
                    f_name = "fire & smoke"
                else:
                    f_name = "fire" if fire_objs[0][0] == 0 else "smoke"
                    
                best_det = (f_name, f_box, 0.99) 
            
            elif ai_res and len(ai_res[0].boxes) > 0:
                dets = []
                for box in ai_res[0].boxes:
                    name = self.ai_worker.model.names[int(box.cls[0])]
                    if name == 'takla' and float(box.conf[0]) < 0.92: continue
                    dets.append((name, box.xyxy[0].cpu().numpy(), float(box.conf[0])))
                if dets:
                    dets.sort(key=lambda x: (x[1][2]-x[1][0])*(x[1][3]-x[1][1]), reverse=True)
                    target_name, target_box, target_conf = dets[0][0], dets[0][1], dets[0][2]
                    
                    c_name = self.get_corrected_direction(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY), target_box, target_name)
                    best_det = (c_name, target_box, target_conf)    

            if best_det:
                self.last_seen_time = time.time()
                if best_det[0] in ['fire', 'smoke']:
                    self.telemetry['target'] = best_det[0]
                    self.bbox_history.append(best_det[1])
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
                print(f"[FAILSAFE] Kritik Batarya Seviyesi (%{current_bat}). Acil inis yapiliyor!")
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
                err_x, err_y = cx - self.cfg.MERKEZ_X, cy - self.cfg.MERKEZ_Y
                bw = box[2] - box[0]
                
                lr = int(err_x * 0.08)
                ud = int(-err_y * 0.08)
                lr = max(-25, min(25, lr))
                ud = max(-25, min(25, ud))
                
                dist_cm = self.telemetry.get('ext_tof', 0) / 10.0
                
                if 10 <= dist_cm < 400:
                    err_dist = dist_cm - self.cfg.HEDEF_IDEAL_MESAFE_CM
                    if err_dist > 10:
                        fb = max(self.cfg.MIN_YAKLASMA_HIZI, min(self.cfg.MAX_YAKLASMA_HIZI, int(err_dist * 0.3)))
                    else:
                        fb = 0
                else:
                    if bw < self.cfg.TETIKLEME_GENISLIK:
                        fb = self.cfg.MAX_YAKLASMA_HIZI - int((bw / self.cfg.TETIKLEME_GENISLIK) * 10)
                        fb = max(self.cfg.MIN_YAKLASMA_HIZI, fb)
                    else:
                        fb = 0

                is_centered = abs(err_x) < self.cfg.YATAY_HASSASIYET and abs(err_y) < self.cfg.DIKEY_HASSASIYET
                is_near = bw >= self.cfg.TETIKLEME_GENISLIK or (0 < dist_cm < self.cfg.HEDEF_IDEAL_MESAFE_CM + 10)

                if is_centered and is_near:
                    self.telemetry['msg'] = f"ACTION: {target.upper()}"
                    self.execute_command(target)
                else:
                    self.state = "APPROACHING"
                    self.telemetry['msg'] = f"FOLLOW: {target.upper()} | BW:{int(bw)}"
                    with self.cmd_lock: 
                        self.tello.send_rc_control(int(lr), int(fb), int(ud), 0)
            else:
                if time.time() - self.last_seen_time > self.cfg.TARAMA_BEKLEME:
                    self.state = "SCANNING"
                    self.telemetry['msg'] = "SEARCHING_TARGET..."
                    with self.cmd_lock: self.tello.send_rc_control(0, 0, 0, 15) 
                else:
                    self.state = "FORWARD_SEARCH"
                    self.telemetry['msg'] = "KEEP_FORWARD"
                    with self.cmd_lock: self.tello.send_rc_control(0, self.cfg.ARAMA_HIZI, 0, 0)
            time.sleep(0.02)

    def ui_loop(self):
        cv2.namedWindow("Tello DeepSync Otonom - AMTAL")
        while self.running:
            raw = None
            if self.frame_read: raw = self.frame_read.frame
            
            if raw is not None and raw.size > 0:
                frame = raw.copy()
                frame = cv2.resize(frame, (960, 720))
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
                            'temp': s['temph'], 'bat': s.get('bat', self.telemetry['bat'])
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
                f_color = (0, 0, 255) if f_cls == 0 else (0, 165, 255)
                cv2.rectangle(frame, (f_coord[0], f_coord[1]), (f_coord[2], f_coord[3]), f_color, 2)
                label = "FIRE" if f_cls == 0 else "SMOKE"
                cv2.putText(frame, label, (f_coord[0], f_coord[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, f_color, 1)

            HUDSystem.draw_fighter_hud(frame, self.cfg, self.telemetry, ai_fps, ai_loaded)
            
            if has_fire: HUDSystem.draw_fire_warning(frame, "FIRE")
            elif has_smoke: HUDSystem.draw_fire_warning(frame, "SMOKE")

            cv2.imshow("Tello DeepSync Otonom - AMTAL YZ", frame)
            
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
        self.telemetry['msg'] = f"DOING: {cmd.upper()}"
        print(f"[ONAY] Hareket komutu veriliyor: {cmd.upper()}")
        with self.cmd_lock:
            try: self.tello.send_rc_control(0,0,0,0); time.sleep(0.5)
            except: pass
            try:
                if cmd == 'soladon': self.tello.rotate_counter_clockwise(self.cfg.DONUS_ACISI)
                elif cmd == 'sagadon': self.tello.rotate_clockwise(self.cfg.DONUS_ACISI)
                elif cmd == 'yukari': self.tello.move_up(self.cfg.YUKARI_GITME_MESAFE)
                elif cmd == 'assagi': self.tello.move_down(self.cfg.ASAGI_GITME_MESAFE)
                elif cmd == 'takla': self.tello.move_up(20); self.tello.move_right(self.cfg.SAGA_SOLA_MESAFE)
                elif cmd == 'parkurson': self.tello.land(); self.running = False
                elif cmd == 'sol': self.tello.move_left(self.cfg.SAGA_SOLA_MESAFE)
                elif cmd == 'sag': self.tello.move_right(self.cfg.SAGA_SOLA_MESAFE)
                elif cmd in ['fire', 'smoke', 'fire & smoke']:
                    print(f"[ACTION] {cmd.upper()} mudahale pozisyonu alindi.")
                    self.tello.move_back(self.cfg.GERI_GITME_MESAFE)
            except Exception as e:
                print(f"[HATA] Komut hatasi (Timeout/Donanim): {e}")
                
            try: time.sleep(0.5); self.tello.send_rc_control(0,0,0,0)
            except: pass
        
        with self.data_lock:
            self.telemetry['target'] = "NONE"
            self.bbox_history.clear()
            self.class_history.clear()
        
        self.state = "SEARCHING"
        self.last_seen_time = time.time(); self.is_busy = False; self.is_moving = False

    def terminate(self):
        print("[SYS] Sistem Kapaniyor...")
        self.running = False
        if hasattr(self, 'ai_worker'): self.ai_worker.running = False
        try:
            if self.is_flying: self.tello.land()
            self.tello.streamoff()
        except: pass
        cv2.destroyAllWindows(); sys.exit(0)

if __name__ == "__main__":
    app = TelloAutonomousApp()
    try:
        app.start()
    except KeyboardInterrupt:
        app.terminate()
