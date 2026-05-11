import asyncio
import json
import base64
import cv2
import numpy as np
import threading
import websockets
import time

class BridgeState:
    def __init__(self):
        self.latest_frame = None
        self.websocket = None
        self.lock = threading.Lock()
        self.takeoff_received = False # JS'den T tuşu gelince True olur

class Tello:
    def __init__(self):
        self.state = BridgeState()
        self.loop = None
        threading.Thread(target=self._run_async_loop, daemon=True).start()
        print("[BRIDGE] Tello Nesnesi Başlatıldı.")

    def connect(self, wait_for_state=True):
        print("[BRIDGE] Simülatör bağlantısı bekleniyor...")
        start_time = time.time()
        while self.state.websocket is None:
            if time.time() - start_time > 10:
                print("[BRIDGE] HATA: Simülatöre bağlanılamadı! (Zaman aşımı)")
                return False
            time.sleep(0.1)
        print("[BRIDGE] BAĞLANDI! ✅")
        return True

    def get_frame_read(self):
        class Frame:
            def __init__(self, s): self.s = s
            @property
            def frame(self):
                with self.s.lock: return self.s.latest_frame
        return Frame(self.state)

    def takeoff(self):
        print("[BRIDGE] Kalkış yapılıyor...")
        self._send({"type": "takeoff"})
        time.sleep(2.0) # Kalkış süresi simülasyonu

    def land(self):
        print("[BRIDGE] İniş yapılıyor...")
        self._send({"type": "land"})
        time.sleep(2.0)

    def send_rc_control(self, left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity):
        self._send({
            "type": "rc", 
            "val": [
                int(left_right_velocity), 
                int(forward_backward_velocity), 
                int(up_down_velocity), 
                int(yaw_velocity)
            ]
        })

    # Gerçek Tello Komutları (cm bazlı varsayalım)
    def move_forward(self, dist_cm):
        print(f"[BRIDGE] İleri gidiliyor: {dist_cm}cm")
        self._send({"type": "move", "dir": "forward", "dist": dist_cm})

    def move_back(self, dist_cm):
        print(f"[BRIDGE] Geri gidiliyor: {dist_cm}cm")
        self._send({"type": "move", "dir": "back", "dist": dist_cm})

    def move_left(self, dist_cm):
        print(f"[BRIDGE] Sola gidiliyor: {dist_cm}cm")
        self._send({"type": "move", "dir": "left", "dist": dist_cm})

    def move_right(self, dist_cm):
        print(f"[BRIDGE] Sağa gidiliyor: {dist_cm}cm")
        self._send({"type": "move", "dir": "right", "dist": dist_cm})

    def move_up(self, dist_cm):
        print(f"[BRIDGE] Yukarı çıkılıyor: {dist_cm}cm")
        self._send({"type": "move", "dir": "up", "dist": dist_cm})

    def move_down(self, dist_cm):
        print(f"[BRIDGE] Aşağı iniliyor: {dist_cm}cm")
        self._send({"type": "move", "dir": "down", "dist": dist_cm})

    def rotate_clockwise(self, angle):
        print(f"[BRIDGE] Sağa dönülüyor: {angle} derece")
        self._send({"type": "rotate", "val": -angle}) # Three.js için sağ yön negatif

    def rotate_counter_clockwise(self, angle):
        print(f"[BRIDGE] Sola dönülüyor: {angle} derece")
        self._send({"type": "rotate", "val": angle}) # Three.js için sol yön pozitif

    def flip_back(self):
        print("[BRIDGE] Geriye takla atılıyor!")
        self._send({"type": "flip", "dir": "back"})
        time.sleep(1.5)

    def send_processed_frame(self, frame):
        """AI tarafından işlenen kareyi Web sitesine gönderir."""
        if self.state.websocket:
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            b64_frame = base64.b64encode(buffer).decode('utf-8')
            self._send({"type": "processed_frame", "data": "data:image/jpeg;base64," + b64_frame})

    def get_battery(self): return 90
    def get_height(self): return int(self.state.latest_alt * 100) if hasattr(self.state, 'latest_alt') else 0
    def get_current_state(self):
        return {
            'h': int(self.state.latest_alt * 100) if hasattr(self.state, 'latest_alt') else 0, 
            'vgx': int(self.state.latest_vx) if hasattr(self.state, 'latest_vx') else 0, 
            'vgy': int(self.state.latest_vy) if hasattr(self.state, 'latest_vy') else 0, 
            'vgz': 0, 
            'temph': 45, 
            'bat': 85
        }
    def send_read_command(self, cmd): 
        if "tof" in cmd.lower(): return "700"
        return "ok"

    def streamon(self): pass
    def streamoff(self): pass

    def _send(self, msg):
        if self.state.websocket and self.loop:
            try:
                self.loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(self.state.websocket.send(json.dumps(msg)))
                )
            except: pass

    def _run_async_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        async def start():
            async with websockets.serve(self._handler, "0.0.0.0", 9999):
                await asyncio.Future()
        self.loop.run_until_complete(start())

    async def _handler(self, ws):
        self.state.websocket = ws
        print("[BRIDGE] Simülatör Bağlantısı Kuruldu.")
        try:
            async for msg in ws:
                d = json.loads(msg)
                if d['type'] == 'frame':
                    img_data = base64.b64decode(d['data'].split(',')[1])
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        with self.state.lock: self.state.latest_frame = img
                if d.get('alt') is not None:
                    self.state.latest_alt = d['alt']
                    self.state.latest_vx = d.get('vx', 0)
                    self.state.latest_vy = d.get('vy', 0)
        except: pass
        finally: 
            if self.state.websocket == ws:
                self.state.websocket = None
        print("[BRIDGE] Simülatör Bağlantısı Kesildi.")
