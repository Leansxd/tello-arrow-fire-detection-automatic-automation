# 🛰️ Tello DeepSync: Otonom Yapay Zeka ve Drone Programlama Master Rehberi

Bu döküman, DJI Tello dronelarını YOLOv8 yapay zeka modeli ve gelişmiş bilgisayarlı görü (Computer Vision) teknikleriyle otonom hale getiren **Tello DeepSync SDK**'sının kapsamlı kullanım kılavuzudur.

---

## 📑 İÇİNDEKİLER
1. [Giriş ve Proje Vizyonu](#1-giriş-ve-proje-vizyonu)
2. [Kurulum ve Yapılandırma](#2-kurulum-ve-yapılandırma)
3. [Sistem Mimarisi](#3-sistem-mimarisi)
4. [Modül Kullanımı ve API Referansı](#4-modül-kullanımı-ve-api-referansı)
5. [Otonom Görev Tasarımı](#5-otonom-görev-tasarımı)
6. [Gelişmiş Özellikler (Yüz, El, Log, Video)](#6-gelişmiş-özellikler)
7. [PID Stabilizasyon](#7-pid-stabilizasyon)
8. [Sorun Giderme](#8-sorun-giderme)

---

## 1. Giriş ve Proje Vizyonu
Tello DeepSync, droneların çevresini algılayabilen akıllı robotlar olduğunu göstermek amacıyla geliştirilmiştir. Yapay zeka ile fiziksel dünyayı birleştiren otonom sistemleri herkes için erişilebilir kılar.

---

## 2. Kurulum ve Yapılandırma
Sistemi çalıştırmak için terminalden şu bağımlılıkları yükleyin:
```bash
pip install opencv-python ultralytics djitellopy numpy pyttsx3
```
Ardından `core/drone_config.py` içerisinden `SIMULASYON_MODU_ZORLA` ayarını kontrol edin.

---

## 3. Sistem Mimarisi
Sistem **Multi-threading** mimarisiyle çalışır:
*   **AI Worker:** YOLOv8 çıkarımı, yüz ve el hareketi tespiti.
*   **Logic Loop:** Hedefe kilitlenme ve PID kontrolü.
*   **Logging:** Saniyelik telemetry ve video kaydı.

---

## 4. Modül Kullanımı ve API Referansı
`ogrenci_gorev_1.py` dosyasında `from core import OtonomSistem` diyerek sistemi başlatabilirsiniz.
*   `drone.baslat()`: Tüm döngüyü aktif hale getirir.
*   `tello.move_left(cm)`, `tello.flip_back()`, `tello.land()` gibi standart SDK komutlarını kullanabilirsiniz.

---

## 5. Otonom Görev Tasarımı
`@drone.hedefte("etiket")` dekoratörü ile drone belirli bir nesneyi gördüğünde ne yapacağını belirleyebilirsiniz.
```python
@drone.hedefte("sol")
def sola_git(tello):
    tello.move_left(50)
```

---

## 6. Gelişmiş Özellikler
*   👤 **Yüz Tanıma (`face`)**: Gerçek zamanlı yüz tespiti.
*   🖐️ **El Hareketi (`fist`, `open_hand`)**: El hareketleriyle drone kontrolü.
*   📂 **Kara Kutu (Logging)**: `logs/` klasörüne otomatik CSV ve Video kaydı.
*   🇩🇪 **Çok Dilli HUD**: Türkçe, İngilizce ve Almanca dil desteği.

---

## 7. PID Stabilizasyon
Drone, hedefleri merkezlemek için **PID (Proportional, Integral, Derivative)** algoritmasını kullanır. Bu sayede sarsıntısız ve hassas bir hizalama sağlar.

---

## 8. Sorun Giderme
*   **Adres kullanımda hatası:** Açık kalan Python süreçlerini kapatın.
*   **Görüntü gelmiyor:** Güvenlik duvarını kontrol edin (UDP portları).
*   **Düşük FPS:** Bilgisayarın güç modunu ve şarj durumunu kontrol edin.

---

> [!IMPORTANT]
> **Güvenlik Notu:** Uçuşlar sırasında drone'dan güvenli mesafede durun. Acil iniş için **"L"** tuşunu kullanın.
