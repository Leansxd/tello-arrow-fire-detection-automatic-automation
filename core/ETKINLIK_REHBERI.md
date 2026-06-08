# 🚀 Tello Otonom Uçuş Modülü & 4 Günlük Eğitim Etkinliği Rehberi

Bu doküman, Almanya'da gerçekleştireceğiniz 4 günlük drone ve yapay zeka kodlama etkinliğiniz için özel olarak hazırlanmıştır. Temel yapay zeka mantığı (`tello_otonom.py` modülü) arka planda gizlenmiş olup, katılımcıların en kolay şekilde Python ile drone kodlamasını amaçlamaktadır.

---

## 🛠️ Modül (Kütüphane) Kullanımı

Öğrenciler (katılımcılar) karmaşık sistem kodlarına dokunmadan, sadece olay (event) bazlı basit fonksiyonlar yazacaklardır. Kütüphane, bir işareti tespit ettiğinde, ona ortalanacak ve uygun mesafeye geldiğinde otomatik olarak sizin atadığınız görevi tetikleyecektir.

### Basit Kod Şablonu:
```python
from tello_otonom import OtonomSistem

# 1. Sistemi Hazırla
drone = OtonomSistem()

# 2. Görevleri Tanımla (Decorator kullanımı)
@drone.hedefte("soladon")
def sola_don_gorevi(tello):
    print("Sola dönüş oku algılandı, dönülüyor!")
    tello.rotate_counter_clockwise(90)

@drone.hedefte("fire")
def yangin_sondur(tello):
    print("Yangın tespit edildi! Güvenlik mesafesine çekiliyor.")
    tello.move_back(50)

# 3. Uçuşu Başlat
drone.baslat()
```

---

## 📅 4 Günlük Almanya Etkinlik Planı (Workshop Konsepti)

Bu etkinlik planı, katılımcıların Python bilgisine giriş yapmasını ve son gün birbiriyle yarışacakları otonom bir görev tasarlamasını içerir.

### 🚩 1. Gün: "Python'a Giriş ve Tello ile İlk Tanışma"
* **Teori (1 Saat):** Python temel komutları, fonksiyon yazma (`def`), kütüphane çağırma (`import`). Drone nedir, yapay zeka dünyayı nasıl görür? (YOLO objelerinin kısa mantığı).
* **Pratik (2 Saat):** 
  * Katılımcılar bilgisayarlarına geçer ve simülatör (Three.js web arayüzünüz) üzerinden kod yazmaya başlar.
  * **İlk Görev:** Drone'u sadece basit komutlarla ("yukari" okunu görünce yukarı çık, "assagi" okunu görünce in) kontrol etmek.
  * *Amaç:* `drone.hedefte()` mantığını kavramak ve kendi fonksiyonlarını yazdıklarında drone'un tepki verdiğini görüp heyecanlanmak.

### 🚩 2. Gün: "Parkur Çözme ve Mantıksal Dönüşler"
* **Teori (45 Dk):** Decorator kavramının basitleştirilmiş anlatımı. Drone için mesafe ayarı ve açısal komutlar (30 cm git, 90 derece dön vs.).
* **Pratik (2 Saat):**
  * Salona fiziksel olarak veya simülatörün içine çeşitli yön okları ("soladon", "sagadon") yerleştirilir.
  * **İkinci Görev:** Katılımcılar, labirent gibi hazırlanmış bir parkuru (örneğin U şeklinde bir pist) hiç klavyeye dokunmadan, sadece yön oklarına verdikleri komutlarla otonom tamamlamaya çalışırlar.
  * *Ekstra (Bonus):* Yaratıcı gruplar, oku gördüğünde drone'a etrafında bir tam tur (`tello.rotate_clockwise(360)`) attırıp sonra dönebilir.

### 🚩 3. Gün: "Arama-Kurtarma (Search & Rescue) Simülasyonu"
* **Teori (45 Dk):** Otonom sistemlerde tehlike algısı, failsafe (hata koruması) ve yapay zeka güveni (AI Confidence).
* **Pratik (2 Saat):**
  * Odaya "Ateş" (fire) veya "Duman" (smoke) işaretleri/resimleri koyulur.
  * **Üçüncü Görev:** Senaryo gereği bir orman yangını simüle edilir. Drone devriye uçuşuna çıkar ("sag", "sol" görevleriyle dolaşır). Ateş gördüğü anda acil durum protokolü yazmaları istenir:
    ```python
    @drone.hedefte("fire")
    def acil_durum(tello):
        print("İTFAİYE ÇAĞRILDI!")
        tello.move_up(40) # Havadan durumu incele
        tello.move_back(60) # Geri kaç
    ```

### 🚩 4. Gün: "Büyük Final: Otonom Drone Yarışması!"
* **Format:** Tüm katılımcılar 3-4 kişilik takımlara ayrılır.
* **Kurallar:** 
  * Ekiplere karmaşık bir parkur haritası verilir. Bu haritada yön okları, "takla" işaretleri, sahte/tuzak yangınlar ve "parkurson" (bitiş) işareti vardır.
  * Ekipler 1 saat boyunca kendi `final_gorevi.py` kodlarını yazarlar. 
  * Hiç kimse uçuş sırasında bilgisayara veya kumandaya dokunamaz. 
  * Drone'un kalkıp ("T" tuşu ile başlatılır), doğru oklara bakarak labirenti çözmesi, yangında güvenli mesafeye çıkıp etrafından dolanması, "takla" işaretinde havada takla atıp jüriyi selamlaması ve "parkurson" işaretinde otonom iniş (`tello.land()`) yapması gerekir.
* **Kapanış:** En hızlı ve hatasız kodlayan takıma ödül verilir ve etkinlik sertifikalarla tamamlanır.

---

## 🎯 Eğitmenler İçin Kritik Notlar
* **Simülasyon Modu:** Kodun `tello_otonom.py` içerisindeki 11. satırında yer alan `SIMULASYON_MODU_ZORLA` ayarı, etkinlik sırasında bilgisayarda deneme yapılırken `True`, gerçek Tello ile salonda uçarken `False` yapılmalıdır. Etkinliğin ilk günleri güvenli ortamda simülatörle yapılıp 3. ve 4. gün gerçek drone'a geçiş yapılabilir.
* **Port Çakışması:** Sistem gerçek Tello (`djitellopy`) kullanırken bazen "WinError 10048" verebilir. Böyle bir durumda arkada açık kalan gizli Python terminalini kapatmanız veya görev yöneticisinden `python.exe`leri sonlandırmanız yeterlidir.
* **Drone Bataryası:** Tello'lar hızlı şarj tüketir. Etkinlik sırasında bol bol yedek batarya veya Powerbank şarj istasyonları (Hub) bulundurun.
