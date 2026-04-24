from ultralytics import YOLO

def main():
    print("YOLOv8 Alev Algılama Modeli Eğitimi Başlıyor!")
    
    # En hafif ve hızlı YOLOv8 modelini kullanıyoruz (Drone üstünde anlık çalışması için)
    model = YOLO("yolov8n.pt")
    
    # Modeli Eğit (Sadece 10 epoch hızlı bir eğitim yapıyoruz)
    results = model.train(
        data="data.yaml",   # Kullanılacak veri setinin yolu
        epochs=10,          # Öğrenme tekrar sayısı (Hızlı sonuç için 10)
        imgsz=640,          # Görüntü kalitesi 
        batch=16,           # Aynı anda islenecek resim sayısı 
        device="cuda",      # Ekrana kartı varsa ('cuda') yoksa ('cpu') kullanılır
        project="Alev_Model",   # Kaydedilecek klasörün adı
        name="deneme1"      # Eğitimin adı
    )
    
    print("Eğitim tamamlandı! En iyi model ağırlıkları 'Alev_Model/deneme1/weights/best.pt' içerisindedir.")

if __name__ == '__main__':
    main()
