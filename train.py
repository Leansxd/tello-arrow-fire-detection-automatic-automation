from ultralytics import YOLO

def main():
    print("YOLOv8 Alev Algılama Modeli Eğitimi Başlıyor!")
    
    model = YOLO("yolov8n.pt")
    
    results = model.train(
        data="data.yaml",
        epochs=10,
        imgsz=640,
        batch=16,
        device="cuda",
        project="Alev_Model",
        name="deneme1"
    )
    
    print("Eğitim tamamlandı! En iyi model ağırlıkları 'Alev_Model/deneme1/weights/best.pt' içerisindedir.")

if __name__ == '__main__':
    main()
