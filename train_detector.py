from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    # train the model
    results = model.train(data="./datasets/detector/data.yaml", project="./", name="detector_models", epochs=15, imgsz=640, exist_ok=True)
