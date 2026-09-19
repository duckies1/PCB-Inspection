from ultralytics import YOLO

# model = YOLO.from_pretrained("Ultralytics/YOLO26")
model = YOLO("yolo26s.pt")  # load a pretrained model (recommended for training)
source = 'http://images.cocodataset.org/val2017/000000039769.jpg'
model.predict(source=source, save=True)