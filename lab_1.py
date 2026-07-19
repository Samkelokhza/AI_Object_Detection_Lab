""" This  code is used to detect all the  moving object that are being displayed on the screen """
from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

source = r"Cars Moving On Road Footage.mp4"  

if source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
    model.predict(source=source, conf=0.35, show=True, save=True)
else:
    model.track(source=source, tracker="bytetrack.yaml", conf=0.35, show=True, save=True)

print("Done. Check runs/ folder.")
