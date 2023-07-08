#%%

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO("best.pt")

# from ndarray
im2 = cv2.imread("sleeping_baby.jpg")
results = model.predict(source=im2, save=True, save_txt=True)
