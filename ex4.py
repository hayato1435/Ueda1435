import cv2
import numpy as np
from ultralytics import YOLO

# ステップ1: 画像を読み込む
image = cv2.imread('ex4.jpg')
model = YOLO("yolov8x.pt")

results = model("https://cs.kwansei.ac.jp/~kitamura/lecture/RyoikiJisshu/images/ex4.jpg", save_conf=True)
boxes = results[0].boxes
for box in boxes:
     x1, y1, x2, y2 = map(int, box.xyxy[0])  # xyxyを使用してボックスの座標を取得
     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# ステップ5: 画像を保存する
cv2.imwrite('output_ex4.jpg', image)
   