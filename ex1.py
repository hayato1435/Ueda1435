import cv2
import numpy as np
from ultralytics import YOLO

# ステップ1: 画像を読み込む
image = cv2.imread('ex1.jpg')

model = YOLO("yolov8x-pose.pt")

results = model("https://cs.kwansei.ac.jp/~kitamura/lecture/RyoikiJisshu/images/ex1.jpg",save_conf=True)

keypoints = results[0].keypoints

# 画像上のキーポイントの位置に丸を描く
for i in range(5,17): 
    cv2.circle(image, (int(keypoints.data[0][i][0]),int(keypoints.data[0][i][1])), 5, (22, 255,242.5), -1)

# 画像上のボーン（骨）の位置に直線を描く
bone_pairs=[(5,6),(5,7),(5,11),(6,8),(6,12),(7,9),(8,10),(11,12),(11,13),(12,14),(13,15),(14,16)]
for i in range(12):
   cv2.line(image, (int(keypoints.data[0][bone_pairs[i][0]][0]),int(keypoints.data[0][bone_pairs[i][0]][1])),
            (int(keypoints.data[0][bone_pairs[i][1]][0]),int(keypoints.data[0][bone_pairs[i][1]][1])),(0, 0, 255), 2)

# 仮のキーポイント座標
"""keypoints = {
    'left_shoulder': (600, 260),
    'right_shoulder': (665, 260),
    'left_elbow': (595, 180),
    'right_elbow': (670, 180),
    'left_wrist': (595, 100),
    'right_wrist': (675, 100),
    'left_hip': (610, 420),
    'right_hip': (650, 420),
    'left_knee': (610, 530),
    'right_knee': (650, 530),
    'left_ankle': (620, 650),
    'right_ankle': (655, 650)
}

# ステップ4: 画像上のキーポイントの位置に丸を描く
for key, point in keypoints.items():
    cv2.circle(image, point, 5, (22, 255,242.5), -1)

# ステップ5: 画像上のボーン（骨）の位置に直線を描く
bone_pairs = [('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow'),('left_shoulder', 'right_shoulder'),
              ('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist'), ('left_shoulder', 'left_hip'),
              ('right_shoulder', 'right_hip'), ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),('right_hip', 'left_hip'),
              ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle')]

for pair in bone_pairs:
    cv2.line(image, keypoints[pair[0]], keypoints[pair[1]], (0, 0, 255), 2)"""

# 画像を表示する
cv2.imshow('Skeleton', image)
cv2.waitKey(0)
cv2.destroyAllWindows()