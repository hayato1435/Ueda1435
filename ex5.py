import cv2
import numpy as np
from ultralytics import YOLO

"""image = cv2.imread('ex5.mp4')
model = YOLO("yolov8x.pt")

results = model("https://cs.kwansei.ac.jp/~kitamura/lecture/RyoikiJisshu/images/ex5.mp4", save_conf=True)
boxes = results[0].boxes
for box in boxes:
     x1, y1, x2, y2 = map(int, box.xyxy[0])  # xyxyを使用してボックスの座標を取得
     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# ステップ5: 画像を保存する
cv2.imwrite('output_ex5.mp4', image)"""


# ステップ1: 動画ファイルを読み込む
video_path = 'ex5.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO("yolov8x.pt")

# ステップ2: 動画の情報を取得
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# ステップ3: 出力動画の設定
output_path = 'output_ex5.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ステップ4: 各フレームに対してモデルを適用
    results = model(frame)

    # ステップ5: 検出結果を取得
    boxes = results[0].boxes

    # ステップ6: 検出された人物の領域を赤枠で囲む
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # xyxyを使用してボックスの座標を取得
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # ステップ7: フレームを出力動画に書き込む
    out.write(frame)

# ステップ8: リソースを解放
cap.release()
out.release()
cv2.destroyAllWindows()

