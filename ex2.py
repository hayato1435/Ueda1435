from ultralytics import YOLO
import cv2
import numpy as np

# YOLOモデルの読み込み
model = YOLO("yolov8x-pose.pt")

# 画像ファイルリスト
image_files = [
    #'ex1.jpg',
    'ex2_307.jpg',
    'ex2_336.jpg',
    'ex2_2015.jpg',
    'ex2_3077.jpg',
    'ex2_5175.jpg'
]

# キーポイントを取得する関数
def get_keypoints(image_file):
    results = model(image_file, save_conf=True)
    keypoints = results[0].keypoints
    return keypoints.data[0] if keypoints is not None else None

# キーポイント間の差異を計算する関数
def calculate_difference(kp1, kp2):
    if kp1 is None or kp2 is None:
        return float('inf')
    return np.linalg.norm(kp1 - kp2)

# 基準画像のキーポイント取得
base_keypoints = get_keypoints('ex1.jpg')

# 各画像のキーポイントを取得し、基準画像との違いを計算
differences = []
for image_file in image_files:
    keypoints = get_keypoints(image_file)
    difference = calculate_difference(base_keypoints, keypoints)
    differences.append((image_file, difference))

# 違いが少ない順にファイル名を表示
differences.sort(key=lambda x: x[1])
for image_file, difference in differences:
    print(f"{image_file}: {difference}")

# 画像上のキーポイントの位置に丸を描く
def draw_keypoints(image_file, keypoints):
    image = cv2.imread(image_file)
    for i in range(len(keypoints)):
        x, y = int(keypoints[i][0]), int(keypoints[i][1])
        #cv2.circle(image, (x, y), 5, (22, 255, 242.5), -1)
    cv2.imshow(image_file, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 基準画像と他の画像のキーポイントを描画
#draw_keypoints('ex1.jpg', base_keypoints)
#for image_file, _ in differences:
    #keypoints = get_keypoints(image_file)
    #draw_keypoints(image_file, keypoints)