from ultralytics import YOLO
import cv2
import numpy as np

# YOLOモデルの読み込み
model = YOLO("yolov8x-pose.pt")

# 基準画像のキーポイント取得
base_image = cv2.imread('ex1.jpg')
base_results = model('ex1.jpg')
base_keypoints = base_results[0].keypoints.data[0] if base_results[0].keypoints is not None else None

# 基準画像のキーポイントをCPUに移動してNumPy配列に変換
if base_keypoints is not None:
    base_keypoints = base_keypoints.cpu().numpy()
# キーポイント間の差異を計算する関数
def calculate_difference(kp1, kp2):
    if kp1 is None or kp2 is None:
        return float('inf')
    return np.linalg.norm(kp1 - kp2)

# 動画ファイルの読み込み
video_file = 'ex3a.mp4'
cap = cv2.VideoCapture(video_file)

# ビデオのプロパティを取得
fps = cap.get(cv2.CAP_PROP_FPS) #フレーム数
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# FPSが0の場合のエラーチェック
if fps == 0:
    print("Error: Failed to get FPS from video.")
    cap.release()
    exit()

# 出力ビデオの設定
output_file = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

#frame_interval = fps / 30
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 1/30秒ごとにフレームを処理
    #if frame_count % frame_interval == 0:
    results = model(frame)
    keypoints = results[0].keypoints.data[0] if results[0].keypoints is not None else None
    
    # フレームのキーポイントをCPUに移動してNumPy配列に変換
    if keypoints is not None:
        keypoints = keypoints.cpu().numpy()

        # 基準画像とのキーポイント差異を計算
    difference = calculate_difference(base_keypoints, keypoints)
        
        # 基準画像と同じ姿勢の場合、骨格を描画
    if difference < 20:  # 差異が小さい場合
            bone_pairs = [(5,6), (5,7), (5,11), (6,8), (6,12), (7,9), (8,10), (11,12), (11,13), (12,14), (13,15), (14,16)]
            for i in range(12):
                cv2.line(frame, 
                         (int(keypoints[bone_pairs[i][0]][0]), int(keypoints[bone_pairs[i][0]][1])),
                         (int(keypoints[bone_pairs[i][1]][0]), int(keypoints[bone_pairs[i][1]][1])),
                         (0, 0, 255), 2)
                #cv2.line(frame, (int(keypoints[0][bone_pairs[i][0]][0]),int(keypoints[0][bone_pairs[i][0]][1])),
            #(int(keypoints[0][bone_pairs[i][1]][0]),int(keypoints[0][bone_pairs[i][1]][1])),(0, 0, 255), 2)
        
    # フレームを出力ビデオに書き込む
    out.write(frame)
    
    #frame_count += 1
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()


#1,ex1.jpgの画像を読み込み、基準となるキーポイントを取得します。
#2,ex3a.mp4の動画から画像を切り取る 動画のFPSを取得し、1/30秒ごとにフレームを処理するための間隔を計算します。
#3,ex1.jpgを読み込む 各フレームのキーポイントを取得し、基準画像のキーポイントと比較します。
#4,切り取った画像が基準画像と同じ姿勢であれば、骨格を赤色で描画します。
#5,処理したフレームを新しい動画ファイルに書き込みます。
