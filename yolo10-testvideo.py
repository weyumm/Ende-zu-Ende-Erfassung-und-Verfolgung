import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLOv10
from deep_sort_realtime.deepsort_tracker import DeepSort
#.\venv\Scripts\activate
# ------------------- 配置参数 -------------------
MAX_TRACK_HISTORY = 30       # 轨迹保留的最大帧数
MAX_COLORS = 10              # 颜色种类
TRACK_ID_SIZE = 0.5          # ID 文字大小
MAX_POINT_DISTANCE = 100     # 轨迹点最大距离（防止误连）
# ------------------- 初始化 -------------------
# 加载 YOLOv10 检测模型
model = YOLOv10("yolov10n.pt")
# 初始化 DeepSort 跟踪器
tracker = DeepSort(max_age=30)
# 创建注释器对象
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
# 预定义颜色（BGR）
track_colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (128, 128, 0)
]
# 轨迹历史存储
track_history = {}
# 视频文件路径
video_path = "D:\\yolo10\\datavideo\\Euro Truck Simulator 2 2025-04-11 22-10-29.mp4"
# 打开视频文件
cap = cv2.VideoCapture(video_path)
# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Couldn't open the video file.")
    exit()
# 循环处理每一帧
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 使用 YOLOv10 进行目标检测
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    # 将检测结果转换为 DeepSort 所需格式
    raw_detections = []
    for i in range(len(detections.xyxy)):
        x1, y1, x2, y2 = detections.xyxy[i].tolist()
        bbox = [x1, y1, x2 - x1, y2 - y1]
        score = detections.confidence[i].item()
        class_id = int(detections.class_id[i].item())
        raw_detections.append([bbox, score, class_id])
    # 更新跟踪器
    tracks = tracker.update_tracks(raw_detections, frame=frame)
    # 提取所有跟踪 ID（包括未确认的）
    track_ids = [track.track_id for track in tracks]
    # 生成标签文本
    labels = [
        f"#{track_id} {model.names[class_id]} {confidence:.2f}"
        for class_id, confidence, track_id
        in zip(detections.class_id, detections.confidence, track_ids)
    ]
    # 标注图像（边界框 + 标签）
    annotated_image = bounding_box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    # 分别绘制轨迹图层
    trajectory_image = frame.copy()
    # 绘制轨迹和当前位置标记
    for track in tracks:
        if not track.is_confirmed():
            continue  # 只绘制已确认的跟踪
        # 确保 track_id 是整数
        try:
            track_id = int(track.track_id)
        except (TypeError, ValueError):
            print(f"Invalid track_id: {track.track_id}, skipping.")
            continue
        # 获取当前跟踪框 (x1, y1, x2, y2)
        bbox = track.to_tlbr()
        x_center = int((bbox[0] + bbox[2]) / 2)
        y_center = int((bbox[1] + bbox[3]) / 2)
        # 初始化轨迹历史
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append((x_center, y_center))
        # 限制轨迹长度
        if len(track_history[track_id]) > MAX_TRACK_HISTORY:
            track_history[track_id].pop(0)
        # 获取颜色
        color_idx = track_id % MAX_COLORS
        color = track_colors[color_idx]
        # 绘制轨迹
        trace = track_history[track_id]
        for j in range(1, len(trace)):
            pt1 = trace[j - 1]
            pt2 = trace[j]
            dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
            if dist < MAX_POINT_DISTANCE:
                cv2.line(trajectory_image, pt1, pt2, color, 2)
        # 绘制当前位置标记
        cv2.circle(trajectory_image, (x_center, y_center), 3, color, -1)
        cv2.putText(trajectory_image, f"{track_id}", (x_center + 5, y_center - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, TRACK_ID_SIZE, (255, 255, 255), 2)
    # 合并轨迹图层与标注图层
    combined_image = cv2.addWeighted(trajectory_image, 0.6, annotated_image, 0.4, 0)
    # 显示结果
    cv2.imshow('Video', combined_image)
    # 按 ESC 键退出
    if cv2.waitKey(1) % 256 == 27:
        print("Escape hit, closing...")
        break
# 释放资源
cap.release()
cv2.destroyAllWindows()