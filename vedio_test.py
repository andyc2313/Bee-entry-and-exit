import os
import json
from train_yolov8 import main

# 影片資料夾路徑
video_folder = "/home/bee/chihli_yolo8/yolov8/Video Converter Movavi"
output_json = "processing_results.json"

# 支援的影片副檔名
video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

def process_video(video_path):
    bee_in, bee_out = main(video_path)
    result = {
        "video": os.path.basename(video_path),
        "bee_in": bee_in,
        "bee_out": bee_out
    }
    print(f"處理結果：{result}")  # 調試訊息
    return result

def WriteJson(new_data, filepath):
    try:
        with open(filepath, 'r') as file:
            try:
                file_data = json.load(file)
            except json.JSONDecodeError:
                file_data = {}  # 如果檔案為空或損壞，初始化為空字典
    except FileNotFoundError:
        file_data = {}

    if "data" not in file_data:
        file_data["data"] = []

    file_data["data"].append(new_data)

    with open(filepath, 'w') as file:
        json.dump(file_data, file, indent=4, ensure_ascii=False)

# 收集影片路徑
video_files = [os.path.join(video_folder, f)
               for f in os.listdir(video_folder)
               if os.path.splitext(f)[1].lower() in video_extensions]

processed = set()

for video_path in video_files:
    video_name = os.path.basename(video_path)
    if video_name in processed:
        print(f"⚠️ 已處理過，跳過：{video_name}")
        continue

    try:
        result = process_video(video_path)
        if result:  # 確保有回傳有效資料
            WriteJson(result, output_json)  # 使用 WriteJson 儲存資料
            processed.add(video_name)
            print(f"✅ 處理完成：{video_name}，寫入結果：{result}")
        else:
            print(f"❌ 無效資料：{video_name}")
    except Exception as e:
        print(f"❌ 處理失敗：{video_path}，錯誤：{e}")
