from ultralytics import YOLO
import cv2
import numpy as np
import math
from collections import deque
import time
import time

# 訓練YOLOv8模型 (這部分不需要在每次運行程式時執行)
model = YOLO("yolov8n.pt")  # 載入預訓練模型
# model.train(data='yolov8/dataset.yaml', epochs=100)  # 訓練模型，根據你的資料集設定路徑

class IDSwitchPreventer:
    def __init__(self, max_disappeared=5, max_distance=50, iou_threshold=0.3, 
                 history_size=5, direction_weight=0.2, min_history_for_direction=3):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.iou_threshold = iou_threshold
        self.history_size = history_size
        self.direction_weight = direction_weight  # 方向匹配的權重
        self.min_history_for_direction = min_history_for_direction  # 計算方向所需的最小歷史幀數

        self.track_history = {}  # {track_id: deque([...])}
        self.lost_tracks = {}    
        self.next_track_id = 0

    def update(self, current_detections, frame_idx):
        detection_candidates = list(current_detections.items())
        unmatched_detections = detection_candidates.copy()
        matched_tracks = set()
        corrected_detections = {}

        y_tolerance = 15  # 可調整的 Y 軸容錯範圍（像素）

        # 第一階段：與 track_history 匹配
        for track_id in list(self.track_history.keys()):
            history = self.track_history[track_id]
            if not history:
                continue

            last_data = history[-1]
            best_match = None
            best_score = float('inf')

            # 計算運動方向（若歷史足夠）
            track_direction = None
            if len(history) >= self.min_history_for_direction:
                track_direction = self._calculate_direction(history)

            for temp_id, det_data in unmatched_detections:
                distance = self._calculate_distance(last_data['center'], det_data['center'])
                iou = self._calculate_iou(last_data['bbox'], det_data['bbox'])

                y_diff = abs(det_data['center'][1] - last_data['center'][1])
                adjusted_max_distance = self.max_distance * (1.2 if y_diff < y_tolerance else 1.0)

                if distance < adjusted_max_distance and iou > self.iou_threshold:
                    direction_score = 0
                    if track_direction is not None:
                        current_vector = np.array(det_data['center']) - np.array(last_data['center'])
                        if np.linalg.norm(current_vector) > 0:
                            current_direction = current_vector / np.linalg.norm(current_vector)
                            direction_score = np.dot(track_direction, current_direction)

                            # 若 Y 軸變化小，放寬方向懲罰
                            if y_diff < y_tolerance:
                                direction_score = max(direction_score, 0.0)

                    score = (
                        0.5 * distance / adjusted_max_distance +
                        0.3 * (1 - iou) -
                        self.direction_weight * direction_score
                    )

                    if score < best_score:
                        best_score = score
                        best_match = (temp_id, det_data)

            if best_match:
                temp_id, det_data = best_match
                corrected_detections[track_id] = det_data
                self._update_track_history(track_id, det_data, frame_idx)
                unmatched_detections.remove(best_match)
                matched_tracks.add(track_id)

                if track_id in self.lost_tracks:
                    del self.lost_tracks[track_id]

        # 第二階段：與 lost_tracks 嘗試匹配
        for lost_id, lost_data in list(self.lost_tracks.items()):
            if lost_id in matched_tracks:
                continue

            if frame_idx - lost_data['last_seen'] > self.max_disappeared:
                del self.lost_tracks[lost_id]
                if lost_id in self.track_history:
                    del self.track_history[lost_id]
                continue

            last_data = lost_data['history'][-1]
            best_match = None
            best_score = float('inf')

            for temp_id, det_data in unmatched_detections:
                distance = self._calculate_distance(last_data['center'], det_data['center'])
                iou = self._calculate_iou(last_data['bbox'], det_data['bbox'])

                y_diff = abs(det_data['center'][1] - last_data['center'][1])
                adjusted_max_distance = self.max_distance * (1.5 if y_diff < y_tolerance else 1.0)

                if distance < adjusted_max_distance:
                    score = 0.7 * distance + 0.3 * (1 - iou)
                    if score < best_score:
                        best_score = score
                        best_match = (temp_id, det_data)

            if best_match:
                temp_id, det_data = best_match
                corrected_detections[lost_id] = det_data
                self._update_track_history(lost_id, det_data, frame_idx)
                unmatched_detections.remove(best_match)
                matched_tracks.add(lost_id)
                del self.lost_tracks[lost_id]

        # 第三階段：創建新追蹤 ID 給沒配對到的偵測
        for temp_id, det_data in unmatched_detections:
            new_track_id = self.next_track_id
            self.next_track_id += 1

            corrected_detections[new_track_id] = det_data
            self._update_track_history(new_track_id, det_data, frame_idx)

        # 更新 lost tracks（累計沒出現的次數）
        for track_id in self.track_history:
            if track_id not in matched_tracks:
                if track_id not in self.lost_tracks:
                    self.lost_tracks[track_id] = {
                        'count': 1,
                        'history': list(self.track_history[track_id]),
                        'last_seen': frame_idx
                    }
                else:
                    self.lost_tracks[track_id]['count'] += 1

        return corrected_detections

    def _calculate_direction(self, history):
        """計算軌跡的平均運動方向（單位向量）"""
        if len(history) < 2:
            return None
            
        # 取最近幾幀的運動向量
        vectors = []
        for i in range(1, min(5, len(history))):
            vec = np.array(history[i]['center']) - np.array(history[i-1]['center'])
            if np.linalg.norm(vec) > 0:
                vectors.append(vec / np.linalg.norm(vec))
        
        if not vectors:
            return None
            
        # 計算平均方向
        avg_direction = np.mean(vectors, axis=0)
        return avg_direction / np.linalg.norm(avg_direction)
    
    def _update_track_history(self, track_id, data, frame_idx):
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=self.history_size)
        self.track_history[track_id].append({
            'bbox': data['bbox'],
            'center': data['center'],
            'frame': frame_idx
        })

    def _calculate_distance(self, c1, c2):
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight

        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        if (boxAArea + boxBArea - interArea) == 0:
            return 0.0
        return interArea / float(boxAArea + boxBArea - interArea)

class BeeCounter:
    def __init__(self, roi_x1, roi_y1, roi_x2, roi_y2):
        """
        初始化蜜蜂計數系統
        
        參數:
            roi_x1, roi_y1: 感興趣區域的左上角坐標
            roi_x2, roi_y2: 感興趣區域的右下角坐標
        """
        self.roi = (roi_x1, roi_y1, roi_x2, roi_y2)
        
        # 計算ROI區域的關鍵位置
        self.roi_height = roi_y2 - roi_y1
        self.roi_width = roi_x2 - roi_x1
        self.roi_center_y = roi_y1 + self.roi_height // 2
        
        # 進入/離開判斷參數
        self.top_zone_ratio = 0.3   # ROI頂部區域占比
        self.bottom_zone_ratio = 0.3  # ROI底部區域占比
        self.min_track_frames = 3   # 追蹤最小幀數
        self.min_consistency = 0.2    # 方向一致性最小比例
        
        # 區域坐標閾值 (原本是硬編碼的值)
        self.y_lower_threshold = 648
        self.y_upper_threshold = 870
        self.x_min_threshold = 250
        self.x_max_threshold = 2250
        
        # 追蹤狀態
        self.bee_status = {}  # {id: {'positions': [], 'in_roi': bool, 'counted': bool, 'first_seen': int, 'last_seen': int}}
        self.in_count = 0
        self.out_count = 0
        self.processed_ids = set()  # 已計數的ID
        self.log = []  # 事件日誌
        self.count_animations = []  # 計數動畫

        self.debug = False
    
    def _is_in_roi(self, x, y):
        """
        判斷點(x,y)是否在ROI區域內
        """
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi
        return roi_x1 <= x <= roi_x2 and roi_y1 <= y <= roi_y2
    
    def _has_sufficient_roi_history(self, bee_id):
        """
        檢查是否有足夠的ROI歷史數據，若回傳 False，則 print 原因和 bee_id
        """
        # 情況 1: bee_id 不存在於追蹤記錄中
        if bee_id not in self.bee_status:
            print(f"[Debug] Bee {bee_id}: Not in tracking records (new bee or lost track)")
            return False
        
        # 情況 2: 追蹤的幀數不足
        if len(self.bee_status[bee_id]['positions']) < self.min_track_frames:
            print(f"[Debug] Bee {bee_id}: Insufficient tracking frames "
                f"({len(self.bee_status[bee_id]['positions'])} < {self.min_track_frames})")
            return False
        
        # 情況 3: 蜜蜂未在 ROI 中停留過
        if not self.bee_status[bee_id]['in_roi']:
            print(f"[Debug] Bee {bee_id}: Never entered ROI")
            return False
        
        # 所有條件均滿足，回傳 True
        return True
    
    def update(self, detections, frame_idx, lost_tracks=None, frame=None):
        """
        參數:
            detections: 檢測結果 {id: {'bbox': (x,y,w,h), 'center': (x,y)}}
            frame_idx: 當前幀索引
            lost_tracks: 丟失的追蹤 {id: {'count': count, 'history': [...]}}
            frame: 當前視頻幀 (用於視覺化)
        
        返回:
            in_count, out_count: 當前的進入和離開計數
        """
        current_ids = set(detections.keys())
        
        # 1. 更新當前可見的蜜蜂狀態
        for bee_id, data in detections.items():
            x, y = data['center']
            in_roi = self._is_in_roi(x, y)
            w, h = data['bbox'][2], data['bbox'][3]  # bbox的寬和高

            # 確保寬度和高度有效
            if w <= 0 or h <= 0:
                continue  # 如果寬度或高度無效，則跳過這個蜜蜂

            current_area = w * h  # 當前的面積
            top_center_y = y - h / 2  # 上緣中心y坐標
            bottom_center_y = y + h / 2  # 下緣中心y坐標
            left_x = x - w/2
            right_x = x + w/2
            if bee_id not in self.bee_status:
                # 新蜜蜂，初始化狀態
                self.bee_status[bee_id] = {
                    'positions': [],      # 位置歷史 (x, y, frame_idx, area)
                    'in_roi': in_roi,     # 是否在ROI中
                    'counted': False,     # 是否已計數
                    'first_seen': frame_idx,  # 首次出現的幀
                    'last_seen': frame_idx,   # 最後出現的幀
                    'init_area': current_area,  # 记录初始面积
                    'top_center_y': top_center_y,  # 上緣中心y坐標
                    'bottom_center_y': bottom_center_y,  # 下緣中心y坐標
                    'init_pos': (x, y),
                    'init_left_x': left_x,
                    'init_right_x': right_x,
                }
            
            # 更新狀態
            self.bee_status[bee_id]['in_roi'] = in_roi
            self.bee_status[bee_id]['last_seen'] = frame_idx

            # 添加當前位置資料：含 area
            self.bee_status[bee_id]['positions'].append((x, y, frame_idx, current_area))
            
            # 更新位置歷史
            if len(self.bee_status[bee_id]['positions']) > 0:
                last_pos = self.bee_status[bee_id]['positions'][-1]
            
            # 限制位置歷史長度
            if len(self.bee_status[bee_id]['positions']) > 30:
                self.bee_status[bee_id]['positions'].pop(0)
        
        # 2. 處理丟失的追蹤
        if lost_tracks:
            for bee_id, lost_data in lost_tracks.items():
                # 忽略當前可見的蜜蜂或已處理的ID
                if bee_id in current_ids or bee_id in self.processed_ids:
                    continue
                
                # 檢查是否有足夠的歷史數據來分析
                if bee_id in self.bee_status and 'history' in lost_data and len(lost_data['history']) >= 3:
                    history = lost_data['history']

                    # 從歷史中獲取位置數據
                    y_positions = [h['center'][1] for h in history[-200:]]  
                    x_positions = [h['center'][0] for h in history[-200:]]

                    # 計算位移
                    x_range = max(abs(x2 - x1) for x1, x2 in zip(x_positions[:-1], x_positions[1:]))

                    # 使用最後觀察到的位置
                    last_pos = history[-1]['center']
                    last_x, last_y = last_pos

                    last_bbox = history[-1]['bbox']
                    last_area = last_bbox[2] * last_bbox[3] if last_bbox[2] > 0 and last_bbox[3] > 0 else 0

                    init_x, init_y = self.bee_status[bee_id]['init_pos']
                    left_x = self.bee_status[bee_id]['init_left_x']
                    right_x = self.bee_status[bee_id]['init_right_x']

                    w, h = data['bbox'][2], data['bbox'][3]
                    current_area = w*h
                    init_area = self.bee_status[bee_id]['init_area']
                    area_ratio = current_area / max(init_area, 1)

                    top_center_y = self.bee_status[bee_id]['top_center_y']
                    bottom_center_y = self.bee_status[bee_id]['bottom_center_y']
                    # print(self.bee_status)
                    print(self._has_sufficient_roi_history(bee_id) and not self.bee_status[bee_id].get('counted', False))
               
                    print('In', 870 > last_y > 620, self.is_in_x_range(last_x), self.is_in_x_range(left_x), self.is_in_x_range(right_x), area_ratio < 0.8)
                    print('Out', 870 > init_y > 620, 870 > top_center_y > 620, 870 > bottom_center_y > 620, self.is_in_x_range(init_x) and
                            init_area < 20000 , area_ratio > 1.2)
                    print("id", bee_id, "top_center_y", top_center_y, "bottom_center_y", bottom_center_y, 
                          "last_x", last_x, "last_y", last_y, "init_x:", init_x,"init_y:", init_y, 'left_x:', left_x, 'right_x:', right_x,
                          "last_area:", last_area,"init_area:", init_area, "area_ratio:", area_ratio)

                    # 檢查是否在ROI內停留過足夠時間
                    if (self._has_sufficient_roi_history(bee_id) and 
                        not self.bee_status[bee_id].get('counted', False)):
                            
                        # 判斷進入蜂巢 (向上運動且在指定區域消失，面積縮小)
                        if (
                            870 > last_y > 620 and
                            self.is_in_x_range(last_x) and
                            self.is_in_x_range(left_x) and
                            self.is_in_x_range(right_x) 
                            # and area_ratio < 0.8
                            ):
                                    
                            self.in_count += 1
                            print(bee_id, 'in+1')
                            self.bee_status[bee_id]['counted'] = True
                            self.processed_ids.add(bee_id)
                                    
                            # 添加動畫顯示
                            if frame is not None:
                                self._add_count_animation("In +1", (int(last_x), int(last_y)))
                        
                        # 判斷離開蜂巢 (向下運動且在指定區域消失，面積擴大)
                        elif (
                            870 > init_y > 620 and
                            self.is_in_x_range(init_x) and
                            init_area < 20000 and
                            area_ratio > 1):

                            self.out_count += 1
                            print(bee_id, 'out+1')
                            self.bee_status[bee_id]['counted'] = True
                            self.processed_ids.add(bee_id)
                                    
                            # 添加動畫顯示
                            if frame is not None:
                                self._add_count_animation("Out +1", (int(last_x), int(last_y)))
                                
        # 更新動畫狀態
        self._update_animations()
        
        return self.in_count, self.out_count

    def is_in_x_range(self, x):
        valid_ranges = [
            (290, 510), (520, 740), (750, 970), (980, 1200),
            (1210, 1430), (1440, 1660), (1670, 1890), (1900, 2120), (2130, 2350)
        ]
        return any(start <= x <= end for start, end in valid_ranges)

    def set_debug(self, debug=True):
        """
        設置調試模式
        """
        self.debug = debug
    
    def _is_in_roi(self, x, y):
        """檢查點(x,y)是否在感興趣區域內"""
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi
        return roi_x1 <= x <= roi_x2 and roi_y1 <= y <= roi_y2
    
    def draw_roi(self, frame):
        """在幀上繪製ROI區域及其分區"""
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi

        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
        
        # 繪製頂部區域
        top_y = int(roi_y1 + self.roi_height * self.top_zone_ratio)
        # cv2.line(frame, (roi_x1, top_y), (roi_x2, top_y), (0, 0, 255), 2)
        # cv2.putText(frame, "Top Zone", (roi_x1 + 10, roi_y1 + 20), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 繪製底部區域
        bottom_y = int(roi_y2 - self.roi_height * self.bottom_zone_ratio)
        
        # cv2.line(frame, (roi_x1, bottom_y), (roi_x2, bottom_y), (255, 0, 0), 2)
        # cv2.putText(frame, "Bottom Zone", (roi_x1 + 10, roi_y2 - 10), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    def _add_count_animation(self, text, position):
        """添加計數動畫"""
        self.count_animations.append({
            'text': text,
            'pos': position,
            'life': 20,  # 動畫持續30幀
            'alpha': 1.0,  # 初始透明度
            # 'id': bee_id  # 關聯的蜜蜂ID
        })
    
    def _update_animations(self):
        """更新所有動畫狀態"""
        for anim in self.count_animations[:]:
            anim['life'] -= 1
            anim['alpha'] = anim['life'] / 30.0  # 逐漸淡出
            
            if anim['life'] <= 0:
                self.count_animations.remove(anim)
    
    def draw_animations(self, frame):
        """在幀上繪製所有活躍的動畫"""
        for anim in self.count_animations:
            if anim['life'] > 0:
                x, y = anim['pos']
                
                # 根據動畫類型選擇顏色
                if anim['text'] == "In +1":
                    color = (0, 255, 0)  # 綠色
                else:
                    color = (0, 0, 255)  # 紅色
                
                # 計算字體大小（隨生命週期變化）
                font_scale = 1.2 + (1.0 - anim['alpha']) * 0.5
                
                # 繪製文字（帶透明度）
                overlay = frame.copy()
                cv2.putText(overlay, anim['text'], (x, y - 20 - int(30 * (1 - anim['alpha']))), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 3)
                
                # 應用透明度
                cv2.addWeighted(overlay, anim['alpha'], frame, 1 - anim['alpha'], 0, frame)
    
    def save_log(self, filename):
        """將計數日誌保存到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"總計: 進入={self.in_count}, 離開={self.out_count}\n\n")
            for event in self.log:
                f.write(f"{event}\n")

# 主追蹤腳本
def main(video_path = None):

    def draw_tracking_centers(frame, corrected_detections):
        """
        在影像上繪製每個追蹤 ID 的中心位置

        參數:
            frame: 當前影格（OpenCV 格式的影像）
            corrected_detections: {'id': {'bbox': (x, y, w, h), 'center': (x, y)}}
        """
        for track_id, data in corrected_detections.items():
            center = data['center']
            x, y = int(center[0]), int(center[1])  # Ensure x, y are integers
            w, h = data['bbox'][2], data['bbox'][3]  # bbox的寬和高
            top_center_y = int(y - h / 2)  # 上緣中心y坐標，轉為整數
            bottom_center_y = int(y + h / 2)  # 下緣中心y坐標，轉為整數

            # 畫圓圈，確保所有坐標都是整數
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 中心位置
            cv2.circle(frame, (x, top_center_y), 5, (0, 255, 0), -1)  # 上緣中心
            cv2.circle(frame, (x, bottom_center_y), 5, (0, 255, 0), -1)  # 下緣中心

    # 載入YOLO模型
    model = YOLO(r'/home/bee/chihli_yolo8/runs/detect/train3/weights/best.pt')

    video_path = '/home/bee/chihli_yolo8/yolov8/test/schedule_20250411_132525.mkv'
    # output_video_path = '/home/bee/chihli_yolo8/yolov8/test/20250406_131916/output_clip_00m00s.mp4'
    
    # output_txt_path = '/home/bee/chihli_yolo8/yolov8/bee_counting_results.txt'
    
    # 初始化視頻捕獲
    cap = cv2.VideoCapture(video_path, cv2.CAP_ANY)
    
    # 獲取視頻屬性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 設置顯示窗口
    cv2.namedWindow('蜜蜂追蹤', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('蜜蜂追蹤', width, height)
    
    # 設置視頻寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 初始化追蹤參數
    confidence_threshold = 0.3  # 增加以獲得更穩定的檢測
    frame_index = 0

    roi_x1 = 250
    roi_x2 = 2250
    roi_y1 = 648 
    roi_y2 = 870 
    
    # 初始化ID切換防止器
    id_preventer = IDSwitchPreventer(
        max_disappeared=20,  # 允許蜜蜂更長時間的消失
        max_distance=100,     # 增加最大匹配距離
        iou_threshold=0.5,   # 降低IoU閾值，因為蜜蜂移動快
        history_size=10      # 為每個蜜蜂追蹤更多歷史
    )
    
    # 初始化蜜蜂計數器
    bee_counter = BeeCounter(roi_x1, roi_y1, roi_x2, roi_y2)
    
    # 用於顯示每個蜜蜂軌跡的字典
    bee_tracks = {}  # {id: {'positions': []}}

    paused = False

    while True:
        ret, frame = cap.read()
        if not ret:
            # 檢查是否真的到了影片末尾
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
                print("已到達影片結尾")
                break
            else:
                print("幀讀取錯誤，嘗試繼續...")
                continue  # 嘗試讀取下一幀

        time.sleep(0.1)
        
        # 追蹤幀索引
        frame_index += 1
        
        # 繪製ROI區域
        bee_counter.draw_roi(frame)
        
        # 運行YOLOv8檢測和追蹤
        results = model.track(frame, persist=True, conf=confidence_threshold, tracker="bytetrack.yaml")
        
        # 過濾結果：只保留 ROI 內的蜜蜂
        if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes_roi = []
            for box in results[0].boxes:
                # 安全檢查1：確保box不是None
                if box is None:
                    continue
                    
                # 安全檢查2：確保有xyxy屬性
                if not hasattr(box, 'xyxy'):
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                # boxes_roi.append(box)

                try:
                    # 獲取邊界框座標
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    boxes_roi.append(box)
                    
                    # 檢查是否在ROI內
                    if (roi_x1 <= center_x <= roi_x2) and (roi_y1 <= center_y <= roi_y2):
                        boxes_roi.append(box)
                except Exception as e:
                    print(f"處理檢測框座標時發生錯誤: {e}")
                    continue
            
            # 替換原始結果中的boxes
            results[0].boxes = boxes_roi

        # 處理當前檢測（只含ROI內的蜜蜂）
        current_detections = {}
        if results and hasattr(results[0], 'boxes') and results[0].boxes:
            for box in results[0].boxes:
                try:
                    # 安全獲取track_id
                    track_id = -1  # 默認值
                    if hasattr(box, 'id') and box.id is not None:
                        try:
                            track_id = int(box.id.item())
                        except:
                            track_id = -1
                    
                    # 獲取邊界框信息
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    
                    # 安全獲取confidence
                    confidence = 0.
                    if hasattr(box, 'conf') and box.conf is not None:
                        try:
                            confidence = float(box.conf.item())
                        except:
                            confidence = 0.0
                    
                    # 只處理高可信度檢測
                    if confidence >= confidence_threshold:
                        current_detections[track_id] = {
                            'bbox': (x, y, w, h),
                            'center': (x, y),
                            'confidence': confidence
                        }
                except Exception as e:
                    print(f"處理檢測框時發生錯誤: {e}")
                    continue

        # 應用ID切換防止
        corrected_detections = id_preventer.update(current_detections, frame_index)

        if frame_index % 100 == 0:
            # 清理超過100幀未見的蜜蜂
            current_ids = set(corrected_detections.keys())
            lost_ids = []
            
            for bid in bee_tracks:
                if bid not in current_ids:
                    # 從BeeCounter獲取最後出現的幀數
                    last_seen = bee_counter.bee_status.get(bid, {}).get('last_seen', 0)
                    if frame_index - last_seen > 100:
                        lost_ids.append(bid)
            
            for lost_id in lost_ids:
                if lost_id in bee_tracks:
                    del bee_tracks[lost_id]
                if lost_id in bee_counter.bee_status:
                    del bee_counter.bee_status[lost_id]

        draw_tracking_centers(frame, corrected_detections)
        
        # 更新蜜蜂計數
        in_count, out_count = bee_counter.update(corrected_detections, frame_index, id_preventer.lost_tracks, frame)

        bee_counter.draw_animations(frame)
        
        # 處理和可視化修正後的檢測
        for bee_id, detection in corrected_detections.items():
            x, y, w, h = detection['bbox']
            center_x, center_y = detection['center']
            
            # 根據蜜蜂是否在ROI中選擇顏色
            color = (0, 255, 0)  # 默認顏色（綠色）
            if bee_counter._is_in_roi(center_x, center_y):
                color = (0, 255, 255)  # 黃色，表示在ROI區域內
            
            # 繪製邊界框和ID
            cv2.rectangle(frame, 
                         (int(x - w/2), int(y - h/2)), 
                         (int(x + w/2), int(y + h/2)), 
                         color, 2)
            
            # 添加蜜蜂ID文字
            cv2.putText(frame, f"bee {bee_id}", 
                       (int(x - w/2), int(y - h/2) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            # 追蹤蜜蜂運動軌跡
            if bee_id not in bee_tracks:
                bee_tracks[bee_id] = {
                    'positions': []
                }
            
            # 添加當前位置到軌跡歷史
            bee_tracks[bee_id]['positions'].append((center_x, center_y))
            if len(bee_tracks[bee_id]['positions']) > 30:  # 保留最後30個位置
                bee_tracks[bee_id]['positions'].pop(0)
            
            # 繪製軌跡（最後10個位置）
            positions = bee_tracks[bee_id]['positions']
            if len(positions) > 2:
                for i in range(1, min(20, len(positions))):
                    cv2.line(frame, 
                            (int(positions[i-1][0]), int(positions[i-1][1])), 
                            (int(positions[i][0]), int(positions[i][1])), 
                            color, 2)
        
        # 顯示蜜蜂計數
        height, width, _ = frame.shape

        # Place text at the top-right corner
        cv2.putText(frame, f"In: {in_count}", (width - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Out: {out_count}", (width - 150, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 顯示和寫入幀
        cv2.imshow('蜜蜂追蹤', frame)
        # out.write(frame)
        
        # 按下'q'鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 清理
    cap.release()
    # out.release()
    # bee_counter.save_log(output_txt_path)
    cv2.destroyAllWindows()
    
    print(f"追蹤完成。總計: 進入={in_count}, 離開={out_count}")
    return in_count, out_count

if __name__ == "__main__":

    main()