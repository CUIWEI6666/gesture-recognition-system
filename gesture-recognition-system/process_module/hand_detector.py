import cv2
import numpy as np
from typing import Dict, Any, List
from .base_detector import BaseDetector
from .gesture_classify import GestureClassifier

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os


class HandDetector(BaseDetector):
    """手部检测器子类 - 兼容新版 MediaPipe"""

    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.5):
        super().__init__()
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.recognizer = None
        self.gesture_classifier = GestureClassifier()
        self.model_path = "hand_landmarker.task"

    def _download_model(self) -> bool:
        """下载手势识别模型"""
        if os.path.exists(self.model_path):
            return True

        print("Downloading hand landmark model...")
        model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

        try:
            urllib.request.urlretrieve(model_url, self.model_path)
            print("Model downloaded successfully")
            return True
        except Exception as e:
            print(f"Model download failed: {e}")
            return False

    def initialize(self) -> bool:
        try:
            if not self._download_model():
                return False

            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=self.model_path),
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_detection_confidence
            )

            self.recognizer = vision.HandLandmarker.create_from_options(options)
            self.is_initialized = True
            print("Hand detector initialized successfully")
            return True
        except Exception as e:
            print(f"Hand detector initialization failed: {e}")
            self.is_initialized = False
            return False

    def _get_finger_states(self, landmarks) -> List[int]:
        """获取手指状态"""
        fingers = []

        # 拇指 (根据x坐标)
        if landmarks[4].x < landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

        # 其他四指 (根据y坐标)
        tips = [8, 12, 16, 20]
        dips = [6, 10, 14, 18]

        for tip, dip in zip(tips, dips):
            if landmarks[tip].y < landmarks[dip].y:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def _draw_hand_landmarks(self, frame: np.ndarray, hand_landmarks, idx: int) -> np.ndarray:
        """绘制21个手部关键点和骨架连线"""
        h, w = frame.shape[:2]

        # 定义骨架连接线（标准21点连接）
        connections = [
            # 拇指
            (0, 1), (1, 2), (2, 3), (3, 4),
            # 食指
            (0, 5), (5, 6), (6, 7), (7, 8),
            # 中指
            (0, 9), (9, 10), (10, 11), (11, 12),
            # 无名指
            (0, 13), (13, 14), (14, 15), (15, 16),
            # 小指
            (0, 17), (17, 18), (18, 19), (19, 20),
            # 手掌连接（手指根部之间）
            (5, 9), (9, 13), (13, 17)
        ]

        # 绘制连接线
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                start_point = (int(hand_landmarks[start_idx].x * w),
                               int(hand_landmarks[start_idx].y * h))
                end_point = (int(hand_landmarks[end_idx].x * w),
                             int(hand_landmarks[end_idx].y * h))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        # 绘制21个关键点
        for i, landmark in enumerate(hand_landmarks):
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            if i == 0:  # 手腕
                color = (255, 255, 0)
                radius = 6
            elif i in [4, 8, 12, 16, 20]:  # 指尖
                color = (0, 0, 255)
                radius = 5
            else:  # 关节
                color = (255, 0, 0)
                radius = 3

            cv2.circle(frame, (x, y), radius, color, -1)

            if self.debug_mode:
                cv2.putText(frame, str(i), (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        if not self.is_initialized:
            return self._empty_results()

        results = {
            'success': False,
            'num_hands': 0,
            'hand_landmarks': [],
            'handedness': [],
            'gestures': [],
            'gesture_names': [],
            'confidences': [],
            'finger_states': []
        }

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection_results = self.recognizer.detect(mp_image)

            if detection_results.hand_landmarks:
                results['success'] = True
                results['num_hands'] = len(detection_results.hand_landmarks)
                results['hand_landmarks'] = detection_results.hand_landmarks

                if hasattr(detection_results, 'handedness') and detection_results.handedness:
                    results['handedness'] = detection_results.handedness

                for idx, hand_landmarks in enumerate(detection_results.hand_landmarks):
                    finger_states = self._get_finger_states(hand_landmarks)
                    results['finger_states'].append(finger_states)

                    handedness = "Right"
                    if detection_results.handedness and idx < len(detection_results.handedness):
                        handedness = detection_results.handedness[idx][0].category_name

                    gesture, confidence = self.gesture_classifier.classify_with_confidence(
                        finger_states, hand_landmarks, handedness
                    )
                    gesture = self.gesture_classifier.smooth_gesture(gesture)

                    results['gestures'].append(gesture)
                    results['gesture_names'].append(
                        self.gesture_classifier.get_gesture_name(gesture)
                    )
                    results['confidences'].append(confidence)

            return results
        except Exception as e:
            if self.debug_mode:
                print(f"Detection error: {e}")
            return self._empty_results()

    def _empty_results(self) -> Dict[str, Any]:
        return {
            'success': False,
            'num_hands': 0,
            'hand_landmarks': [],
            'handedness': [],
            'gestures': [],
            'gesture_names': [],
            'confidences': [],
            'finger_states': []
        }

    def draw_results(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        output_frame = frame.copy()
        h, w = output_frame.shape[:2]

        if not results['success']:
            cv2.putText(output_frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return output_frame

        # 绘制21个关键点和骨架连线
        if results['hand_landmarks']:
            for idx, hand_landmarks in enumerate(results['hand_landmarks']):
                output_frame = self._draw_hand_landmarks(output_frame, hand_landmarks, idx)

        # 显示手势结果
        y_offset = 30
        for i, gesture_name in enumerate(results['gesture_names']):
            confidence = results['confidences'][i] if i < len(results['confidences']) else 0
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255) if confidence > 0.5 else (0, 0, 255)

            handedness = ""
            if results['handedness'] and i < len(results['handedness']):
                handedness = results['handedness'][i][0].category_name
                handedness = f" ({handedness})"

            text = f"Hand {i + 1}{handedness}: {gesture_name} ({confidence:.0%})"
            cv2.putText(output_frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 35

        # 手指状态（调试模式）
        if self.debug_mode and results['finger_states']:
            x_start = w - 150
            y_start = 30
            cv2.putText(output_frame, "Finger States:", (x_start, y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            for i, finger_states in enumerate(results['finger_states']):
                for j, (name, state) in enumerate(zip(finger_names, finger_states)):
                    color = (0, 255, 0) if state == 1 else (0, 0, 255)
                    status = "UP" if state == 1 else "DOWN"
                    cv2.putText(output_frame, f"{name}: {status}",
                                (x_start, y_start + 25 + i * 80 + j * 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 显示手部数量
        hand_text = f"Hands detected: {results['num_hands']} (21 landmarks)"
        cv2.putText(output_frame, hand_text, (10, output_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return output_frame

    def release(self) -> None:
        if self.recognizer:
            self.recognizer.close()
            self.is_initialized = False
            print("Hand detector resources released")