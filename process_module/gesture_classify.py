from typing import List, Tuple, Dict, Any
from collections import Counter
import math


class GestureClassifier:
    """手势分类器 - 支持数字手势 1-5"""

    # 基础手势
    GESTURE_OK = "OK"
    GESTURE_THUMBS_UP = "THUMBS_UP"
    GESTURE_FIST = "FIST"
    GESTURE_PEACE = "PEACE"
    GESTURE_OPEN_PALM = "OPEN_PALM"
    GESTURE_ROCK = "ROCK"
    GESTURE_CALL = "CALL"
    GESTURE_LOVE = "LOVE"

    # 数字手势 1-5
    GESTURE_NUMBER_1 = "NUMBER_1"
    GESTURE_NUMBER_2 = "NUMBER_2"
    GESTURE_NUMBER_3 = "NUMBER_3"
    GESTURE_NUMBER_4 = "NUMBER_4"
    GESTURE_NUMBER_5 = "NUMBER_5"

    GESTURE_UNKNOWN = "UNKNOWN"

    # 英文显示名称
    GESTURE_NAMES = {
        GESTURE_OK: "OK",
        GESTURE_THUMBS_UP: "Thumbs Up",
        GESTURE_FIST: "Fist",
        GESTURE_PEACE: "Peace",
        GESTURE_OPEN_PALM: "Open Palm",
        GESTURE_ROCK: "Rock",
        GESTURE_CALL: "Call",
        GESTURE_LOVE: "Love",
        GESTURE_NUMBER_1: "1",
        GESTURE_NUMBER_2: "2",
        GESTURE_NUMBER_3: "3",
        GESTURE_NUMBER_4: "4",
        GESTURE_NUMBER_5: "5",
        GESTURE_UNKNOWN: "Unknown"
    }

    def __init__(self):
        self.gesture_history = []
        self.history_size = 5

    def classify(self, finger_states: List[int],
                 landmarks: List = None,
                 handedness: str = "Right") -> str:
        if not finger_states or len(finger_states) < 5:
            return self.GESTURE_UNKNOWN

        thumb, index, middle, ring, pinky = finger_states

        # 计算伸直的手指数量
        extended_count = sum(finger_states)

        # ========== 数字手势 1-5 ==========

        # 数字 1: 只有食指伸直
        if extended_count == 1 and index == 1 and thumb == 0 and middle == 0 and ring == 0 and pinky == 0:
            return self.GESTURE_NUMBER_1

        # 数字 2: 食指和中指伸直（V字手势）
        if extended_count == 2 and index == 1 and middle == 1 and thumb == 0 and ring == 0 and pinky == 0:
            return self.GESTURE_NUMBER_2

        # 数字 3: 食指、中指、无名指伸直
        if extended_count == 3 and index == 1 and middle == 1 and ring == 1 and thumb == 0 and pinky == 0:
            return self.GESTURE_NUMBER_3

        # 数字 4: 食指、中指、无名指、小指伸直（拇指弯曲）
        if extended_count == 4 and index == 1 and middle == 1 and ring == 1 and pinky == 1 and thumb == 0:
            return self.GESTURE_NUMBER_4

        # 数字 5: 五根手指全部伸直（张开手掌）
        if extended_count == 5:
            return self.GESTURE_NUMBER_5

        # ========== 其他手势 ==========

        # 1. 握拳 - 所有手指弯曲
        if extended_count == 0:
            return self.GESTURE_FIST

        # 2. 点赞 - 只有拇指伸直
        if thumb == 1 and extended_count == 1:
            return self.GESTURE_THUMBS_UP

        # 3. 剪刀手/胜利 - 如果食指和中指伸直，且拇指弯曲（已经覆盖，但保留作为备用）
        if index == 1 and middle == 1 and ring == 0 and pinky == 0:
            if thumb == 0:
                return self.GESTURE_PEACE

        # 4. OK手势 - 拇指和食指形成圆圈
        if index == 0 and thumb == 0 and middle == 1 and ring == 1 and pinky == 1:
            if landmarks:
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                distance = math.sqrt(
                    (thumb_tip.x - index_tip.x) ** 2 +
                    (thumb_tip.y - index_tip.y) ** 2
                )
                if distance < 0.05:
                    return self.GESTURE_OK
            return self.GESTURE_OK

        # 5. 摇滚手势 - 食指和小指伸直
        if index == 1 and pinky == 1 and middle == 0 and ring == 0:
            return self.GESTURE_ROCK

        # 6. 打电话手势 - 拇指和小指伸直
        if thumb == 1 and pinky == 1 and index == 0 and middle == 0 and ring == 0:
            return self.GESTURE_CALL

        return self.GESTURE_UNKNOWN

    def _get_thumb_angle(self, landmarks) -> float:
        """计算拇指角度"""
        try:
            thumb_tip = landmarks[4]
            thumb_ip = landmarks[3]
            thumb_mcp = landmarks[2]

            v1 = (thumb_tip.x - thumb_ip.x, thumb_tip.y - thumb_ip.y)
            v2 = (thumb_mcp.x - thumb_ip.x, thumb_mcp.y - thumb_ip.y)

            dot = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

            if mag1 == 0 or mag2 == 0:
                return 0

            cos_angle = dot / (mag1 * mag2)
            cos_angle = max(-1.0, min(1.0, cos_angle))

            return math.degrees(math.acos(cos_angle))
        except:
            return 0

    def classify_with_confidence(self, finger_states: List[int],
                                 landmarks: List = None,
                                 handedness: str = "Right") -> tuple:
        gesture = self.classify(finger_states, landmarks, handedness)

        # 根据手指数量计算置信度
        extended_count = sum(finger_states)

        confidence_map = {
            self.GESTURE_FIST: 0.95 if extended_count == 0 else 0.85,
            self.GESTURE_NUMBER_1: 0.95 if extended_count == 1 else 0.85,
            self.GESTURE_NUMBER_2: 0.95 if extended_count == 2 else 0.85,
            self.GESTURE_NUMBER_3: 0.95 if extended_count == 3 else 0.85,
            self.GESTURE_NUMBER_4: 0.95 if extended_count == 4 else 0.85,
            self.GESTURE_NUMBER_5: 0.95 if extended_count == 5 else 0.85,
            self.GESTURE_PEACE: 0.92,
            self.GESTURE_THUMBS_UP: 0.90,
            self.GESTURE_OK: 0.88,
            self.GESTURE_ROCK: 0.90,
            self.GESTURE_CALL: 0.85,
        }

        confidence = confidence_map.get(gesture, 0.70)

        if gesture == self.GESTURE_UNKNOWN:
            confidence = 0.35

        return gesture, confidence

    def smooth_gesture(self, current_gesture: str) -> str:
        """平滑手势识别结果"""
        self.gesture_history.append(current_gesture)
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)

        counter = Counter(self.gesture_history)
        most_common = counter.most_common(1)[0]

        if most_common[1] >= self.history_size // 2 + 1:
            return most_common[0]
        return current_gesture

    def get_gesture_name(self, gesture_type: str) -> str:
        return self.GESTURE_NAMES.get(gesture_type, self.GESTURE_NAMES[self.GESTURE_UNKNOWN])

    def get_gesture_number(self, gesture_type: str) -> int:
        """获取手势对应的数字（如果是数字手势）"""
        number_map = {
            self.GESTURE_NUMBER_1: 1,
            self.GESTURE_NUMBER_2: 2,
            self.GESTURE_NUMBER_3: 3,
            self.GESTURE_NUMBER_4: 4,
            self.GESTURE_NUMBER_5: 5,
        }
        return number_map.get(gesture_type, 0)