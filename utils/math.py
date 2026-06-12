import math
from typing import List, Tuple


def calculate_distance(point1: Tuple[float, float],
                       point2: Tuple[float, float]) -> float:
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_angle(point1: Tuple[float, float],
                    point2: Tuple[float, float],
                    point3: Tuple[float, float]) -> float:
    v1 = (point1[0] - point2[0], point1[1] - point2[1])
    v2 = (point3[0] - point2[0], point3[1] - point2[1])

    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    if mag1 == 0 or mag2 == 0:
        return 0

    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def get_finger_states(landmarks) -> List[int]:
    """获取5个手指的伸直状态"""
    fingers = []

    # 拇指
    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # 其他四指
    tips = [8, 12, 16, 20]
    dips = [6, 10, 14, 18]

    for i in range(4):
        if landmarks[tips[i]].y < landmarks[dips[i]].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def normalize_coordinates(landmarks, width: int, height: int) -> List[Tuple[float, float]]:
    pixel_coords = []
    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        pixel_coords.append((x, y))
    return pixel_coords