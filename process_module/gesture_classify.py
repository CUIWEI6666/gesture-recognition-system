from utils.math import calculate_distance, calculate_angle

class GestureClassifier:
    """
    @brief 手势分类器，支持基础手势与数字手势识别
    @details 识别手势：OK、点赞、剪刀手、握拳、张开手掌，以及数字1-5
    """
    def __init__(self):
        # MediaPipe 手部关键点索引（21个点）
        self.tip_ids = [4, 8, 12, 16, 20]  # 大拇指、食指、中指、无名指、小指的指尖索引
        self.pip_ids = [3, 6, 10, 14, 18]  # 对应手指的第二关节（PIP）
        self.mcp_ids = [2, 5, 9, 13, 17]   # 对应手指的掌指关节（MCP）

    def get_finger_states(self, landmarks):
        """
        @brief 判断每根手指的伸直/弯曲状态
        @param landmarks: 手部21个关键点坐标列表
        @return list: 每根手指的状态，1=伸直，0=弯曲
        """
        fingers = []

        # 1. 大拇指：根据与食指掌指关节的距离判断是否外展
        thumb_tip = landmarks[self.tip_ids[0]]
        index_mcp = landmarks[self.mcp_ids[1]]
        thumb_abducted = calculate_distance(thumb_tip, index_mcp) > 0.1
        fingers.append(1 if thumb_abducted else 0)

        # 2. 其他四指：指尖是否高于第二关节（伸直）
        for i in range(1, 5):
            tip = landmarks[self.tip_ids[i]]
            pip = landmarks[self.pip_ids[i]]
            fingers.append(1 if tip.y < pip.y else 0)

        return fingers

    def classify(self, landmarks):
        """
        @brief 综合判断，返回识别到的手势名称
        @param landmarks: 手部21个关键点坐标列表
        @return str: 手势名称
        """
        fingers = self.get_finger_states(landmarks)

        # ========== 1. 数字手势 ==========
        # 数字1：仅食指伸直
        if fingers == [0, 1, 0, 0, 0]:
            return "数字 1"
        # 数字2：食指+中指伸直
        elif fingers == [0, 1, 1, 0, 0]:
            return "数字 2"
        # 数字3：食指+中指+无名指伸直
        elif fingers == [0, 1, 1, 1, 0]:
            return "数字 3"
        # 数字4：除大拇指外全部伸直
        elif fingers == [0, 1, 1, 1, 1]:
            return "数字 4"
        # 数字5：所有手指伸直（和张开手掌复用）
        elif fingers == [1, 1, 1, 1, 1]:
            return "数字 5 / 张开手掌"

        # ========== 2. 基础手势 ==========
        # 张开手掌：和数字5复用，已在上面判断
        # 握拳：所有手指弯曲
        elif fingers == [0, 0, 0, 0, 0]:
            return "握拳"

        # 剪刀手：和数字2复用，已在上面判断

        # 点赞：仅大拇指伸直向上，其余手指弯曲
        elif fingers[0] == 1 and fingers[1:] == [0, 0, 0, 0]:
            thumb_tip = landmarks[self.tip_ids[0]]
            thumb_mcp = landmarks[self.mcp_ids[0]]
            # 额外判断大拇指方向（向上）
            if thumb_tip.y < thumb_mcp.y:
                return "点赞"

        # OK手势：拇指+食指指尖相触，其余手指伸直
        ok_dist = calculate_distance(landmarks[self.tip_ids[0]], landmarks[self.tip_ids[1]])
        if ok_dist < 0.05 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
            return "OK 手势"

        return "未知手势"