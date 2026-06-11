from utils.math import calculate_distance, calculate_angle

class GestureClassifier:
    """
    @brief 手势分类器（基于MediaPipe 全部21个手部关键点）
    @details 支持：数字1/2/3/4/5、OK、点赞、剪刀手、握拳、张开手掌
    """
    def __init__(self):
        # ========== 完整21个关键点分组（全覆盖） ==========
        self.wrist = 0                  # 手腕 基点

        # 大拇指 4个点：根关节、中关节、末关节、指尖
        self.thumb = [1, 2, 3, 4]
        # 食指 4个点
        self.index = [5, 6, 7, 8]
        # 中指 4个点
        self.middle = [9, 10, 11, 12]
        # 无名指 4个点
        self.ring = [13, 14, 15, 16]
        # 小指 4个点
        self.pinky = [17, 18, 19, 20]

        # 把五根手指统一存入列表，方便遍历：拇指、食指、中指、无名指、小指
        self.fingers = [self.thumb, self.index, self.middle, self.ring, self.pinky]

    def get_single_finger_state(self, landmarks, finger_points):
        """
        @brief 单根手指状态判断（使用该手指全部4个关键点）
        @param landmarks: 整只手21个关键点
        @param finger_points: 当前手指的4个点位 [根,中,末,尖]
        @return int: 1=伸直  0=弯曲
        """
        root = landmarks[finger_points[0]]
        mid = landmarks[finger_points[1]]
        end = landmarks[finger_points[2]]
        tip = landmarks[finger_points[3]]

        # 利用 根-中-末 三点计算夹角（使用手指全部关节点）
        angle = calculate_angle(root, mid, end)
        # 伸直：夹角接近180°；弯曲：夹角明显变小
        if angle > 120:
            return 1
        else:
            return 0

    def get_all_fingers_state(self, landmarks):
        """
        @brief 遍历5根手指，得到全部手指状态（用到完整21个关键点）
        @return list: [拇指,食指,中指,无名指,小指]  1伸直 / 0弯曲
        """
        state_list = []
        # 遍历5根手指，每根都用自身4个关键点判断
        for finger in self.fingers:
            s = self.get_single_finger_state(landmarks, finger)
            state_list.append(s)
        return state_list

    def classify(self, landmarks):
        """
        @brief 核心手势识别（基于21个关键点综合判断）
        @param landmarks: 单只手完整21个关键点坐标
        @return str: 识别结果
        """
        # 1. 获取五根手指整体状态（已使用全部21点）
        f_state = self.get_all_fingers_state(landmarks)

        # 2. 提取常用单点（后续判断复用）
        wrist_pt = landmarks[self.wrist]
        thumb_tip = landmarks[self.thumb[3]]
        index_tip = landmarks[self.index[3]]

        # ==================== 数字手势 1~5 ====================
        # 数字1：仅食指伸直
        if f_state == [0, 1, 0, 0, 0]:
            return "数字 1"
        # 数字2：食指、中指伸直（剪刀手）
        elif f_state == [0, 1, 1, 0, 0]:
            return "数字 2 / 剪刀手"
        # 数字3：食指、中指、无名指伸直
        elif f_state == [0, 1, 1, 1, 0]:
            return "数字 3"
        # 数字4：食指、中指、无名指、小指伸直
        elif f_state == [0, 1, 1, 1, 1]:
            return "数字 4"
        # 数字5：全部手指伸直（张开手掌）
        elif f_state == [1, 1, 1, 1, 1]:
            return "数字 5 / 张开手掌"

        # ==================== 常规手势 ====================
        # 握拳：所有手指全部弯曲
        elif f_state == [0, 0, 0, 0, 0]:
            return "握拳"

        # 点赞：仅大拇指伸直，其余四指弯曲，且大拇指向上抬起
        elif f_state[0] == 1 and f_state[1:] == [0, 0, 0, 0]:
            # 结合手腕+拇指关键点判断方向（用到手腕+拇指全部点位）
            if thumb_tip.y < wrist_pt.y:
                return "点赞"

        # OK手势：拇指指尖 & 食指指尖相触，其余手指伸直
        tip_dist = calculate_distance(thumb_tip, index_tip)
        if tip_dist < 0.06 and f_state[2] == 1 and f_state[3] == 1 and f_state[4] == 1:
            return "OK 手势"

        # 无法识别
        return "未知手势"