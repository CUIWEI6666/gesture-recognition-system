import cv2
import numpy as np
from typing import List
from PIL import Image, ImageDraw, ImageFont


class DisplayManager:
    """【答辩级终极美化版】手势识别界面"""

    def __init__(self, window_name: str = "Gesture Recognition System"):
        self.window_name = window_name
        # 加载字体（Windows自带，无需额外安装）
        self.font_normal = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
        self.font_large = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 120)  # 粗体大数字
        self.font_small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 14)

        # 高级配色（科技感蓝绿主题）
        self.colors = {
            'bg_card': (45, 45, 55),  # 卡片深灰背景
            'border_blue': (255, 180, 0),  # 蓝色渐变边框
            'accent_green': (100, 255, 120),  # 成功绿
            'accent_orange': (0, 170, 255),  # 警告橙
            'accent_red': (80, 80, 255),  # 错误红
            'accent_purple': (255, 80, 255),  # 数字紫
            'text_white': (255, 255, 255),  # 白色文字
            'shadow': (0, 0, 0)  # 阴影
        }

        # 手势图标（无问号问题）
        self.gesture_names = {
            "1": "数字 1", "2": "数字 2", "3": "数字 3", "4": "数字 4", "5": "数字 5",
            "Fist": "握拳", "Thumbs Up": "点赞", "OK": "OK确认", "Peace": "剪刀手",
            "Open Palm": "张开手掌", "Rock": "摇滚", "Call": "打电话", "Unknown": "未识别"
        }

        self.toast_message = ""
        self.toast_timer = 0
        self.gesture_anim_timer = 0  # 手势触发动画计时器

    def _blur_background(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                         blur_radius: int = 15) -> np.ndarray:
        """【真实毛玻璃】高斯模糊背景，iOS风格"""
        # 截取区域做高斯模糊
        roi = img[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, (blur_radius, blur_radius), 0)
        # 叠加半透明深色遮罩
        overlay = np.full(blurred.shape, self.colors['bg_card'], dtype=np.uint8)
        img[y1:y2, x1:x2] = cv2.addWeighted(blurred, 0.3, overlay, 0.7, 0)
        # 加1px圆角边框
        cv2.rectangle(img, (x1, y1), (x2, y2), self.colors['border_blue'], 1, cv2.LINE_AA)
        return img

    def _put_text_with_shadow(self, img: np.ndarray, text: str, pos: tuple, font, color: tuple,
                              offset: int = 2) -> np.ndarray:
        """文字加阴影，立体感拉满"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        # 先画阴影
        draw.text((pos[0] + offset, pos[1] + offset), text, font=font, fill=self.colors['shadow'])
        # 再画文字
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def show_toast(self, message: str, duration: int = 45) -> None:
        self.toast_message = message
        self.toast_timer = duration
        self.gesture_anim_timer = duration

    def draw_info_panel(self, frame: np.ndarray, fps: float,
                        num_hands: int, debug_mode: bool) -> np.ndarray:
        """【优化】左侧信息栏（毛玻璃卡片）"""
        output = frame.copy()
        h, w = output.shape[:2]

        # 毛玻璃卡片
        output = self._blur_background(output, 15, 15, 240, 120)

        # FPS显示（带颜色分级）
        fps_color = self.colors['accent_green'] if fps >= 25 else self.colors['accent_orange'] if fps >= 15 else \
        self.colors['accent_red']
        output = self._put_text_with_shadow(output, f"⚡ 帧率: {fps:.0f} FPS", (30, 25), self.font_normal, fps_color)

        # 手数量
        output = self._put_text_with_shadow(output, f"✋ 检测到手: {num_hands} 只", (30, 60), self.font_normal,
                                            self.colors['text_white'])

        # 调试状态
        if debug_mode:
            output = self._put_text_with_shadow(output, "🔧 调试模式: 开启", (30, 95), self.font_small,
                                                self.colors['accent_orange'])

        return output

    def draw_gesture_result(self, frame: np.ndarray, gestures: List[str],
                            confidences: List[float]) -> np.ndarray:
        """【优化】右侧手势结果卡片"""
        output = frame.copy()
        h, w = output.shape[:2]

        for i, (gesture, conf) in enumerate(zip(gestures, confidences)):
            # 置信度颜色
            if conf > 0.75:
                color = self.colors['accent_green']
                level = "高置信度"
            elif conf > 0.5:
                color = self.colors['accent_orange']
                level = "中置信度"
            else:
                color = self.colors['accent_red']
                level = "低置信度"

            # 【核心】数字手势居中特效（外发光+阴影+渐变）
            if gesture in ["1", "2", "3", "4", "5"]:
                # 动画效果：手势刚触发时放大
                scale = 1.1 if self.gesture_anim_timer > 30 else 1.0
                font_size = int(120 * scale)
                font_large = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", font_size)

                text_size = font_large.getbbox(gesture)
                text_x = (w - (text_size[2] - text_size[0])) // 2
                text_y = (h - (text_size[3] - text_size[1])) // 2

                # 3层外发光+阴影+主体文字
                output = self._put_text_with_shadow(output, gesture, (text_x, text_y), font_large,
                                                    self.colors['accent_purple'], offset=4)
                output = self._put_text_with_shadow(output, gesture, (text_x, text_y), font_large,
                                                    self.colors['accent_purple'], offset=2)
                output = self._put_text_with_shadow(output, gesture, (text_x, text_y), font_large,
                                                    self.colors['text_white'])

            # 右侧手势卡片
            card_x, card_y = w - 270, 15 + i * 95
            output = self._blur_background(output, card_x, card_y, w - 15, card_y + 85)

            # 手势名称
            gesture_name = self.gesture_names.get(gesture, "未识别")
            output = self._put_text_with_shadow(output, gesture_name, (card_x + 20, card_y + 25), self.font_normal,
                                                color)

            # 渐变置信度进度条
            bar_w = int(200 * conf)
            cv2.rectangle(output, (card_x + 20, card_y + 55), (card_x + 220, card_y + 68), (30, 30, 40), -1,
                          cv2.LINE_AA)
            # 渐变进度条
            for x in range(bar_w):
                ratio = x / 200
                bar_color = (
                    int(self.colors['border_blue'][0] * (1 - ratio) + self.colors['accent_green'][0] * ratio),
                    int(self.colors['border_blue'][1] * (1 - ratio) + self.colors['accent_green'][1] * ratio),
                    int(self.colors['border_blue'][2] * (1 - ratio) + self.colors['accent_green'][2] * ratio)
                )
                cv2.line(output, (card_x + 20 + x, card_y + 55), (card_x + 20 + x, card_y + 68), bar_color, 1)

            output = self._put_text_with_shadow(output, f"{conf:.0%} · {level}", (card_x + 20, card_y + 72),
                                                self.font_small, self.colors['text_white'])

        if self.gesture_anim_timer > 0:
            self.gesture_anim_timer -= 1
        return output

    def draw_controls(self, frame: np.ndarray) -> np.ndarray:
        """【优化】底部状态栏"""
        output = frame.copy()
        h, w = output.shape[:2]

        # 底部毛玻璃状态栏
        output = self._blur_background(output, 15, h - 50, w - 15, h - 15)
        output = self._put_text_with_shadow(output, "⌨️  Q退出  |  D调试  |  H帮助", (w // 2 - 130, h - 42),
                                            self.font_small, self.colors['text_white'])

        # Toast触发提示
        if self.toast_timer > 0:
            self.toast_timer -= 1
            text_size = self.font_normal.getbbox(self.toast_message)
            text_w = text_size[2] - text_size[0]
            toast_x = (w - text_w) // 2
            # 绿色毛玻璃提示框
            output = self._blur_background(output, toast_x - 25, h // 2 - 35, toast_x + text_w + 25, h // 2 + 25)
            output = self._put_text_with_shadow(output, self.toast_message, (toast_x, h // 2 - 25), self.font_normal,
                                                self.colors['accent_green'])

        return output

    def draw_help_screen(self, frame: np.ndarray) -> np.ndarray:
        """【优化】全屏帮助界面"""
        output = frame.copy()
        h, w = output.shape[:2]

        # 全屏高斯模糊遮罩
        output = cv2.GaussianBlur(output, (31, 31), 0)
        overlay = np.full(output.shape, (20, 20, 30), dtype=np.uint8)
        output = cv2.addWeighted(output, 0.2, overlay, 0.8, 0)

        help_texts = [
            "🎯 手势识别系统 操作说明",
            "──────────────────────────────",
            "",
            "📌 数字手势:",
            "  1 → 仅食指伸直",
            "  2 → 食指+中指伸直（V字）",
            "  3 → 食指+中指+无名指伸直",
            "  4 → 四指伸直，拇指弯曲",
            "  5 → 五指全部伸直",
            "",
            "✋ 常用手势:",
            "  握拳 → 所有手指弯曲",
            "  点赞 → 仅拇指向上伸直",
            "  OK → 拇指+食指指尖相触",
            "  摇滚 → 食指+小指伸直",
            "  打电话 → 拇指+小指伸直",
            "",
            "⌨️ 快捷键:",
            "  Q → 退出程序",
            "  D → 开启/关闭调试模式",
            "  H → 显示/关闭此帮助",
            "",
            "──────────────────────────────",
            "按 H 关闭帮助界面"
        ]

        for i, text in enumerate(help_texts):
            output = self._put_text_with_shadow(output, text, (w // 2 - 170, 70 + i * 28), self.font_small,
                                                self.colors['text_white'])

        return output

    def draw_finger_status(self, frame: np.ndarray, finger_states: List[int],
                           hand_index: int = 0) -> np.ndarray:
        """调试用手指状态（默认隐藏）"""
        output = frame.copy()
        if not finger_states or len(finger_states) < 5:
            return output

        finger_names = ['拇指', '食指', '中指', '无名指', '小指']
        x_start = 30
        y_start = 140 + hand_index * 130

        output = self._blur_background(output, x_start - 10, y_start - 10, x_start + 180, y_start + 130)

        for i, (name, state) in enumerate(zip(finger_names, finger_states)):
            color = self.colors['accent_green'] if state == 1 else self.colors['accent_red']
            status = "✅ 伸直" if state == 1 else "❌ 弯曲"
            output = self._put_text_with_shadow(output, f"{name}: {status}", (x_start, y_start + i * 22),
                                                self.font_small, color)

        return output

    def show(self, frame: np.ndarray) -> None:
        cv2.imshow(self.window_name, frame)

    def destroy(self) -> None:
        cv2.destroyWindow(self.window_name)
