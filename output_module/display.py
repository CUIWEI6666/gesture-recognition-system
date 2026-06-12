import cv2
import numpy as np
from typing import List


class DisplayManager:
    """显示管理器 - 负责画面绘制、文字渲染、UI美化"""
    
    def __init__(self, window_name: str = "Gesture Recognition System"):
        self.window_name = window_name
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_large = cv2.FONT_HERSHEY_DUPLEX
        
        # 颜色定义 (BGR格式)
        self.colors = {
            'primary': (0, 255, 0),      # Green
            'secondary': (255, 165, 0),  # Orange
            'warning': (0, 0, 255),      # Red
            'info': (255, 255, 0),       # Cyan
            'white': (255, 255, 255),    # White
            'black': (0, 0, 0),          # Black
            'purple': (255, 0, 255),     # Purple
            'blue': (255, 0, 0),         # Blue
            'background': (50, 50, 50)   # Gray
        }
    
    def draw_info_panel(self, frame: np.ndarray, fps: float, 
                        num_hands: int, debug_mode: bool) -> np.ndarray:
        """绘制信息面板"""
        output = frame.copy()
        
        # FPS
        cv2.putText(output, f"FPS: {fps:.1f}", (10, 30), self.font, 0.6, self.colors['info'], 2)
        
        # Hand count
        cv2.putText(output, f"Hands: {num_hands}", (10, 60), self.font, 0.6, self.colors['primary'], 2)
        
        # Debug mode
        debug_color = self.colors['warning'] if debug_mode else self.colors['secondary']
        cv2.putText(output, f"Debug: {'ON' if debug_mode else 'OFF'}", 
                   (10, 90), self.font, 0.5, debug_color, 1)
        
        return output
    
    def draw_gesture_result(self, frame: np.ndarray, gestures: List[str],
                           confidences: List[float]) -> np.ndarray:
        """绘制手势识别结果"""
        output = frame.copy()
        
        for i, (gesture, conf) in enumerate(zip(gestures, confidences)):
            if conf > 0.7:
                color = self.colors['primary']
            elif conf > 0.5:
                color = self.colors['secondary']
            else:
                color = self.colors['warning']
            
            # 数字手势用大字体显示
            if gesture in ["1", "2", "3", "4", "5"]:
                # 在屏幕中央显示大数字
                h, w = output.shape[:2]
                text_size = cv2.getTextSize(gesture, self.font_large, 3, 3)[0]
                text_x = (w - text_size[0]) // 2
                text_y = (h - text_size[1]) // 2
                cv2.putText(output, gesture, (text_x, text_y), 
                           self.font_large, 3, self.colors['purple'], 3)
            
            # 在面板显示
            text = f"Gesture {i+1}: {gesture} ({conf:.0%})"
            cv2.putText(output, text, (10, 130 + i * 35), self.font, 0.6, color, 2)
        
        return output
    
    def draw_controls(self, frame: np.ndarray) -> np.ndarray:
        """绘制控制说明"""
        output = frame.copy()
        h = output.shape[0]
        
        cv2.putText(output, "Press 'q' to quit | 'd' debug | 'h' help", 
                   (10, h - 10), self.font, 0.5, self.colors['white'], 1)
        
        return output
    
    def draw_help_screen(self, frame: np.ndarray) -> np.ndarray:
        """绘制帮助屏幕"""
        output = frame.copy()
        h, w = output.shape[:2]
        
        # Semi-transparent overlay
        overlay = output.copy()
        cv2.rectangle(overlay, (50, 50), (w - 50, h - 50), self.colors['black'], -1)
        output = cv2.addWeighted(overlay, 0.7, output, 0.3, 0)
        
        help_texts = [
            "=== GESTURE RECOGNITION HELP ===",
            "",
            "Number Gestures (1-5):",
            "  1 - Only index finger",
            "  2 - Index + middle fingers (V sign)",
            "  3 - Index + middle + ring fingers",
            "  4 - Four fingers (thumb bent)",
            "  5 - Open palm (all fingers)",
            "",
            "Other Gestures:",
            "  Fist        - Make a fist (0)",
            "  Thumbs Up   - Thumbs up gesture",
            "  OK          - OK sign",
            "  Rock        - Rock on sign",
            "  Call        - Call me gesture",
            "",
            "Controls:",
            "  Q - Quit program",
            "  D - Toggle debug mode",
            "  H - Show/hide help",
            "",
            "Press H to close this help"
        ]
        
        for i, text in enumerate(help_texts):
            cv2.putText(output, text, (70, 100 + i * 30), 
                       self.font, 0.55, self.colors['white'], 1)
        
        return output
    
    def draw_finger_status(self, frame: np.ndarray, finger_states: List[int],
                          hand_index: int = 0) -> np.ndarray:
        """绘制手指状态（调试模式）"""
        output = frame.copy()
        
        if not finger_states or len(finger_states) < 5:
            return output
        
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        x_start = frame.shape[1] - 140
        y_start = 30 + hand_index * 100
        
        for i, (name, state) in enumerate(zip(finger_names, finger_states)):
            color = self.colors['primary'] if state == 1 else self.colors['warning']
            status = "UP" if state == 1 else "DOWN"
            cv2.putText(output, f"{name}: {status}", (x_start, y_start + i * 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 显示伸直手指数量
        extended_count = sum(finger_states)
        cv2.putText(output, f"Count: {extended_count}", (x_start, y_start + 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return output
    
    def show(self, frame: np.ndarray) -> None:
        """显示图像"""
        cv2.imshow(self.window_name, frame)
    
    def destroy(self) -> None:
        """销毁窗口"""
        cv2.destroyWindow(self.window_name)
