import cv2
import numpy as np
from typing import Optional
import time


class CameraInput:
    """摄像头输入模块"""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.is_opened = False
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        
    def open(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"无法打开摄像头 {self.camera_id}")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.is_opened = True
            print(f"摄像头打开成功 - 分辨率: {self.width}x{self.height}")
            return True
        except Exception as e:
            print(f"打开摄像头失败: {e}")
            return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        if not self.is_opened:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = current_time
        
        return frame
    
    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.is_opened = False
            print("摄像头资源已释放")