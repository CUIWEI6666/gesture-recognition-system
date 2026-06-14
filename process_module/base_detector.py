from abc import ABC, abstractmethod
import cv2
import numpy as np
from typing import Dict, Any


class BaseDetector(ABC):
    """抽象基类，定义检测器的通用接口 / Определение общего интерфейса для детектора"""

    def __init__(self):
        self.is_initialized = False
        self.debug_mode = False

    @abstractmethod
    def initialize(self) -> bool:
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        pass

    @abstractmethod
    def draw_results(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        pass

    def set_debug_mode(self, enabled: bool) -> None:
        self.debug_mode = enabled
        print(f"调试模式/Режим отладки: {'开启/включить' if enabled else '关闭/Закрыть'}")
