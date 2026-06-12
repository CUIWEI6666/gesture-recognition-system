#!/usr/bin/env python3
import cv2
import sys
import argparse
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from input_module.camera_input import CameraInput
from process_module.hand_detector import HandDetector
from output_module.display import DisplayManager


class GestureRecognitionSystem:
    def __init__(self, camera_id: int = 0, debug_mode: bool = False):
        self.camera = CameraInput(camera_id=camera_id)
        self.detector = HandDetector()
        self.display = DisplayManager()
        self.debug_mode = debug_mode
        self.is_running = False
        self.show_help = False
        
        self.last_action_time = 0
        self.action_cooldown = 1.0
        
        # 手势动作映射（包括数字1-5）
        self.gesture_actions = {
            'NUMBER_1': self._action_number_1,
            'NUMBER_2': self._action_number_2,
            'NUMBER_3': self._action_number_3,
            'NUMBER_4': self._action_number_4,
            'NUMBER_5': self._action_number_5,
            'OK': self._action_ok,
            'THUMBS_UP': self._action_thumbs_up,
            'FIST': self._action_fist,
            'PEACE': self._action_peace,
            'OPEN_PALM': self._action_open_palm,
            'ROCK': self._action_rock,
            'CALL': self._action_call,
        }
    
    def _action_number_1(self):
        print("🟢 Number 1 detected!")
        
    def _action_number_2(self):
        print("🟢 Number 2 detected!")
        
    def _action_number_3(self):
        print("🟢 Number 3 detected!")
        
    def _action_number_4(self):
        print("🟢 Number 4 detected!")
        
    def _action_number_5(self):
        print("🟢 Number 5 detected!")
    
    def _action_ok(self):
        current_time = time.time()
        if current_time - self.last_action_time > self.action_cooldown:
            print("👌 OK gesture detected")
            self.last_action_time = current_time
            
    def _action_thumbs_up(self):
        current_time = time.time()
        if current_time - self.last_action_time > self.action_cooldown:
            print("👍 Thumbs up detected")
            self.last_action_time = current_time
            
    def _action_fist(self):
        current_time = time.time()
        if current_time - self.last_action_time > self.action_cooldown:
            print("✊ Fist detected")
            self.last_action_time = current_time
            
    def _action_peace(self):
        current_time = time.time()
        if current_time - self.last_action_time > self.action_cooldown:
            print("✌️ Peace sign detected")
            self.last_action_time = current_time
            
    def _action_open_palm(self):
        current_time = time.time()
        if current_time - self.last_action_time > self.action_cooldown:
            print("🖐️ Open palm detected")
            self.last_action_time = current_time
            
    def _action_rock(self):
        current_time = time.time()
        if current_time - self.last_action_time > self.action_cooldown:
            print("🤘 Rock on detected")
            self.last_action_time = current_time
    
    def _action_call(self):
        current_time = time.time()
        if current_time - self.last_action_time > self.action_cooldown:
            print("🤙 Call me gesture detected")
            self.last_action_time = current_time
    
    def _execute_gesture_action(self, gesture: str) -> None:
        """根据手势执行对应的交互动作"""
        if gesture in self.gesture_actions:
            self.gesture_actions[gesture]()
    
    def initialize(self) -> bool:
        print("=" * 50)
        print("Gesture Recognition System v3.0 (Numbers 1-5)")
        print("=" * 50)
        
        if not self.camera.open():
            return False
        print("Camera module initialized")
        
        self.detector.set_debug_mode(self.debug_mode)
        if not self.detector.initialize():
            return False
        print("Hand detector initialized")
        
        print("=" * 50)
        print("Supported gestures: 1, 2, 3, 4, 5, Fist, Thumbs Up, OK, Peace, Rock, Call")
        print("=" * 50)
        
        return True
    
    def run(self) -> None:
        if not self.initialize():
            print("System initialization failed")
            return
        
        self.is_running = True
        print("Press 'q' to quit | 'd' debug | 'h' help")
        print("=" * 50)
        
        try:
            while self.is_running:
                frame = self.camera.read_frame()
                if frame is None:
                    break
                
                frame = cv2.flip(frame, 1)
                results = self.detector.detect(frame)
                
                # 执行手势动作
                if results['success'] and results['gestures']:
                    for gesture in results['gestures']:
                        self._execute_gesture_action(gesture)
                
                # 绘制UI
                display_frame = self.display.draw_info_panel(
                    frame, self.camera.fps, results['num_hands'], self.debug_mode
                )
                display_frame = self.detector.draw_results(display_frame, results)
                
                if results['gesture_names'] and results['confidences']:
                    display_frame = self.display.draw_gesture_result(
                        display_frame, results['gesture_names'], results['confidences']
                    )
                
                if self.debug_mode and results['finger_states']:
                    for i, finger_states in enumerate(results['finger_states']):
                        display_frame = self.display.draw_finger_status(
                            display_frame, finger_states, i
                        )
                
                display_frame = self.display.draw_controls(display_frame)
                
                if self.show_help:
                    display_frame = self.display.draw_help_screen(display_frame)
                
                self.display.show(display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    self.detector.set_debug_mode(self.debug_mode)
                elif key == ord('h'):
                    self.show_help = not self.show_help
                    
        except Exception as e:
            print(f"Runtime error: {e}")
        finally:
            self.stop()
    
    def stop(self) -> None:
        self.is_running = False
        self.detector.release()
        self.camera.release()
        self.display.destroy()
        print("System shutdown")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    system = GestureRecognitionSystem(camera_id=args.camera, debug_mode=args.debug)
    system.run()


if __name__ == "__main__":
    main()
