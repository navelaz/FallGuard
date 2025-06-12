import robomaster
from robomaster import robot
import time
import cv2
import numpy as np
from ultralytics import YOLO
import os

class TelloYOLODetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        print(f"🤖 Modelo YOLO cargado: {model_path}")
        print(f"📊 Clases detectables: {list(self.model.names.values())}")
        
        self.tl_drone = robot.Drone()
        self.tl_drone.initialize()
        self.tl_flight = self.tl_drone.flight
        self.tl_camera = self.tl_drone.camera
        self.tl_battery = self.tl_drone.battery
        
        self.Kp_yaw, self.Kd_yaw = 0.4, 0.2
        self.Kp_z, self.Kd_z = 0.3, 0.1
        self.prev_error_yaw, self.prev_error_z = 0, 0
        
        self.fall_detected = False
        self.fall_detection_count = 0
        self.fall_threshold = 3
        
        self.Kp_forward = 0.02
        self.target_box_area = 50000
        self.fall_confidence_threshold = 0.70
        
        self.inspection_mode = True
        self.inspection_yaw_speed = 15
        self.inspection_step_duration = 2.0
        self.inspection_pause_duration = 1.0
        self.last_inspection_time = time.time()
        self.inspection_direction = 1
        self.inspection_step_count = 0
        self.max_inspection_steps = 8
        self.is_inspecting = False
        self.inspection_start_time = 0
        
    def start_camera(self):
        print(f"🔋 Batería del dron: {self.tl_battery.get_battery()}%")
        self.tl_camera.start_video_stream(display=False)
        print("📹 Cámara iniciada")
        
    def detect_objects(self, frame):
        results = self.model(frame, conf=self.confidence_threshold)
        return results[0]
    
    def draw_detections(self, frame, results):
        annotated_frame = results.plot()
        y_offset = 30
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.model.names[class_id]
            cv2.putText(annotated_frame, f"{class_name}: {confidence:.2f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        return annotated_frame
    
    def inspection_pattern(self):
        current_time = time.time()
        if not self.is_inspecting:
            self.is_inspecting = True
            self.inspection_start_time = current_time
            yaw_command = self.inspection_yaw_speed * self.inspection_direction
            self.tl_flight.rc(0, 0, 0, yaw_command)
        elif current_time - self.inspection_start_time >= self.inspection_step_duration:
            self.tl_flight.rc(0, 0, 0, 0)
            self.is_inspecting = False
            self.inspection_step_count += 1
            self.last_inspection_time = current_time
            if self.inspection_step_count >= self.max_inspection_steps:
                self.inspection_direction *= -1
                self.inspection_step_count = 0
    
    def should_inspect(self):
        current_time = time.time()
        if self.is_inspecting:
            return True
        return current_time - self.last_inspection_time >= self.inspection_pause_duration
    
    def stop_inspection(self):
        if self.is_inspecting or self.inspection_mode:
            self.tl_flight.rc(0, 0, 0, 0)
            self.is_inspecting = False
            self.inspection_mode = False
            self.last_inspection_time = time.time()
    
    def control_drone_to_target(self, target_x, target_y, frame_shape, box_area=None):
        frame_h, frame_w = frame_shape[:2]
        cx, cy = frame_w // 2, frame_h // 2
        error_yaw = target_x - cx
        derivative_yaw = error_yaw - self.prev_error_yaw
        yaw_speed = int(self.Kp_yaw * error_yaw + self.Kd_yaw * derivative_yaw)
        yaw_speed = np.clip(yaw_speed, -90, 90)
        
        error_z = cy - target_y
        derivative_z = error_z - self.prev_error_z
        ud = int(self.Kp_z * error_z + self.Kd_z * derivative_z)
        ud = np.clip(ud, -20, 20)
        
        forward_speed = 0
        if box_area is not None:
            if box_area < 24000:
                forward_speed = 25
            elif box_area > 60000:
                forward_speed = -10
        self.tl_flight.rc(0, forward_speed, ud, yaw_speed)
        self.prev_error_yaw, self.prev_error_z = error_yaw, error_z
    
    def handle_fall_detection(self, results, frame):
        fall_detected_this_frame = False
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            if 'fall' in class_name.lower() and confidence > self.fall_confidence_threshold:
                fall_detected_this_frame = True
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                area = (x2 - x1) * (y2 - y1)
                self.control_drone_to_target(cx, cy, frame.shape, area)
                print(f"⚠️  CAÍDA DETECTADA ({confidence:.2f})")
                break
        
        if fall_detected_this_frame:
            self.fall_detection_count += 1
            if self.fall_detection_count >= self.fall_threshold and not self.fall_detected:
                self.fall_detected = True
                self.trigger_fall_alert()
        else:
            self.fall_detection_count = max(0, self.fall_detection_count - 1)
            if self.fall_detection_count == 0:
                self.fall_detected = False
        return fall_detected_this_frame
    
    def trigger_fall_alert(self):
        print("🚨 ALERTA DE CAÍDA CONFIRMADA 🚨")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.screenshot_path = f"fall_detected_{timestamp}.jpg"
    
    def emergency_landing(self):
        try:
            self.tl_flight.rc(0, 0, 0, 0)
            time.sleep(0.5)
            self.tl_flight.land().wait_for_completed(timeout=10)
        except:
            try:
                self.tl_flight.stop()
            except:
                pass
    
    def run_detection(self, enable_takeoff=False):
        self.start_camera()
        if enable_takeoff:
            print("🚁 Despegando...")
            self.tl_flight.takeoff().wait_for_completed()
            time.sleep(2)
        
        try:
            while True:
                frame = self.tl_camera.read_cv2_image(strategy="newest", timeout=5)
                if frame is None:
                    continue
                
                frame = cv2.resize(frame, (640, 480))
                results = self.detect_objects(frame)
                annotated_frame = self.draw_detections(frame, results)
                fall_detected = self.handle_fall_detection(results, frame)
                
                if len(results.boxes) == 0 and not fall_detected:
                    if self.should_inspect():
                        self.inspection_pattern()
                else:
                    if self.is_inspecting:
                        self.stop_inspection()
                    self.inspection_mode = False
                
                cv2.imshow("Detección Tello", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"tello_{timestamp}.jpg", annotated_frame)
                elif key == ord('e') or key == 27:
                    self.emergency_landing()
                    break
                elif key == ord('i'):
                    self.inspection_mode = not self.inspection_mode
                    if not self.inspection_mode:
                        self.stop_inspection()
        except KeyboardInterrupt:
            print("🛑 Interrumpido por usuario")
        finally:
            self.cleanup(enable_takeoff)
    
    def cleanup(self, was_flying=False):
        if was_flying:
            self.tl_flight.land().wait_for_completed()
        self.tl_camera.stop_video_stream()
        cv2.destroyAllWindows()
        self.tl_drone.close()
        print("✅ Fin del programa")

if __name__ == '__main__':
    MODEL_PATH = r'C:\Users\Naho\OneDrive\Desktop\fallDetection.pt'
    CONFIDENCE_THRESHOLD = 0.5
    ENABLE_TAKEOFF = True

    if not os.path.exists(MODEL_PATH):
        print("❌ Modelo no encontrado")
        exit()

    try:
        for i in range(5, 0, -1):
            print(f"Despegue en {i}...")
            time.sleep(1)
        detector = TelloYOLODetector(MODEL_PATH, CONFIDENCE_THRESHOLD)
        detector.run_detection(enable_takeoff=ENABLE_TAKEOFF)
    except Exception as e:
        print(f"❌ Error: {e}")
