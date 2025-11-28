# from ultralytics import YOLO
# import cv2
# import requests
#
# # ESP32 IP address
# ESP32_IP = "http://192.168.41.200/"  # Replace with your ESP32's actual IP
#
# # Initialize webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
#
# # Load YOLOv8 model
# model = YOLO("../yolo-Weights/yolov8n.pt")
#
# # COCO classes
# classNames = [
#     "person"
# ]
#
# try:
#     while True:
#         success, img = cap.read()
#         if not success:
#             break
#
#         person_detected = False
#         results = model(img, stream=True)
#
#         for r in results:
#             for box in r.boxes:
#                 cls = int(box.cls[0])
#                 if cls < len(classNames) and classNames[cls] == "person":
#                     person_detected = True
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     conf = round(float(box.conf[0]), 2)
#
#                     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
#                     cv2.putText(img, f"person {conf}", (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
#
#         if person_detected:
#             try:
#                 response = requests.get(f"{ESP32_IP}/person")
#                 print(f"Person detected → ESP32 response: {response.status_code}, {response.text}")
#             except requests.RequestException as e:
#                 print(f"Error sending request to ESP32: {e}")
#
#         cv2.imshow("Camera", img)
#         if cv2.waitKey(1) == ord('q'):
#             break
#
# finally:
#     cap.release()
#     cv2.destroyAllWindows()


from ultralytics import YOLO
import cv2
import requests
import time

# ESP32 IP address
ESP32_IP = "http://172.30.9.192/"  # Replace with your ESP32's actual IP

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load YOLOv8 model
model = YOLO("../yolo-Weights/yolov8n.pt")

# Only interested in detecting "person"
classNames = ["person"]

try:
    while True:
        success, img = cap.read()
        if not success:
            break

        person_detected = False
        results = model(img, stream=True)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls < len(classNames) and classNames[cls] == "person":
                    person_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = round(float(box.conf[0]), 2)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(img, f"person {conf}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Send to ESP32 if a person is detected
        if person_detected:
            try:
                response = requests.get(f"{ESP32_IP}/person", timeout=1)
                print(f"[{time.strftime('%H:%M:%S')}] Person detected → ESP32 response: {response.status_code}, {response.text}")
                time.sleep(0.1)  # Add 0.1 second delay
            except requests.RequestException as e:
                print(f"Error sending request to ESP32: {e}")

        cv2.imshow("Camera", img)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
