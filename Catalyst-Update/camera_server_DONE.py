# # camera_server.py - Optimized for low latency
# import cv2
# import time  # Added missing import
# from fastapi import FastAPI, WebSocket
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse
# import uvicorn
# from ultralytics import YOLO
# import asyncio
# import requests
#
# # Camera configuration
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduced resolution
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
# cap.set(cv2.CAP_PROP_FPS, 15)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
#
# # Lightweight YOLO model configuration
# model = YOLO("yolov8n.pt").half()  # FP16 for faster inference
# model.fuse()  # Fuse layers for faster inference
# model.conf = 0.5  # Higher confidence threshold
#
# # ESP32 configuration
# ESP32_IP = "http://192.168.1.16/"
# classNames = ["person"]
#
# app = FastAPI()
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
#
# async def notify_esp32():
#     try:
#         response = requests.get(f"{ESP32_IP}/person", timeout=1)
#         print(f"ESP32 notified - Status: {response.status_code}")
#     except Exception as e:
#         print(f"ESP32 notification failed: {str(e)}")
#
#
# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         # Clear camera buffer
#         for _ in range(5):
#             cap.read()
#
#         while True:
#             start_time = time.time()
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             # Process detection every 2nd frame
#             person_detected = False
#             if int(time.time() * 2) % 2 == 0:
#                 results = model(frame, half=True, verbose=False, imgsz=320)
#                 for r in results:
#                     for box in r.boxes:
#                         cls = int(box.cls[0])
#                         if cls < len(classNames) and classNames[cls] == "person":
#                             person_detected = True
#                             x1, y1, x2, y2 = map(int, box.xyxy[0])
#                             conf = round(float(box.conf[0]), 2)
#                             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
#                             cv2.putText(frame, f"P {conf}", (x1, y1 - 10),
#                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#
#                 if person_detected:
#                     asyncio.create_task(notify_esp32())
#
#             # Optimized JPEG compression
#             _, buffer = cv2.imencode('.jpg', frame, [
#                 cv2.IMWRITE_JPEG_QUALITY, 50,
#                 cv2.IMWRITE_JPEG_OPTIMIZE, 1
#             ])
#
#             await websocket.send_bytes(buffer.tobytes())
#
#             # Dynamic sleep for consistent FPS
#             processing_time = time.time() - start_time
#             await asyncio.sleep(max(0, 0.066 - processing_time))  # ~15 FPS
#
#     except Exception as e:
#         print(f"WebSocket Error: {str(e)}")
#     finally:
#         cap.release()
#         await websocket.close()
#
# def generate_frames_with_detection():
#     while True:
#         success, frame = cap.read()
#         if not success:
#             continue
#
#         person_detected = False
#         results = model(frame, half=True, verbose=False, imgsz=320)
#         for r in results:
#             for box in r.boxes:
#                 cls = int(box.cls[0])
#                 if cls < len(classNames) and classNames[cls] == "person":
#                     person_detected = True
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     conf = round(float(box.conf[0]), 2)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
#                     cv2.putText(frame, f"P {conf}", (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#
#         if person_detected:
#             # Notify ESP32 (synchronously here)
#             try:
#                 response = requests.get(f"{ESP32_IP}/person", timeout=1)
#                 print(f"[VIDEO_FEED] ESP32 notified - Status: {response.status_code}")
#             except Exception as e:
#                 print(f"[VIDEO_FEED] ESP32 notification failed: {str(e)}")
#
#         # Encode frame
#         ret, buffer = cv2.imencode('.jpg', frame, [
#             cv2.IMWRITE_JPEG_QUALITY, 50,
#             cv2.IMWRITE_JPEG_OPTIMIZE, 1
#         ])
#         if not ret:
#             continue
#
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#
#
# @app.get("/video_feed")
# async def video_feed():
#     return StreamingResponse(generate_frames_with_detection(), media_type="multipart/x-mixed-replace; boundary=frame")
#
#
# if __name__ == "__main__":
#     uvicorn.run(
#         app,
#         host="0.0.0.0",
#         port=5001,
#         ws_ping_interval=1000,
#         ws_ping_timeout=500,
#         timeout_keep_alive=300
#     )

# camera_server.py - Background detection, persistent camera feed
import cv2
import time
import asyncio
import requests
from threading import Thread, Lock
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from ultralytics import YOLO

# Camera configuration
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# YOLOv8n model
model = YOLO("yolov8n.pt").half()
model.fuse()
model.conf = 0.5
classNames = ["person"]

# ESP32
ESP32_IP = "http://192.168.1.103/"

# FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared frame and lock
frame_lock = Lock()
shared_frame = None

# Background detection loop
def camera_loop():
    global shared_frame
    last_notify = 0
    cooldown = 1  # seconds between ESP32 notifications

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        person_detected = False
        results = model(frame, half=True, verbose=False, imgsz=320)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls < len(classNames) and classNames[cls] == "person":
                    person_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = round(float(box.conf[0]), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(frame, f"P {conf}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if person_detected and time.time() - last_notify > cooldown:
            last_notify = time.time()
            try:
                requests.get(f"{ESP32_IP}/person", timeout=1)
                print(f"[BG_LOOP] ESP32 notified")
            except Exception as e:
                print(f"[BG_LOOP] ESP32 notification failed: {str(e)}")

        with frame_lock:
            shared_frame = frame.copy()

        time.sleep(1 / 15)  # ~15 FPS


# WebSocket stream
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            start_time = time.time()
            with frame_lock:
                frame = shared_frame.copy() if shared_frame is not None else None

            if frame is None:
                await asyncio.sleep(0.1)
                continue

            _, buffer = cv2.imencode('.jpg', frame, [
                cv2.IMWRITE_JPEG_QUALITY, 50,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ])
            await websocket.send_bytes(buffer.tobytes())

            processing_time = time.time() - start_time
            await asyncio.sleep(max(0, 0.066 - processing_time))  # ~15 FPS
    except Exception as e:
        print(f"WebSocket Error: {str(e)}")
    finally:
        try:
            await websocket.close()
        except RuntimeError as e:
            print(f"[WS Finalize] WebSocket already closed: {str(e)}")


# Optional: MJPEG video feed (e.g., for testing in browser)
def generate_frames_with_detection():
    while True:
        with frame_lock:
            frame = shared_frame.copy() if shared_frame is not None else None

        if frame is None:
            continue

        ret, buffer = cv2.imencode('.jpg', frame, [
            cv2.IMWRITE_JPEG_QUALITY, 50,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1
        ])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames_with_detection(), media_type="multipart/x-mixed-replace; boundary=frame")


# Start background detection
t = Thread(target=camera_loop, daemon=True)
t.start()

@app.get("/occupancy_status")
async def get_occupancy_status():
    """Simple endpoint to check current occupancy status"""
    with frame_lock:
        frame = shared_frame.copy() if shared_frame is not None else None

    if frame is None:
        return {"person_detected": False}

    # Quick detection check
    results = model(frame, half=True, verbose=False, imgsz=320)
    person_detected = any(
        int(box.cls[0]) < len(classNames) and classNames[int(box.cls[0])] == "person"
        for r in results for box in r.boxes
    )

    return {"person_detected": person_detected}

# Run server
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5001,
        ws_ping_interval=1000,
        ws_ping_timeout=500,
        timeout_keep_alive=300
    )
