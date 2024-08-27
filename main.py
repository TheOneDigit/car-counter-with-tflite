import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
from sort import *
import cvzone

PATH_TO_MODEL_DIR = 'model.tflite'
PATH_TO_LABELS = 'labels.txt'
VIDEO_PATH = 'test_video.webm'
OUTPUT_PATH = 'output_video.mp4'
MIN_CONF_THRESH = float(0.5)

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [400, 297, 400, 1000]
total_count = []

interpreter = tflite.Interpreter(model_path=PATH_TO_MODEL_DIR)
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))


frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('Break: End of video')
        break
    
    # Pre-process the frame
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imH, imW, _ = frame.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
        print('There is no floating model')

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    detections = np.empty((0, 5))

    for i in range(len(scores)):
        if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            
            object_name = labels[int(classes[i])]
            label = '%s: %.2f%%' % (object_name, scores[i] * 100)  # Example: 'person: 72.00%'

            current_array = np.array([xmin, ymin, xmax, ymax, scores[i]])
            detections = np.vstack((detections, current_array))

    results_tracker = tracker.update(detections)
    limits = [400, 297, 400, 1000]
    cv2.line(frame, (0, 297), (imW, 297), (0, 0, 255), 2)

    for result in results_tracker:
        x1, y1, x2, y2, track_id = result
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        track_id = int(track_id)

        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(frame, f'ID: {track_id}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        if 0 < cx < imW and 297 - 20 < cy < 297 + 20:
            if total_count.count(track_id) == 0:
                total_count.append(track_id)

    cvzone.putTextRect(frame, f'Count: {len(total_count)}', (50, 50))
    
    # Display the frame
    cv2.imshow('Processed Frame', frame)
    out.write(frame)

    # Stop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

