import datetime
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

CONFIDENCE_THRESHOLD = 0.6
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

model=YOLO('best.pt')
tracker = DeepSort(max_age=50)

cap=cv2.VideoCapture('Baby_sleeping_timelapse.mp4')
# cap=cv2.imread('sleeping_baby.jpg')

my_file = open("class.txt", "r")
data = my_file.read()
class_list = data.split("\n")

while True:
    start = datetime.datetime.now()

    ret, frame = cap.read()
    if not ret:
        print('Cam Error')
        break

    detection = model.predict(source=[frame], save=False)[0]
    results = []

    for data in detection.boxes.data.tolist(): # data : [xmin, ymin, xmax, ymax, confidence_score, class_id]
        confidence = float(data[4])
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        label = int(data[5])
        results.append([[xmin, ymin, xmax-xmin, ymax-ymin], confidence, label])
        
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.putText(frame, class_list[label]+' '+str(round(confidence, 3)) + '%', (xmin, ymin), cv2.FONT_ITALIC, 1, WHITE, 2)

        

    tracks = tracker.update_tracks(results, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1) # id 표시 박스
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
        print("track_id : ", track_id)
        
    end = datetime.datetime.now()

    total = (end - start).total_seconds()
    # print(f'Time to process 1 frame: {total * 1000:.0f} milliseconds')

    fps = f'FPS: {1 / total:.2f}'
    cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    
    # 시작 및 종료
    cv2.imshow('frame', frame)

    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()