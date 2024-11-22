import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


cap = cv2.VideoCapture("../videos/people.mp4")  # For Video


model = YOLO("../yoloweights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask=cv2.imread("mask.png")

#tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limitsUp=[103,161,296,161]
limitsDown=[527,489,735,489]

totcountUp=[]
totcountDown=[]

while True:
    success, img = cap.read()
    imgRegion =cv2.bitwise_and(img,mask)

    imgGraphics = cv2.imread("graphics.png",cv2.IMREAD_UNCHANGED)
    img=cvzone.overlayPNG(img,imgGraphics,(730,260))

    results = model(imgRegion, stream=True)
    detections= np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == "person"and conf>0.3:
                #cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                               #scale=0.6, thickness=1,offset=3)
                #cvzone.cornerRect(img, (x1, y1, w, h), l=9,rt=5)
                currarray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currarray))

    resultsTracker = tracker.update(detections)
    cv2.line(img,(limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(0,255,0),5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5,colorR=(255,0,255))
        cvzone.putTextRect(img, f'  {int(Id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cx,cy =x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limitsUp[0]<cx< limitsUp[2] and limitsUp[1]-15<cy<limitsUp[1]+15:
             if totcountUp.count(Id)==0:
                 totcountUp.append(Id)
                 cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (255,0, 0), 5)
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if totcountDown.count(Id)==0:
             totcountDown.append(Id)
             cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (255,0, 0), 5)


    cv2.putText(img,str(len(totcountUp)),(929,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)
    cv2.putText(img, str(len(totcountDown)), (1191,345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 238), 7)
    #cv2.imshow("imgRegion", imgRegion)
    cv2.imshow("Image", img)
    cv2.waitKey(1)