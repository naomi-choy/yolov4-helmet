# YOLO object detection
import cv2 as cv
import numpy as np
import time

configPath = "./configs/yolov4-helmet-detection.cfg"
weightPath = "./configs/yolov4-helmet-detection.weights"
names = "./configs/yolov4-helmet-detection.names"

# Load names of classes and get random colors
classes = open(names).read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet(configPath, weightPath)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

ln = net.getLayerNames()
# print(len(ln), ln)
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def execute(img):
    # construct a blob from the image
    blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=False, crop=False)
    r = blob[0, 0, :, :]

    net.setInput(blob)
    # t0 = time.time()
    outputs = net.forward(ln)
    # t = time.time()
    # print('time=', t-t0)

    r0 = blob[0, 0, :, :]
    r = r0.copy()

    boxes = []
    confidences = []
    classIDs = []
    h, w = img.shape[:2]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    execute(frame)

    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
      break