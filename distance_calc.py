import cv2 as cv
import numpy as np
import time
from traffic_signal import read_traffic_lights_object
from tkinter import messagebox
import tkinter as tk
from speedlimit import  read_speed_limit
root = tk.Tk()
root.withdraw()
from alert_sms import *
# Distance constants
KNOWN_DISTANCE = 20 # CM
SIGN_WIDTH = 7.0  # CM
SIGNAL_WIDTH = 7.0
Distance_level = 0
travedDistance = 0
changeDistance = 0
velocity = 0
# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
RED = (0, 0, 255)

# defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file
class_names = []
with open("./data/ts.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('./data/ts-config.cfg', './data/ts-model.weights')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

# object detector funciton /method
def image_ref_detector(image, Distance_level):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]

        label = "%s : %f" % (class_names[classid], score)
        if Distance_level < 10:
            Distance_level = 10

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 2:  # person class id
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
        elif classid == 3:
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data.
    return data_list, Distance_level,


def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length


# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance

def speedFinder(distance, takenTime):
    speed = distance/takenTime
    return speed

def averageFinder(valuesList, numberElements):
    sizeOfList = len(valuesList)
    lastMostElement = sizeOfList - numberElements
    lastPart = valuesList[lastMostElement:]
    average = sum(lastPart)/(len(lastPart))
    return average


# reading the reference image from dir
ref_stop = cv.imread('./cases/stop.jpg')
ref_signal = cv.imread("./cases/traffic-signal.jpeg")
stop_data, distance_level = image_ref_detector(ref_stop,Distance_level)
signal_data, distance_level = image_ref_detector(ref_signal,Distance_level)
print(stop_data,signal_data)
stop_width_in_rf = stop_data[0][1]
signal_width_in_rf = signal_data[0][1]

# finding focal length

focal_stop = focal_length_finder(KNOWN_DISTANCE, SIGN_WIDTH, stop_width_in_rf)
focal_signal = focal_length_finder(KNOWN_DISTANCE,SIGNAL_WIDTH,signal_width_in_rf)
speedList = []
DistanceStop = []
DistanceSignal = []
distance = 0
averageSpeed = 0
intialDisntace = 0
stationary_counter = 0
window_name ="output"
cap = cv.VideoCapture(0)
col = "red"
color = RED
speedlimit =0
while True:
    intialTime = time.time()
    distance_stop = distance_finder(focal_stop, KNOWN_DISTANCE, stop_width_in_rf)
    distance_signal = distance_finder(focal_signal, KNOWN_DISTANCE, signal_width_in_rf)
    DistanceStop.append(distance_stop)
    DistanceSignal.append(distance_signal)
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    yoloNet.setInput(blob)
    #gathering information from network and push to layers of object
    output_layers_names = yoloNet.getUnconnectedOutLayersNames()
    layerOutputs = yoloNet.forward(output_layers_names)

    #initialization
    boxes = []
    confidences = []
    class_ids = []
    #for each detection from each utput layers
    #get the confidence level ,class ids , bounding boxes

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    # it will remove the duplicate detections in our detection
    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes) > 0:


        for i in indexes.flatten():
            if class_names[class_ids[i]] == 'stop':
                distance = distance_finder(focal_stop, SIGN_WIDTH, w)
                DistanceStop.append(distance)
                avergDistnce = averageFinder(DistanceStop, 6)
                # print(avergDistnce)
                roundedDistance = round((avergDistnce * 0.0254), 2)
                # Drwaing Text on the screen
                Distance_level = int(distance)
                if intialDisntace != 0:
                    changeDistance = distance - intialDisntace
                    distanceInMeters = changeDistance * 0.0254
                    velocity = speedFinder(distanceInMeters, changeInTime)
                    speedList.append(velocity)
                    averageSpeed = averageFinder(speedList, 6)
                # intial Distance
                intialDisntace = avergDistnce

                changeInTime = time.time() - intialTime

                if averageSpeed < 0:
                    averageSpeed = averageSpeed * -1
                print(averageSpeed)

                if round(distance, 1) > 7 and round(distance, 1) < 20:
                    if stationary_counter!=6:
                        stationary_counter += 1
                    else:
                        col="green"
                        color=GREEN
                else:
                    stationary_counter -=1
                    if stationary_counter < 0:
                        col="red"
                        color=RED
                        stationary_counter=0


                print(stationary_counter)
                if col=="red":
                    if averageSpeed < 10 and round(distance, 2) < 6:
                        messagebox.showwarning("Warning","Traffic Violated")
                        # sms_alert("ABC Mathur","MH09AB6748","AB73789938HB","Stop Sign Broken")
                        sms_alert_twilio("ABC Mathur", "MH09AB6748", "AB73789938HB", "Stop Sign Broken")
                        # sms_alert_f2sms("ABC Mathur", "MH09AB6748", "AB73789938HB", "Stop Sign Broken",7977508215,8082283288)

            elif class_names[class_ids[i]] == "trafficlight":

                distance = distance_finder(focal_signal, SIGNAL_WIDTH, w)
                stop_flag = read_traffic_lights_object(img, x, y, w, h)
                color = GREEN if stop_flag else RED
                print(stop_flag)
                DistanceSignal.append(distance)
                avergDistnce = averageFinder(DistanceSignal, 6)
                # print(avergDistnce)
                roundedDistance = round((avergDistnce * 0.0254), 2)
                # Drwaing Text on the screen
                Distance_level = int(distance)
                if intialDisntace != 0:
                    changeDistance = distance - intialDisntace
                    distanceInMeters = changeDistance * 0.0254
                    velocity = speedFinder(distanceInMeters, changeInTime)
                    speedList.append(velocity)
                    averageSpeed = averageFinder(speedList, 6)
                # intial Distance
                intialDisntace = avergDistnce

                changeInTime = time.time() - intialTime
                if averageSpeed < 0:
                    averageSpeed = averageSpeed * -1
                print(averageSpeed)

                if averageSpeed < 10 and round(distance, 1) < 6 and stop_flag == False:
                    messagebox.showwarning("Warning","Traffic Violated")
                    sms_alert("ABC Mathur","MH09AB6748","AB73789938HB","Red Signal Broken")

            elif class_names[class_ids[i]]=="speedlimit":
                if speedlimit:
                    time.sleep(2)


                speedlimit = read_speed_limit(img,x,y,w,h)

                distance = distance_finder(focal_signal, SIGN_WIDTH, w)

                DistanceStop.append(distance)
                avergDistnce = averageFinder(DistanceStop, 6)
                # print(avergDistnce)
                roundedDistance = round((avergDistnce * 0.0254), 2)
                # Drwaing Text on the screen
                Distance_level = int(distance)
                if intialDisntace != 0:
                    changeDistance = distance - intialDisntace
                    distanceInMeters = changeDistance * 0.0254
                    velocity = speedFinder(distanceInMeters, changeInTime)
                    speedList.append(velocity)
                    averageSpeed = round(averageFinder(speedList, 6),1)
                # intial Distance
                intialDisntace = avergDistnce

                changeInTime = time.time() - intialTime
                if averageSpeed < 0:
                    averageSpeed = averageSpeed * -1
                print(speedlimit,averageSpeed*5+20)
                if round(distance, 1) < 15:
                    if speedlimit!=0 and averageSpeed*5+20 > speedlimit:
                        messagebox.showwarning("Warning", "Traffic Violated")
                        sms_alert("Omkar Shinde","MH09AB6748","AB73789938HB","Red Signal Broken")

            label = str(class_names[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.putText(img, label + " " + confidence, (x, y + 20), FONTS, 1, (255, 0, 0), 2)
    cv.rectangle(img, (10, 5), (130, 28), BLACK, -1)
    cv.putText(img, f'Dis: {round(distance, 2)} CM', (13, 17), FONTS, 0.48, COLORS[2], 2)


    cv.imshow(window_name, img)
    key = cv.waitKey(1)
    if key == 27:
        break
    elif cv.getWindowProperty(window_name,cv.WND_PROP_VISIBLE) < 1:
        break


cap.release()
cv.destroyAllWindows()

