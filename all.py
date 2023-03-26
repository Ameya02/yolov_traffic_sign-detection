import cv2
import numpy as np
import os 
import eel
from tkinter import filedialog
from tkinter import messagebox
import tkinter as tk
root = tk.Tk()
import pathlib

def checkfile(path):
     path = os.path.expanduser(path)

     if not os.path.exists(path):
        return path

     root, ext = os.path.splitext(os.path.expanduser(path))
     dir       = os.path.dirname(root)
     fname     = os.path.basename(root)
     candidate = fname+ext
     index     = 0
     ls        = set(os.listdir(dir))
     while candidate in ls:
             candidate = "{}_{}{}".format(fname,index,ext)
             index    += 1
     return os.path.join(dir,candidate)
eel.init('web')
@eel.expose
def obdetect():
    # This below line will reads the weights and config file and create the network
    # our TS model in V4 format
    net = cv2.dnn.readNet('./data/ts-config.cfg', './data/ts-model.weights')
    # the below 2 line will used if our machine consist of Gpu if it consist the Gpu it will used to inecrease the framerate
    # and performance of our detectio
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    with open("./data/ts.names", "r") as f:
        classes = f.read().splitlines()

    # the below statment will start our camera for detection
    cap = cv2.VideoCapture(0)
    # This will be used for font of our detected object name
    font = cv2.FONT_HERSHEY_PLAIN
    # it will generate the colors for different classes of object
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    window_name='output'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)


    while True:
        _, img = cap.read()
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)

        #gathering information from network and push to layers of object
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

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
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y + 20), font, 1, (255, 0, 0), 2)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif cv2.getWindowProperty(window_name,cv2.WND_PROP_VISIBLE) < 1:
            break


    cap.release()
    cv2.destroyAllWindows()

@eel.expose
def viddetect():
    net = cv2.dnn.readNet('./data/ts-config.cfg', './data/ts-model.weights')
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    with open("./data/ts.names", "r") as f:
        classes = f.read().splitlines()
    root.withdraw()
    video = filedialog.askopenfilename(filetypes=[("Video",".MP4"),("Video",".MKV"),("Video",".WAV")])
    
    
    cap = cv2.VideoCapture(video)
    if cap.isOpened() == False:
        return
    file_extension = pathlib.Path(video).suffix
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
   
    size = (frame_width, frame_height)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    #window_name = 'output'
    #cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    outputfile=checkfile(os.getcwd()+'/results/output'+file_extension)
    result = cv2.VideoWriter(outputfile, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
    while True:
        _, img = cap.read()
        
        try:
            height, width, _ = img.shape
        
       
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
            net.setInput(blob)
            output_layers_names = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_names)

            boxes = []
            confidences = []
            class_ids = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.2:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i], 2))
                    color = colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)
                    print("conf"+confidence)
            #cv2.imshow(window_name, img)
            result.write(img)
            #key = cv2.waitKey(1)
            #if key == 27:
            #break
            #elif cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            #break

        except [AttributeError, Exception ]:
            print("Video Processed")
            break
        
    

    cap.release()
    tk.messagebox.showinfo(title="Success", message="The output is saved on the device")
    result.release()
    cv2.destroyAllWindows()

@eel.expose
def imgdetect():
    net = cv2.dnn.readNet('./data/ts-config.cfg', './data/ts-model.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    classes = []
    with open("./data/ts.names", "r") as f:
        classes = f.read().splitlines()
    root.withdraw()
    video = filedialog.askopenfilename(filetypes =[("image", ".jpeg"),
                    ("image", ".png"),
                    ("image", ".jpg"),])
    file_extension = pathlib.Path(video).suffix
    img = cv2.imread(video)
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
    window_name = 'output'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)


    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence >0.5 :
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w,h])
                confidences.append((float(confidence))) 
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i],2))
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y+3), font, 0.4, (255, 255, 255), 1,cv2.LINE_AA)
    outputfile=checkfile(os.getcwd()+'/results/output'+file_extension)
    cv2.imwrite(outputfile,img)
  
    tk.messagebox.showinfo(title="Success", message="The output is saved on the device")
    cv2.imshow(window_name, img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


eel.start('index.html',size=(1000,700) , mode='edge')

