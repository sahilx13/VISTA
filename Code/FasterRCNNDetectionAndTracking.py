# Author - Sahil Sharma, Bidya Dash

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from Object_detection import object_detection_api
from utils import *
from sort import *

classes = ["batman", "superman", "wonderwoman", "Joker", "Persons"]

# from IPython.display import clear_output
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
# initialize Sort object and video capture
vid = cv2.VideoCapture("../Test/wonderwoman_short.mp4")
mot_tracker = Sort()#while(True):
classname = "wonderwoman"
# classes = ["batman", "superman", "wonder-woman", "joker", "person"]
count = 0
obj = []
frames = {}
previous_frame = None
while True:
    ret, frame = vid.read()
    if not ret:
        break
    detections = object_detection_api(frame, classname)
    # detections = detect_image(pilimg)
    #detections = detections[0].unsqueeze(0)
    # print(detections.size())
    if detections is not None:
        tracked_objects = mot_tracker.update(detections)
        unique_labels = detections[:, -1].unique()
        n_cls_preds = len(unique_labels)
        # print(tracked_objects)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:

            color = colors[int(obj_id) % len(colors)]
            color = [i * 255 for i in color]
            cls = classes[int(cls_pred)]
            cv2.rectangle(frame, (int(x1), int(y1)),(int(x2), int(y2)), color, 4)
            # cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60,y1), color, -1)
            cv2.putText(frame, cls + "-" + str(int(obj_id)),
                                            (int(x1), int(y1)+ 25), cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (255,255,255), 3)
            if obj_id not in obj:
                obj.append(obj_id)
                cv2.imwrite("../Results/frames/Wonderwoman/Start_frame"+str(obj_id)+".jpg", frame)
                frames[obj_id] = []
                frames[obj_id].append(count)
            elif obj_id in obj:
                cv2.imwrite("../Results/frames/Wonderwoman/Last_frame"+str(obj_id)+".jpg", frame)
                frames[obj_id].append(count)
    count += 1
    cv2.imshow("", frame)
    cv2.imwrite("../Results/wonderwoman/"+str(count)+".jpg", frame)
    if cv2.waitKey(16) & 0xFF == ord("q"):
        break
vid.release()
cv2.destroyAllWindows()
f = open("../Results/frames/"+classname+"/wonderwoman_test.txt", "w")
f.write(str(frames))
f.write(str(count))
f.close()

# Video Generating function
def generate_video():
    image_folder = '.'  # make sure to use your folder
    video_name = 'supermaniac.avi'
    os.chdir("../Results/superman/")
    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or
              img.endswith(".jpeg") or
              img.endswith("png")]
    # Array images should only consider
    # the image files ignoring others if any
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 1, (width, height))
    # Appending the images to the video one by one
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
        # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated

