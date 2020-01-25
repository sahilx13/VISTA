#Author - Sahil Sharma, Bidya Dash

import torchvision
import cv2
from PIL import Image
import time
from Addition import Test
import torch
import torch.nn as nn

def flatten(x):
    N = x.shape[0]  # read in N, C, H, W
    return x.view(N, -1)

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

# T.cuda.device(0)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person',    'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
   'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
   'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

#COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person']

def get_prediction(img, threshold=0):
    img = Image.fromarray(img)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # img = T.Tensor(img)  # Defining PyTorch Transform
    # print(img.shape)
    img = transform(img)  # Apply the transform to the image
    model.cuda()
    img = img.cuda()
    pred = model([img])  # Pass the image to the model
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].detach().cpu().clone().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().clone().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().clone().numpy())
    try:
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]  # Get list of index with score greater than threshold.
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]
    except IndexError:
        pred_boxes = None
        pred_class = None
    return pred_boxes, pred_class


def object_detection_api(img, class_name, threshold=0.70, rect_th=3, text_size=2, text_th=2):

    boxes, pred_cls = get_prediction(img, threshold)  # Get predictions
    classes = ["batman", "superman", "wonderwoman", "Joker", "Persons"]
    if pred_cls is not None:
        for i in range(len(boxes)):
            if pred_cls[i] == 'person':
                y1, x1 = boxes[i][0]
                y2, x2 = boxes[i][1]
                crop = img[int(x1):int(x2), int(y1):int(y2)]
                # cv2.imshow(" ", crop)
                pred = Test.test_image(crop)
                # print(pred)
                predict = torch.max(pred, axis=1)
                #print(predict)
                # print("Predicted class:"+classes[predict[1]])
                # print("Scores:"+str(pred))
                if classes[predict[1]] == class_name:
                    # print(classes[pred[1]])
                    detection = torch.Tensor([[int(y1), int(x1), int(y2), int(x2), 1, predict[0], predict[1]]])
                    # cv2.rectangle(img, (int(y1),int(x1)), (int(y2),int(x2)), color=(255, 0, 0),thickness=rect_th)  # Draw Rectangle with the coordinates
                    # cv2.putText(img, classes[predict[1]], (int(y1),int(x1)+35), cv2.FONT_HERSHEY_SIMPLEX, text_size,
                    #             (0,0 , 255), thickness=text_th)
                    # cv2.imwrite("Results/superman_new2.jpg",img)
                    # break
                    #cv2.putText(img, classes[pred[1]], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),thickness=text_th)
                    return detection
                # cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 0, 255), thickness=rect_th)  # Draw Rectangle with the coordinates
                # cv2.putText(img, pred_cls[i], (int(y1), int(x1)+35), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 255),thickness=text_th)  # Write the prediction class
               # cv2.imwrite("Results/wonderwoman1_new.jpg", img)
    # cv2.imwrite("frameCar2.jpeg", img)
    # cv2.imshow(" ", img)



cap = cv2.VideoCapture("Test/wonderwoman_short.mp4")
# cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # print(frame.shape)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to RGB
    #frame = cv2.resize(frame, (400, 400))
    object_detection_api(frame)
    # time.sleep(0.001)
    if cv2.waitKey(16) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()