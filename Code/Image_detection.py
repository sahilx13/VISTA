import torchvision
import cv2
from PIL import Image
import time
import matplotlib.pyplot as plt
from Addition import Test, Flatten
import torch

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
    # print(pred[0]['labels'])
    # print(pred[0]['boxes'])
    # print(pred[0]['scores'])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].detach().cpu().clone().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().clone().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().clone().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]  # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    pred_score = pred_score[:pred_t + 1]
    return pred_boxes, pred_class, pred_score


def object_detection_api(img, classes, threshold=0.4, rect_th=3, text_size=1, text_th=3):
    boxes, pred_cls, score = get_prediction(img, threshold)  # Get predictions
    # print(boxes)
    # print(pred_cls)
    # print(score)
    for i in range(len(boxes)):
        if pred_cls[i] == 'person':
            (y1, x1) = boxes[i][0]
            (y2, x2) = boxes[i][1]
            # print(int(x1))
            # print(x2)
            # print(y1)
            # print(y2)
            crop = img[int(x1):int(x2), int(y1):int(y2)]
            # plt.figure(figsize=(20, 30))
            # plt.imshow(crop)
            # plt.show()
            pred = Test.test_image(crop)
            print(classes[pred[1]])
            if classes[pred[1]] in classes:
                detection = torch.Tensor([[y1, x1, y2, x2, 1, pred[0], pred[1]]])
                return detection
            # cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)  # Draw Rectangle with the coordinates
            # cv2.putText(img, classes[pred[1]], (int(y1), int(x1)+25), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),thickness=text_th)  # Write the prediction class
            return None
    # plt.figure(figsize=(40, 40))
    # plt.imshow(img)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()


# cap = cv2.VideoCapture("disaster.avi")
cap = cv2.VideoCapture("../Test/batman-video.mp4")
# # cap = cv2.imread("survivor2.jpg")
# cap = cv2.imread("Test/joker.jpg")
# # cap = cv2.imread("puppy-dog.jpeg")
classes = ["Batman", "Superman", "Wonder-Woman", "Joker"]
object_detection_api(cap, classes)
