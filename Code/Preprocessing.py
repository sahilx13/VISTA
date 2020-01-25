# Author : Sahil Sharma, Bidya Dash
# import the necessary packages
from imutils import paths
import argparse
import requests
import cv2
import os
import random
from sklearn.model_selection import train_test_split
import numpy as np


class WebScrape:
    def __init__(self, in_path, out_path):
        self.rows = open(in_path).read().strip().split("\n")
        self.out_path = out_path

    def download_image(self):
        # print(rows)
        total = 0
        # loop the URLs
        for url in self.rows:
            try:
                # try to download the image
                r = requests.get(url, timeout=60)

                # save the image to disk
                p = os.path.sep.join([self.out_path + "{}.jpg".format(str(total).zfill(8))])
                f = open(p, "wb")
                f.write(r.content)
                f.close()

                # update the counter
                print("[INFO] downloaded: {}".format(p))
                total += 1

            # handle if any exceptions are thrown during the download process
            except:
                print("[INFO] error downloading {}...skipping".format(p))

    def delete_invalid_image(self):
        # loop over the image paths we just downloaded
        for imagePath in paths.list_images(self.out_path):
            # initialize if the image should be deleted or not
            delete = False

            # try to load the image
            try:
                image = cv2.imread(imagePath)

                # if the image is `None` then we could not properly load it
                # from disk, so delete it
                if image is None:
                    delete = True

            # if OpenCV cannot load the image then the image is likely
            # corrupt so we should delete it
            except:
                print("Except")
                delete = True

            # check to see if the image should be deleted
            if delete:
                print("[INFO] deleting {}".format(imagePath))
                os.remove(imagePath)


class DataPreprocess:
    def resize(self, out_path, write_loc):
        # loop over the image paths we just downloaded
        count = 0
        for imagePath in paths.list_images(out_path):
            try:
                image = cv2.imread(imagePath)
                image = cv2.resize(image, (224, 224))
                cv2.imwrite(write_loc+str(count)+".jpg", image)
                count += 1
            except Exception as e:
                print(e)
                print("couldn't write this image, skipping")


    def prepare(self, file_loc, test_size=0.25):

        # grab the image paths and randomly shuffle them
        imagePaths = sorted(list(paths.list_images(file_loc)))
        random.seed(42)
        random.shuffle(imagePaths)
        data = []
        labels = []
        # count = 0
        label1 = 0
        label0 = 0
        label2 = 0
        label3 = 0
        label4 = 0
        total = 0
        # loop over the input images
        for imagePath in imagePaths:
            # load the image, pre-process it, and store it in the data list
            image = cv2.imread(imagePath)
            data.append(image)

            # extract the class label from the image path and update the
            # labels list
            label = imagePath.split(os.path.sep)[-2]
            # print(label)
            total += 1
            if label == "Batman":
                label = 0
                label0 += 1
            elif label == "Superman":
                label = 1
                label1 += 1
            elif label == "Wonderwoman":
                label = 2
                label2 += 1
            elif label == "Joker":
                label = 3
                label3 += 1
            elif label == "Persons":
                label = 4
                label4 += 1
            labels.append(label)
        # print(count)
        data = np.array(data)
        labels = np.array(labels)
        print("Batman : " + str(label0))
        print("Superman : " + str(label1))
        print("Wonderwoman : " + str(label2))
        print("Joker : " + str(label3))
        print("Persons : " + str(label4))
        print("Total images : " + str(total))
        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing
        (trainX, testX, trainY, testY) = train_test_split(data,
                                                          labels, test_size=test_size, random_state=42)
        print("Split : " + str(1-test_size) + " : " + str(test_size))
        print("Splitting the set into Training : " + str(int(total * (1-test_size))) + " Test : " + str(int(total * test_size)))
        return trainX, testX, trainY, testY


def main():
    in_path = "url/persons.txt"
    out_path = "Output/Persons/"
    write_loc = "Data/Persons/"
    file_loc = "Data"
    ws = WebScrape(in_path, out_path)
    # ws.download_image()
    # ws.delete_invalid_image()
    process = DataPreprocess()
    process.resize(out_path, write_loc)
    # print(process.prepare("Data"))
    # process.prepare("Data")

if __name__ == "__main__":
    main()