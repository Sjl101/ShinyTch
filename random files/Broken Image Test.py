from os import listdir
import cv2

yourDirectory = 'E:/Training_Data/Train/'
#for filename in listdir('C:/tensorflow/models/research/object_detection/images/train'):
for filename in listdir(yourDirectory):
  if filename.endswith(".jpg"):
    print(yourDirectory+filename)
    #cv2.imread('C:/tensorflow/models/research/object_detection/images/train/'+filename)
    cv2.imread(yourDirectory+filename)