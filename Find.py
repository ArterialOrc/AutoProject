from keras.models import load_model
import argparse
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils import contours

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained Keras model")
ap.add_argument("-l", "--label-bin", required=True,
                help="path to label binarizer")
args = vars(ap.parse_args())

model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

img = cv2.imread('./2car.jpg')
#convert my image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
contour = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contour,_=contours.sort_contours(contour[0])

rectangle = [0,0]
array_variants=[]
x_arr=[]
y_arr=[]
w_arr=[]
h_arr=[]
for cnt in contour:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    if len(approx)>=4 and len(approx)<=10: #polygons with 4 points is what I need.
        area = cv2.contourArea(cnt)
        rectangle = [cv2.contourArea(cnt), cnt, approx]
        x, y, w, h = cv2.boundingRect(rectangle[1])
        if(w>30 and h>20):
            image = img[y:y+h,x:x+w]
            image = cv2.resize(image, dsize=(64, 64))
            image = image.astype("float") / 255.0
            image = image[None, :]
            preds=model.predict(image)
            if(preds[0][1]>preds[0][0]):
                x_arr.append(x)
                y_arr.append(y)
                w_arr.append(w)
                h_arr.append(h)
                i=preds[0][1]
                array_variants.append(i)

array_variants=np.array(array_variants)
j=array_variants.argmax()
x, y, w, h =x_arr[j],y_arr[j],w_arr[j],h_arr[j]
draw=img[y:y+h,x:x+w]

plt.imshow(draw,cmap = 'gray')
plt.show()