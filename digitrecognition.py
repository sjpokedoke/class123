import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

#setting an https context
if (not os.environ.get("PYTHONHTTPSVERIFY", '') and getattr(ssl, "_create_unverified_context", None)):
    ssl._create_default_https_context = ssl._create_unverified_context

#fetching the data
X, y = fetch_openml("mnist_784", version = 1, return_X_y = True)

print(pd.Series(y).value_counts())

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
nclasses = len(classes)

#splitting the data
xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)

xtrainscaled = xtrain/255.0
xtestscaled = xtest/255.0

clf = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(xtrainscaled, ytrain)
ypred = clf.predict(xtestscaled)

accuracy = accuracy_score(ytest, ypred)
print("Accuracy is: ", accuracy)

#starting the camera
cap = cv2.VideoCapture(0)

while(True):
    try:
        #capture frame by frame
        ret, frame = cap.read()

        #changing everything in the frame to gray
        gray = cv2.cvtColor(frame, cv2.COlOR_BGR2GRAY)
        
        #drawing a box in the center of the video
        height, width = gray.shape

        upperleft = (int(width/2-56), int(height/2-56))
        bottomright = (int(width/2+56), int(height/2+56))

        cv2.rectangle(gray, upperleft, bottomright, (0, 255, 0), 2) 

        #only considering the area inside the rectangle
        roi = gray[upperleft[1]:bottomright[1], upperleft[0]:bottomright[0]]

        #converting cv2 to pil
        im_pil = Image.fromarray(roi)

        #convert to gray scale image-"L" l format means each pixel is represented by a single value form 0 to 255
        image_bw = im_pil.convert("L")
        image_bwresize = image_bw.resize((28, 28), Image.ANTIALIAS)

        #inverting the image
        image_bwresizeinverted = PIL.ImageOps.invert(image_bwresize)
        pixel_filter = 20

        #converting to scaler quantity
        minpixel = np.percentile(image_bwresizeinverted, pixel_filter)

        #using clip to limit the values from 0,255
        image_bwresizeinvertedscaled = np.clip(image_bwresizeinverted-minpixel, 0, 255)
        maxpixel = np.max(image_bwresizeinverted)

        #converting into an array
        image_bwresizeinvertedscaled = np.asarray(image_bwresizeinvertedscaled)/maxpixel

        #creating a test sample and making a prediction
        testsample = np.array(image_bwresizeinvertedscaled).reshape(1, 784)
        testprediction = clf.predict(testsample)

        print("Predicted class is: ", testprediction)

        #display the resulting frame
        cv2.imshow("frame", gray)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()