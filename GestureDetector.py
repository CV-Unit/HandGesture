# coding: utf-8

from keras.models import load_model
import numpy as np
import cv2
import pickle
import time

minValue = 70
x0 = 400
y0 = 200
height = 200
width = 200

def binaryMask(frame, x0, y0, width, height):
    cv2.rectangle(frame,(x0,y0),(x0+width, y0+height),(0,255,0))
    roi = frame[y0:y0+height, x0:x0+width]
    skin = skinMask(roi)
    cv2.imshow("skin", skin)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(skin, kernel)
    cv2.imshow("erosion", erosion)
    dilation = cv2.dilate(erosion, kernel)
    cv2.imshow("dilation", dilation)
    gray = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
    ret,res = cv2.threshold(th3, minValue, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return res

def skinMask(roi):
	skinCrCbHist = np.zeros((256,256), dtype= np.uint8)
	cv2.ellipse(skinCrCbHist, (113,155),(23,25), 43,
                0, 360, (255,255,255), -1)
	YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
	(y,Cr,Cb) = cv2.split(YCrCb)
	skin = np.zeros(Cr.shape, dtype = np.uint8)
	(x,y) = Cr.shape
	for i in range(0, x):
		for j in range(0, y):
			if skinCrCbHist [Cr[i][j], Cb[i][j]] > 0:
				skin[i][j] = 255
	res = cv2.bitwise_and(roi,roi, mask = skin)
	return res

MODEL_NAME = "samples_model.h5"
LABEL_NAME = "samples_labels.dat"

with open(LABEL_NAME, "rb") as f:
    lb = pickle.load(f)

model = load_model(MODEL_NAME)

#video="http://admin:admin@192.168.10.102:8081/"
#cap = cv2.VideoCapture(video)

cap = cv2.VideoCapture(0)

framecount = 0
fps = ""

start = time.time()
while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640,480))

    if ret == True:

        roi = binaryMask(frame, x0, y0, width, height)
        roi1 = cv2.resize(roi,(100,100))
        roi1 = np.expand_dims(roi1, axis=2)
        roi1 = np.expand_dims(roi1, axis=0)
        prediction = model.predict(roi1)

        gesture = lb.inverse_transform(prediction)[0]

        cv2.putText(frame, 'HandGesture:', (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 200), 3)
        cv2.putText(frame,gesture,(430, 170), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

        framecount = framecount + 1
        end  = time.time()
        second = (end - start)
        if( second >= 1):
            fps = 'FrameRate:%sfps' %(framecount)
            start = time.time()
            framecount = 0

    cv2.putText(frame,fps,(400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,150,0),2,1)
    cv2.imshow('Original',frame)
    cv2.imshow('ROI', roi)

    key = cv2.waitKey(5) & 0xff

    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break

    elif key == ord('i'):
        y0 = y0 - 5
    elif key == ord('k'):
        y0 = y0 + 5
    elif key == ord('j'):
        x0 = x0 - 5
    elif key == ord('l'):
        x0 = x0 + 5