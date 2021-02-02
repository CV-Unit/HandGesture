# coding: utf-8
import numpy as np
import cv2
import time
import os

minValue = 70
sample_nums = 0
counter = 0
gestname = ""
path = ""
saveimg = False
x0 = 400
y0 = 200
height = 200
width = 200

def saveROIImg(img):
    global counter, saveimg, gestname, sample_nums
    if counter > sample_nums:
        saveimg = False
        counter = 0
        gestname = ""
        return 
    counter = counter + 1
    name = gestname + "~" +str(counter)
    print("Saving img:",name)
    cv2.imwrite(path+name + ".png", img)
    time.sleep(0.6)
    
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
    cv2.imshow("gray", gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    cv2.imshow("blur", blur)
    th3 = cv2.adaptiveThreshold(blur, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
    ret,res = cv2.threshold(th3, minValue, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if saveimg == True:
        saveROIImg(res)
    return res

def skinMask(roi):
	skinCrCbHist = np.zeros((256,256), dtype= np.uint8)
	cv2.ellipse(skinCrCbHist, (113,155),(23,25), 43, 0,
                360, (255,255,255), -1)
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

cap = cv2.VideoCapture(0)
framecount = 0
fps = ""
start = time.time()
while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 3)
    frame = cv2.resize(frame, (640,480))

    if ret == True:
        roi = binaryMask(frame, x0, y0, width, height)    
        framecount = framecount + 1
        end  = time.time()
        second = (end - start)
        if( second >= 1):
            fps = 'FrameRate:%sfps' %(framecount)
            start = time.time()
            framecount = 0

    cv2.putText(frame,fps,(400,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,150,0),2,1)
    cv2.putText(frame,'C --> CreateFloder',(400,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,100,200),2,1)
    cv2.putText(frame,'S --> SaveSamples',(400,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,100,200),2,1)

    cv2.imshow('Original',frame)
    cv2.imshow('ROI', roi)

    key = cv2.waitKey(5) & 0xff
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
    elif key == ord('c'):
        gestname = input("输入存放手势的文件夹名称: ")
        sample_nums = int(input("输入存放手势图片数目: "))
        try:
            os.makedirs(gestname)
        except OSError as e:
            print(gestname+'文件夹已创建')
        path = "./"+gestname+"/"
    elif key == ord('s'):
        if gestname=='':
            print("请先输入一个存放文件夹的名字")
        else:
            saveimg = True

    elif key == ord('i'):
        y0 = y0 - 5
    elif key == ord('k'):
        y0 = y0 + 5
    elif key == ord('j'):
        x0 = x0 - 5
    elif key == ord('l'):
        x0 = x0 + 5
        
        

