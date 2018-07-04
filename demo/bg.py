import numpy as np
import cv2

#BackgroundSubtractorMOG2
#opencv自带的一个视频
cap = cv2.VideoCapture('./cutout.mp4')
#创建一个3*3的椭圆核
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
#创建BackgroundSubtractorMOG2
fgbg = cv2.createBackgroundSubtractorMOG2()

while(cap.isOpened()):
	ret, frame = cap.read()
	if ret:
		fgmask = fgbg.apply(frame)
		#形态学开运算去噪点
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
		fgmask = cv2.dilate(fgmask, kernel1)
		#寻找视频中的轮廓
		im, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		bg = np.full(fgmask.shape, 0)

		for c in contours:
			#计算各轮廓的周长
			perimeter = cv2.arcLength(c,True)
			if perimeter > 188:
				x,y,w,h = cv2.boundingRect(c)
				bg[y:y+h,x:x+w] = 255
		frame[bg<128] = 200
		cv2.imshow('frame',frame)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
	else:
		break

cap.release()
cv2.destroyAllWindows()