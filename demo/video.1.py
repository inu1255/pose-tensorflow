#!/usr/bin/env python
# coding=utf-8

import cv2
import argparse
import requests
import json
import numpy as np

parse = argparse.ArgumentParser()
parse.add_argument("-i", "--input", help= "the input file .mp4")
parse.add_argument("-o", "--output", help= "the output dir")
parse.add_argument("--api", default="http://127.0.0.1:3000", help= "save images")
args = parse.parse_args()

first = 0
def onMouse(event, x, y, flags, prams): 
	global first
	if event == cv2.EVENT_LBUTTONDOWN:
		print(first)
		if first==0:
			first = 1

cv2.namedWindow('imshow')
cv2.setMouseCallback('imshow',onMouse)
def main():
	global first
	cap = cv2.VideoCapture('cutout.MP4')
	fps = cap.get(cv2.CAP_PROP_FPS)
	bx = 9e8
	by = 9e8
	ex = 0
	ey = 0
	term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: break
		if first>0:
			if first==1:
				gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
				fm = cv2.Laplacian(gray, cv2.CV_64F).var()
				print('fm:%d'%fm)
				if fm<400:
					first = 2
					_, image = cv2.imencode('.jpg', frame)
					files = {"f": image.tobytes()}
					r = requests.post("%s/api/pose/detection"%args.api, files=files)
					pose_text = r.text.replace('\0', '')
					d = json.loads(pose_text)
					for point in d["data"][0]:
						if point[0]:
							bx = int(min(bx, point[0]))
							by = int(min(by, point[1]))
							ex = int(max(ex, point[0]))
							ey = int(max(ey, point[1]))
			hsv =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			mask = cv2.inRange(hsv, np.array((0., 30.,10.)), np.array((180.,256.,255.)))
			if first==2:
				# print(bx,by,ex,ey)
				track_window=(bx,by,ex-bx,ey-by)
				print(track_window)
				maskroi = mask[by:ey, bx:ex]
				hsv_roi = hsv[by:ey, bx:ex]
				roi_hist = cv2.calcHist([hsv_roi],[0],maskroi,[180],[0,180])
				cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

				dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
				dst &= mask
				ret, track_window = cv2.CamShift(dst, track_window, term_crit)
				print(ret, track_window)
				pts = cv2.boxPoints(ret)
				pts = np.int0(pts)
				img2 = cv2.polylines(frame,[pts],True, 255,2)
		cv2.imshow('imshow',frame)
		if cv2.waitKey(10)==27:
			break
			


if __name__ == '__main__':
	main()