#!/usr/bin/env python
# coding=utf-8

import cv2
import argparse
import requests
import json
import os
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle

parse = argparse.ArgumentParser()
parse.add_argument("-i", "--input", default="cutout.MP4", help= "the input file .mp4")
parse.add_argument("-o", "--output", help= "the output dir")
parse.add_argument("--api", default="", help= "api such as: http://127.0.0.1:3000")
parse.add_argument("--fps", type=int, default=10, help= "fps")
args = parse.parse_args()

def distence(point1, point2):
	return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

class Detector(object):
	def __init__(self, device):
		self.device = device
		self.cap = cv2.VideoCapture(device)
		self.fps = self.cap.get(cv2.CAP_PROP_FPS)
		self.cnt = 0
		self.total = 0
		self.cache = Frame([])
		if not os.path.exists('demo/cache/%s'%self.device):
			os.mkdir('demo/cache/%s'%self.device)
	
	def run(self):
		prev_cnt = 0
		interval = self.fps / args.fps
		total = 0
		while self.cap.isOpened():
			ret, frame = self.cap.read()
			if not ret: break
			self.cnt += 1
			cnt = int(self.cnt/interval)
			if cnt <= prev_cnt:
				continue
			prev_cnt = cnt
			cache_file =  'demo/cache/%s/%d'%(self.device,self.cnt)
			if os.path.exists(cache_file):
				with open(cache_file,'rb') as f:
					data = pickle.load(f)
			else:
				if args.api:
					_, image = cv2.imencode('.jpg', frame)
					files = {"f": image.tobytes()}
					r = requests.post("%s/api/pose/detection"%args.api, files=files)
					pose_text = r.text.replace('\0', '')
					d = json.loads(pose_text)
					data = d["data"]
				else:
					from lib import detection
					data = detection(frame)
				with open(cache_file,'wb') as f:
					pickle.dump(data, f)
			for person in data:
				for point in person:
					if point[0]:
						cv2.circle(frame, (int(point[0]),int(point[1])), 1, (0,0,255),1)
			self.cache = self.cache.concat(Frame(data))
			total += self.cache.clear()
			self.cache.draw(frame)

			cv2.imshow('imshow',frame)
			if cv2.waitKey(10)==27:
				break
		total += self.cache.count()
		print('total:',total)
		return total

class Frame(object):
	def __init__(self, data):
		self.persons = [Person(person) for person in data]
		self.persons = [person for person in self.persons if (person.x+person.y)>0]
	def concat(self, frame):
		dist = Frame([])
		print('---------concat--------')
		print('len', len(frame.persons))
		if len(frame.persons)>0:
			for person in self.persons:
				dst = 0
				per = None
				for one in frame.persons:
					scale = min(person.r, one.r)/max(person.r,one.r)
					# if person.id==4 and one.id==21:
					# 	print(scale, person.distence(one), person.r, one.r)
					if person.lost>args.fps/2:
						if scale < 0.90:
							continue
					elif scale < 0.80:
						continue
					tmp = person.distence(one)
					if per is None or tmp<dst:
						dst = tmp
						per = one
				if per is not None and dst < person.r:
					# print(person.id, person.live, person.lost, min(person.r, per.r)/max(person.r,per.r), person.distence(per), person.r, per.r)
					person.moveTo(per)
					dist.persons.append(person)
					frame.persons.remove(per)
				else:
					person.lost += 1
					dist.persons.append(person)

		print('new person', len(frame.persons))
		for one in frame.persons:
			dist.persons.append(one)
		print('current:', len(dist.persons))
		return dist
	def count(self):
		total = 0
		min_live = args.fps
		for person in self.persons:
			if person.live>min_live:
				total += 1
		return total	
	def clear(self):
		max_lost = args.fps * 2
		min_live = args.fps * 2
		persons = []
		total = 0
		for person in self.persons:
			if person.lost<max_lost:
				persons.append(person)
			elif person.live>min_live and person.displacement()>person.r:
				print(person.id, person.displacement(), person.r)
				total += 1
		self.persons = persons
		return total
	def draw(self, frame):
		for i, person in enumerate(self.persons):
			if person.live<person.lost:
				continue
			cv2.putText(frame, '%d:%d-%d'%(person.id, person.live, person.lost), (person.x-20, person.y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
		

class Person(object):
	def __init__(self, person):
		Person.count += 1
		self.id = Person.count
		n,x,y,minx,maxx = 0,0,0,1e8,0
		for point in person:
			if point[0]:
				n+=1
				x += point[0]
				y += point[1]
				# if point[0]<minx: minx = point[0]
				# if point[0]>maxx: maxx = point[0]
		if n>2:
			self.x = x = int(x/n)
			self.y = y = int(y/n)
		else:
			self.x = 0
			self.y = 0
		self.p1 = [self.x, self.y]
		self.p2 = [self.x, self.y]
		self.r = 0 # maxx - minx
		for point in person:
			if point[0]:
				tmp = distence(point, (x, y))
				if tmp>self.r:
					self.r = tmp
		self.live = 0
		self.lost = 0
	def distence(self, person):
		return distence((self.x, self.y), (person.x, person.y))
	def moveTo(self, person):
		if person.x<self.p2[0]: self.p2[0] = person.x
		if person.y<self.p2[1]: self.p2[1] = person.y
		if person.x>self.p1[0]: self.p1[0] = person.x
		if person.y>self.p1[1]: self.p1[1] = person.y
		self.x = person.x
		self.y = person.y
		self.r = person.r
		self.live += 1
		self.lost = 0
	def displacement(self):
		return distence(self.p1, self.p2)

if __name__ == '__main__':
	Person.count = 0
	detector = Detector(args.input)
	detector.run()