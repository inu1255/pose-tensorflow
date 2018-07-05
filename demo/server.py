#!/usr/bin/env python
# coding=utf-8

import os
import sys
import numpy as np
from scipy.misc import imread
from flask import Flask, request, jsonify
from lib import detection
from video import Detector, args

app = Flask(__name__)

@app.route('/detection', methods=['POST','GET'])
def detection_():
	f = None
	if request.method == 'POST':
		f = request.files.get('f')
	else:
		f = request.args.get('f')
	if f is None:
		return jsonify({'no': 400, "msg": "缺少文件参数"})
	try:
		image = imread(f, mode='RGB')
	except Exception as e:
		print(e)
		return jsonify({'no': 404, 'msg': '找不到文件'})
	person_conf_multi = detection(image)
	return jsonify({'no': 200, 'data': person_conf_multi.tolist()})

@app.route('/vedio', methods=['POST','GET'])
def vedio_():
	f = request.args.get('f')
	fps = request.args.get('fps', args.fps)
	if f is None:
		return jsonify({'no': 400, "msg": "缺少文件参数"})
	if not os.path.exists(f):
		return jsonify({'no': 404, "msg": "文件不存在"})
	detector = Detector(f, float(fps))
	total = detector.run()
	data = {'total': total}
	return jsonify({'no': 200, 'data': data})

if __name__ == '__main__':
	app.run('127.0.0.1', 3008)
