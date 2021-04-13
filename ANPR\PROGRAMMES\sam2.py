#!/usr/bin/python3

import jetson.inference
import jetson.utils
import argparse
import sys
import tempfile
from PIL import Image  
import numpy
import cv2
import pytesseract
import time
import datetime
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")
is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]
try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
i=0
l=1
with tempfile.TemporaryDirectory() as tmpdirname: 
	while True:
		
		img, width, height = camera.CaptureRGBA(zeroCopy=1)
		imgpath = tmpdirname +'test'+str(i)+'.jpg'
		jetson.utils.saveImageRGBA(imgpath, img, width, height)
		
		detections = net.Detect(img, overlay=opt.overlay)
		
		
		print("detected {:d} objects in image".format(len(detections)))
		
		for detection in detections:	
			k = detection.Confidence
			confidence = k *100
		
			for confidence in range (0,1):
				print(detection)
				imgpath1 = tmpdirname +'test'+str(i)+'.jpg'
				im = Image.open(imgpath1)
				cropped = im.crop((detection.Left, detection.Top, detection.Right, detection.Bottom))
				path = "/home/jetson/Pictures/Plates/"+'cropped'+str(i) + ".jpg"
				cropped.save(path)
				data = pytesseract.image_to_string(path, lang='eng', config='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

				print(data)
				dt = datetime.datetime.now()
				file1 = open('output.txt','+a')
				file1.write(str(data) + dt.strftime("\n \t %X \t %x \n"))
				file1.close()
		i=i+1
			
		output.Render(img)

		output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

		net.PrintProfilerTimes()
			

		if not output.IsStreaming():
			break
			



