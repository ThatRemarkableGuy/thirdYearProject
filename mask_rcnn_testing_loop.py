# import the necessary packages
import numpy as np
import argparse
import random
import time
import cv2
import os
import argparse
from collections import Counter



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-m", "--mask-rcnn", required=True, help="base path to mask-rcnn directory")
ap.add_argument("-v", "--visualize", type=int, default=0, help="whether or not we are going to visualize each instance")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="minimum threshold for pixel-wise mask segmentation")
args = vars(ap.parse_args())

labelsPath = os.path.sep.join([args["mask_rcnn"],
	"object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")


colorsPath = os.path.sep.join([args["mask_rcnn"], "colors.txt"])
COLORS = open(colorsPath).read().strip().split("\n")
COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
COLORS = np.array(COLORS, dtype="uint8")


weightsPath = os.path.sep.join([args["mask_rcnn"],
	"frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
	"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])


print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)


wholeImage = cv2.imread(args["image"]);
(imgHeight,imgWidth) = wholeImage.shape[:2]
cornerCoords = np.zeros((4, 2))
noOfInput = 0


def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY, noOfInput, testCord
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(wholeImage,(x,y),1,(255,0,0),-1)
        mouseX,mouseY = x,y
        cornerCoords[noOfInput][0] = mouseX
        cornerCoords[noOfInput][1] = mouseY
        noOfInput += 1
        print(mouseX)
        print(mouseY)
        print('noOfInput is: %s' % (noOfInput))


while(noOfInput < 4):
	cv2.imshow('Test', wholeImage)
	cv2.setMouseCallback('Test', draw_circle)
	cv2.waitKey(20)

#start loop here
objectFound = False
overallStatistics = []
for heightBoxes in range(8, 11):
	for widthBoxes in range(1, 11):

		M = imgHeight//heightBoxes
		N = imgWidth//widthBoxes

		count = 0;

		objectArray = []
		relevantArray = []

		for y in range(0, imgHeight, M):
			for x in range(0, imgWidth, N):
				y1 = y + M
				x1 = x + N
				tiles = wholeImage[y:y+M, x:x+N]
				(H,W) = tiles.shape[:2]
				print("x,y,x+N,y+M: ", x,y,x+N,y+M)

				resizeConstant = 1
				objectFound = False
				while objectFound == False:
					print("resizeConstant is : %s" % (resizeConstant))
					
					cv2.imshow('Test', tiles)
					#Uncomment to save out tiles
					#outputString = '/Users/RyanDivers/masktest2/outputPicture18x20/%s.png' % (pictureNum)
					#print(type(outputString))
					#cv2.imwrite(outputString, tiles)
					#pictureNum += 1
					print('Image dimensions: ', tiles.shape)



					tilesWidth = int(tiles.shape[1] * resizeConstant)
					tilesHeight = int(tiles.shape[0] * resizeConstant)
					dim = (tilesWidth, tilesHeight)
					resizedTiles = cv2.resize(tiles, dim, interpolation = cv2.INTER_AREA)
					W = tilesWidth
					H = tilesHeight
					print('New Image dimensions', resizedTiles.shape)
					cv2.waitKey(1)

					blob = cv2.dnn.blobFromImage(resizedTiles, swapRB=True, crop=False)
					net.setInput(blob)
					print("Count Value is : ", count)
					start = time.time()

					try: #Otherwise error when the input segment is too small
						(boxes,masks) = net.forward(["detection_out_final", "detection_masks"])
					except:
						print("Net Forward Error")
						resizeConstant += 0.1
						if(resizeConstant > 3):
							objectFound = True
						continue


				

					clone = resizedTiles.copy()
					for i in range(0, boxes.shape[2]):
						classID = int(boxes[0, 0, i, 1])
						confidence = boxes[0, 0, i, 2]

						if confidence >  0.5:
							print('Item found')
							objectFound = True
							count += 1
							
							box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
							(startX, startY, endX, endY) = box.astype("int")

							boxW = endX - startX
							boxH = endY - startY

							mask = masks[i, classID]
							try:
								mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_NEAREST)
							except:
								print("Resize Error, input image is too small")
								continue

							
							mask = (mask > args["threshold"])

							roi = clone[startY:endY, startX:endX]

							objectArrayInput = '%s : resolutionConstant %s' % (LABELS[classID], resizeConstant)

							objectArray.append(objectArrayInput)
							if(LABELS[classID] == 'book'):
								relevantArray.append('Not relevant')
							elif(LABELS[classID] == 'bench'):
								relevantArray.append('Not relevant')

							else:
								relevantArray.append('relevant')

							if args["visualize"] > 0:
						
								visMask = (mask * 255).astype("uint8")
								instance = cv2.bitwise_and(roi, roi, mask=visMask)

								
								cv2.imshow("ROI", roi)
								cv2.imshow("Mask", visMask)
								cv2.imshow("Segmented", instance)

							roi = roi[mask]

							color = random.choice(COLORS)
							blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

							clone[startY:endY, startX:endX][mask] = blended

							color = [int(c) for c in color]
							cv2.rectangle(clone, (startX, startY), (endX, endY), color, 2)

							text = "{}: {:.4f}".format(LABELS[classID], confidence)
							cv2.putText(clone, text, (startX, startY - 5),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
						
					if resizeConstant > 3:
						objectFound = True
					else:
						resizeConstant += 0.1
						
					cv2.imshow("Output", clone)
					cv2.waitKey(1)

					print('Testing')




		relCount = Counter(relevantArray)

		additionString = 'relevant: %s' % (relCount['relevant']) + '\n' + 'Not relevant: %s' % (relCount['Not relevant'])
		objectArray.append(additionString)

		outputName = 'Boxes%sx%sRes%s.txt' % (heightBoxes, widthBoxes, resizeConstant)
		inputString = outputName + ' relevant: %s' % (relCount['relevant']) +  ' Not relevant: %s' % (relCount['Not relevant'])
		overallStatistics.append(inputString)
		np.savetxt('/Users/RyanDivers/masktest2/txtOutput/' + outputName, objectArray, delimiter=" ", fmt= "%s")
#end loop here
print(overallStatistics)
np.savetxt('/Users/RyanDivers/masktest2/txtOutput/' + 'Master.txt', overallStatistics, delimiter=" ", fmt="%s")




        
