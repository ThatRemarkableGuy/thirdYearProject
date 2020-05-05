import numpy as np
import argparse
import cv2


#parse arguements 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="input file path")
ap.add_argument("-o", "--output", required=True, help="output file path")
args = vars(ap.parse_args())


vs = cv2.VideoCapture(args["input"])
writer = None
frameCount = 0

firstTime = True
while True:
	#checks if the frame has been read
	(read, frame) = vs.read()

	#The frame is not read when the end of the video is reached
	if not read:
		break

	wholeImage = frame

	if(firstTime):
		cornerCoords = np.zeros((4,2))
		noOfInput = 0

		def draw_circle(event,x,y,flags,param):
		    global mouseX,mouseY, noOfInput
		    if event == cv2.EVENT_LBUTTONDOWN:
		        cv2.circle(wholeImage,(x,y),1,(255,0,0),-1)
		        mouseX,mouseY = x,y
		        
		        cornerCoords[noOfInput][0] = mouseX
		        cornerCoords[noOfInput][1] = mouseY

		        noOfInput += 1
		#get ruser input of corners
		while(noOfInput < 4):
			cv2.imshow('userInput', wholeImage)
			cv2.setMouseCallback('userInput', draw_circle)
			print('Please select a corner of the pitch')
			cv2.waitKey(20)

		cornerCoords = cornerCoords.astype(int)
		firstTime = False

	

	grey = cv2.cvtColor(wholeImage, cv2.COLOR_BGR2GRAY)

	grey = np.float32(grey)
	

	dst = cv2.cornerHarris(grey, 2, 3, 0.04)


	ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
	dst = np.uint8(dst)

	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(grey, np.float32(centroids), (5,5), (-1, -1), criteria)


	cutDownCorners = []
	cutDownCornersCount = 0

	cornersLength = len(corners) 
	outputArraySize = (cornersLength, 2)
	countedCorners = np.zeros(outputArraySize)
	countedCornersCount = 0


	#Group the corners together within a hardcoded range
	for i in range(0, len(corners)):
		currentCorner = corners[i]
		relevantCorners = []
		relevantCount = 0
		sizeChange = True
		index = -1

		cornerX = corners[i][0].astype(float)
		cornerY = corners[i][1].astype(float)
		


		if([cornerX, cornerY] in countedCorners.tolist()):
			continue

		
		while sizeChange == True:
			currentRelevantSize = len(relevantCorners)
			if(len(relevantCorners) == 0):
				currentCorner = corners[i]
			elif((currentCorner[0] == relevantCorners[index][0]) and (currentCorner[1] == relevantCorners[index][1])):
				pass
				
			else:
				currentCorner = relevantCorners[index]

			for j in range(0, len(corners)):
				xDiff = currentCorner[0] - corners[j][0]
				yDiff = currentCorner[1] - corners[j][1]
				if((xDiff.astype(float) < 20.5) and (xDiff.astype(float) > -20.5)):
					if((yDiff.astype(float) < 20.5) and (yDiff.astype(float) > -20.5)):
						
						xValue = 0
						yValue = 0
						skip = False
						for x, sublist in enumerate(relevantCorners):
							if(corners[j][0] in sublist):
								xValue = x
							if(corners[j][1] in sublist):
								yValue = x
							if((xValue == yValue) and (xValue != 0) and (yValue != 0)):
								skip = True
								
						if(skip == True):
							continue
						

						relevantCorners.append([])
						relevantCorners[relevantCount].append(corners[j][0])
						relevantCorners[relevantCount].append(corners[j][1])
						relevantCount += 1
			
			if(len(relevantCorners) == currentRelevantSize):
				sizeChange = False
			else:
				index += 1
		
		#find the minimum Y value of the grouped corners
		minY = 100000
		for l in range(0, len(relevantCorners)):
			if(minY > relevantCorners[l][1]):
				minY = relevantCorners[l][1]
				xIndex = l
		
		cutDownCorners.append([])
		cutDownCorners[cutDownCornersCount].append(relevantCorners[xIndex])
		cutDownCornersCount += 1

	for i in range(0, len(cutDownCorners)):
		cutDownCorners[i] = cutDownCorners[i][0]
	np.set_printoptions(suppress = True)
	cutDownCorners = np.unique(cutDownCorners, axis = 0)

	finalCoords = np.zeros((4, 3))
	#Hard coded dimensions of the output plane
	finalCoords[0] = [0,0,1]
	finalCoords[3] = [110, 0, 1]
	finalCoords[2] = [110, 70, 1]
	finalCoords[1] = [0, 70, 1]

	retval, mask3 = cv2.findHomography(cornerCoords, finalCoords, cv2.RANSAC, 5.0)

	cutDownCorners = cutDownCorners.reshape(-1, 1, 2).astype(np.float32)
	finalCutDownCorners = cv2.perspectiveTransform(cutDownCorners, retval)

	goodCorners = []
	goodCornersCount = 0
	#Remove corners outside of a given range
	for i in range(0, len(finalCutDownCorners)):
		if((finalCutDownCorners[i][0][0] < 110.0) and (finalCutDownCorners[i][0][0] > 0.0)):
			if((finalCutDownCorners[i][0][1] < 70.0) and (finalCutDownCorners[i][0][1] > 0.0)):
				goodCorners.append([])
				goodCorners[goodCornersCount].append(finalCutDownCorners[i][0][0])
				goodCorners[goodCornersCount].append(finalCutDownCorners[i][0][1])
				goodCornersCount += 1

	im_dst = cv2.warpPerspective(wholeImage, retval, (110, 70))
	for i in range(0, len(goodCorners)):
		cv2.circle(im_dst, (int(goodCorners[i][0]), int(goodCorners[i][1])), 1, (0, i, 0), 2)

	frame = im_dst
	cv2.imshow('Output', frame)
	print('Frame done')
	frameCount += 1
	print(frameCount)
	#Stop the program after a set amount of frames, to make the generation of output quicker
	if(frameCount == 180):
		writer.release()
		vs.release()
		print('Done')
		cv2.waitKey(0)
		break
	
	#Initialise the writer object
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)


	
	print('written')
	writer.write(frame)

#Destroy the writer and video objects once the program has finished
writer.release()
vs.release()