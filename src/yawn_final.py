from turtle import shape
import numpy as np
import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
#to utilize the webcam
cam = cv2.VideoCapture(0)
yawn_thresh = 30

class Yawning:
	def calculate_yawning(shape):
		upper_lip = shape[50:53]
		upper_lip = np.concatenate((upper_lip, shape[61:64]))

		lower_lip = shape[56:59]
		lower_lip = np.concatenate((lower_lip, shape[65:68]))

		upper_lip_mean = np.mean(upper_lip, axis=0)
		lower_lip_mean = np.mean(lower_lip, axis=0)
		#calculating distance between lower & upper lip
		distance = dist.euclidean(upper_lip_mean,lower_lip_mean)
		return distance
	
	def detect_yawn(face_model,landmark_model):
		while True :
			suc,frame = cam.read()
			if not suc :
				break
			#detecting face & converting it into grayscale
			img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			faces = face_model(img_gray)
			for face in faces:
				x1 = face.left()
				y1 = face.top()
				x2 = face.right()
				y2 = face.bottom()
				cv2.rectangle(frame,(x1,y1),(x2,y2),(200,0,00),2)
				#detecting Landmarks points
				shapes = landmark_model(img_gray,face)
				shape = face_utils.shape_to_np(shapes)
				#marking the lower and upper lip
				lip = shape[48:60]
				cv2.drawContours(frame,[lip],-1,(0, 165, 255),thickness=3)
				#calculating the lip distance
				obj = Yawning
				distance_between_lips = obj.calculate_yawning(shape)
				if distance_between_lips > yawn_thresh :
					cv2.putText(frame, f'DRIVER IS YAWNING',(frame.shape[1]//2 - 170 ,frame.shape[0]//2),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,200),4)
			cv2.imshow('Webcam',frame)
			#to quit the window
			if cv2.waitKey(1) & 0xFF == ord('q') :
				break
		cam.release()
		cv2.destroyAllWindows()

if __name__=="__main__":
	obj = Yawning
	#Returns the default face detector
	face_model = dlib.get_frontal_face_detector()
	#taking an image of a human face as input and identifying the locations of important facial landmarks
	landmark_model = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
	obj.detect_yawn(face_model,landmark_model)


"""
References:
1. http://dlib.net/python/index.html#dlib.get_frontal_face_detector
2. http://dlib.net/python/index.html#dlib.shape_predictor
"""