import cv2
import numpy as np
import math


def solvePnP(imgPoints, objPoints, K, D):
	retval, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, K, D)
	R, J = cv2.Rodrigues(rvec)
	return R, tvec


def groundToNumberPlateTransformation(R_numberPlateToCamera, T_numberPlateToCamera, heightOfNumberPlateFromGround):
	T_groundToCamera = np.matmul(R_numberPlateToCamera, np.array([[0],[-heightOfNumberPlateFromGround],[0]])) + T_numberPlateToCamera
	xo = np.matmul(K, T_groundToCamera)
	R_groundToCamera = np.matmul(R_numberPlateToCamera, np.array([[1,0,0],[0,0,-1],[0,1,0]]))
	P_groundToCamera = np.concatenate((R_groundToCamera, T_groundToCamera), 1)
	return P_groundToCamera, R_groundToCamera, T_groundToCamera

def cameraToGround(P_groundToCamera, imgPoints, K):
	objPoints = []
	for [u,v,w] in imgPoints:
		[u1, v1, w1] = np.matmul(np.linalg.inv(K), [u,v,w])
		a = u1/w1 * P_groundToCamera[2][0] - P_groundToCamera[0][0]
		b = u1/w1 * P_groundToCamera[2][1] - P_groundToCamera[0][1]
		c = u1/w1 * P_groundToCamera[2][3] - P_groundToCamera[0][3]
		d = v1/w1 * P_groundToCamera[2][0] - P_groundToCamera[1][0]
		e = v1/w1 * P_groundToCamera[2][1] - P_groundToCamera[1][1]
		f = v1/w1 * P_groundToCamera[2][3] - P_groundToCamera[1][3]
		A = np.array([[a, b],[d, e]])
		B = np.array([-c, -f])
		x = np.linalg.solve(A, B) 
		objPoints.append(x)
	return objPoints

def originInImage(T_groundToCamera, K):
	X = np.matmul(K, T_groundToCamera)
	print(X[0]/X[2], X[1]/X[2])

imgPoints = np.array([(349,160),(368,160),(368,192),(349,192)], dtype=np.float32)
objPoints = np.array([(-35.,57.,0.),(35.,57.,0.),(35.,-57.,0.),(-35.,-57.,0.)], dtype=np.float32)
K = np.asarray([(624.77231988, 0., 305.88428381), (0., 624.47737775, 260.76879217), ( 0., 0., 1.)])
D = np.asarray([(9.04226563e-02, -1.32869626e-01, -7.61975496e-04, -1.40342881e-04, -2.68507190e-01)])
R_numberPlateToCamera, T_numberPlateToCamera = solvePnP(imgPoints, objPoints, K, D)

heightOfCameraFromGround = input("Enter height of camera from ground in mm :- ")
heightOfNumberPlateFromGround = heightOfCameraFromGround - T_numberPlateToCamera[1] 
P_groundToCamera, R_groundToCamera, T_groundToCamera = groundToNumberPlateTransformation(R_numberPlateToCamera, T_numberPlateToCamera, heightOfNumberPlateFromGround )

originInImage(T_groundToCamera, K)

imgPoints = np.array([[351, 336, 1], [448, 338, 1]])
objPoints = cameraToGround(P_groundToCamera, imgPoints, K)
print(objPoints)

