import cv2
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def get_intrinsic_matrix(points_in_2D):
	points = [[u1, v1],[u2, v2],[u3, v3]]
	a = np.array([[u1 * u2 + v1 * v2, u1 + u2, v1 + v2, 1],[u1 * u3 + v1 * v3, u1 + u3, v1 + v3, 1],[u3 * u2 + v3 * v2, u3 + u2, v3 + v2, 1]])
	[[w1],[w2],[w3],[w4]] = scipy.linalg.null_space(a)
	w = np.array([[w1, 0, w2], [0, w1, w3], [w2, w3, w4]])
	K = np.linalg.inv(np.transpose(np.linalg.cholesky(w)))
	K = K / K[2][2]
	print(K.astype(int))
	return K

def get_rotation_matrix(K, points_in_2D):
	points = [[u1, v1],[u2, v2],[u3, v3]]	
	K_inv = np.linalg.inv(K)	
	a = np.matmul(K_inv, np.array([[u1], [v1], [1]]))
	a = a / np.linalg.norm(a)
	b = np.matmul(K_inv, np.array([[u2], [v2], [1]]))
	b = b / np.linalg.norm(b)
	c = np.matmul(K_inv, np.array([[u3], [v3], [1]]))
	c = c / np.linalg.norm(c)
	rotation_matrix = np.concatenate((np.concatenate((a,b),1),c), 1)
	print(rotation_matrix)
	return rotation_matrix

def get_translation_matrix(K, rotation_matrix, h, w, H):
	x,y = w // 2, h // 2
	point = np.array([[x],[y],[1]])
	constraint1 = np.matmul(np.linalg.inv(K), point)
	tx_ = constraint1[0] / constraint1[2]
	ty_ = constraint1[1] / constraint1[2]
	tz = H / (rotation_matrix[0][2] * tx_ + rotation_matrix[1][2] * ty_ + rotation_matrix[2][2])
	translation_matrix = np.array([tx_ * tz , ty_ * tz , tz])
	print(translation_matrix)	
	return translation_matrix 

def decompose_projection_matrix(projection_matrix):
	Hinf = projection_matrix[:,:3]
	h = projection_matrix[:,-1]

	Xo = np.matmul(np.linalg.inv(Hinf), h)
	print(Xo)
	q, r = np.linalg.qr(np.linalg.inv(Hinf))
	R = np.transpose(q)
	print(R)	
	K = np.linalg.inv(r)
	print(K)
	T = np.matmul(R, X0)
	print(T)
	
	return Xo, R, K, T
	
#[u1,v1,u2,v2,u3,v3] = list(map(int, input().split()))
u1,v1,u2,v2,u3,v3 = -217,70,1806,31,427,4906
points = [[u1, v1],[u2, v2],[u3, v3]]
K = get_intrinsic_matrix(points)
R = get_rotation_matrix(K, points)
tvec = get_translation_matrix(K, R, 576, 720, 7420)
extrinisc_matrix = np.concatenate((R, tvec), 1)

projection_matrix = np.matmul(K, extrinisc_matrix)
X0, R, K, T = decompose_projection_matrix(projection_matrix)

