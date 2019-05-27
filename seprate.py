import cv2
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

file_name = '/home/aman/Auto_Camera_Calibration/uvlc-record-2019-05-21-12h50m30s-2017_0627_145452_001'
with open(file_name, 'rb') as fp:
	a = np.array(pkl.load(fp))

blank = cv2.imread(file_name+'.jpg', 0)
blank = 255 - blank

rhos, thetas = a[:,0], a[:,1]

degrees = thetas * 180 / np.pi
bins, z, _ = plt.hist(degrees, bins = 18, range=(-90,90))

[p,q,r,s] = bins.argsort()[::-1][:4]
range1 = None
range2 = None

	
if (np.absolute(p-q) < 2 and (bins[p] - bins[q]) / bins[p] < 0.1):
	range1 = (z[p] + z[q]) / 2.0  , (z[p] + z[q]) / 2.0  + 10
	if (np.absolute(r-s) < 2 and (bins[r] - bins[s]) / bins[r] < 0.1):
		range2 = (z[r] + z[s]) / 2.0 , (z[r] + z[s]) / 2.0 + 10

	else:
		if ((r == 0 and s == 17)  or (r == 17 and s == 0)):
			range2a = (-85,-90)
			range2b = (85, 90)
		else:	
			range2 = z[r], z[r+1]
else:
	if ((p == 0 and q == 17)  or (p == 17 and q == 0)):
		range1a = (-85,-90)
		range1b = (85, 90)
		if (np.absolute(r-s) < 2 and (bins[r] - bins[s]) / bins[r] < 0.1):
			range2 = (z[r] + z[s]) / 2.0 , (z[r] + z[s]) / 2.0 + 10			
		else:
			range2 = z[r], z[r+1]	
	else:
		range1 = z[p], z[p+1]
		if (np.absolute(q-r) < 2 and (bins[q] - bins[r]) / bins[q] < 0.1):
			range2 = (z[q] + z[r]) / 2.0 , (z[q] + z[r]) / 2.0 + 10			
		else:
			range2 = z[q], z[q+1]	

blank1 = np.zeros(blank.shape, np.uint8)
blank2 = np.zeros(blank.shape, np.uint8)
if (range1 is not None):
	print(range1)		
	if range2 is not None:
		print(range2)
	else:
		print(range2a)
		print(range2b)
else:
	print(range1a)
	print(range1b)
	if range2 is not None:
		print(range2)
	else:
		print(range2a)
		print(range2b)

h,w = blank.shape
h = int(h / 5)
w = int(w / 5)
for rho, theta in zip(rhos, thetas):
	if (range1 is not None):
		if (range1[0] <= theta * 180 / np.pi <= range1[1]):
			for i in range(h * 5):
				x = int(w*2 + rho/np.cos(theta) + (h*2 - i)*np.tan(theta))
				if 0 <= x < w * 5:
					blank1[i][x] = blank1[i][x] + 2		
	else:
		if (range1a[0] <= theta * 180 / np.pi <= range1a[1]) or (range1b[0] <= theta * 180 / np.pi <= range1b[1]):
				for i in range(h * 5):
					x = int(w*2 + rho/np.cos(theta) + (h*2 - i)*np.tan(theta))
					if 0 <= x < w * 5:
						blank1[i][x] = blank1[i][x] + 2		
	if range2 is not None:
		if (range2[0] <= theta * 180 / np.pi <= range2[1]):
			for i in range(h * 5):
				x = int(w*2 + rho/np.cos(theta) + (h*2 - i)*np.tan(theta))
				if 0 <= x < w * 5:
					blank2[i][x] = blank2[i][x] + 2		
	else:
		if (range2a[0] <= theta * 180 / np.pi <= range2a[1]) or (range2b[0] <= theta * 180 / np.pi <= range2b[1]):
			for i in range(h * 5):
				x = int(w*2 + rho/np.cos(theta) + (h*2 - i)*np.tan(theta))
				if 0 <= x < w * 5:
					blank2[i][x] = blank2[i][x] + 2		

cv2.imwrite("blank1.jpg", blank1)
cv2.imwrite("blank2.jpg", blank2)
cv2.imwrite("blank.jpg", blank)
