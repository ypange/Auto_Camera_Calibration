import cv2
import numpy as np
import matplotlib.pyplot as plt

def dist(x,y):
    return (x*x+y*y) ** 0.5

def correct_fisheye(src_size,dest_size,dx,dy,factor):
    """ returns a tuple of source coordinates (sx,sy)
        (note: values can be out of range)"""
    # convert dx,dy to relative coordinates
    rx, ry = dx-(dest_size[0]/2), dy-(dest_size[1]/2)
    # calc theta
    r = dist(rx,ry)/(dist(src_size[0],src_size[1])/factor)
    if 0==r:
        theta = 1.0
    else:
        theta = atan(r)/r
    # back to absolute coordinates
    sx, sy = (src_size[0]/2)+theta*rx, (src_size[1]/2)+theta*ry
    # done
    return (int(round(sx)),int(round(sy)))


src_size = (1920, 1080)
dest_size = (1920, 1080)
factor = 3

"""sxs = []
sys = []
for y in range(1080):
    for x in range(1920):
        rx, ry = x - (dest_size[0] / 2), y - (dest_size[1] / 2)
        r = dist(rx, ry) / (dist(src_size[0], src_size[1]) / factor)
        if 0 == r:
            theta = 1.0
        else:
            theta = np.arctan(r) / r
        sx, sy = (src_size[0] / 2) + theta * rx, (src_size[1] / 2) + theta * ry
        sxs.append(int(round(sx)))
        sys.append(int(round(sy)))
plt.scatter(sxs, sys)
plt.show()

"""
xs = list(range(1920)) * 1080
ys = []
for i in range(1080):
    t = [i]
    y = t * 1920
    ys.append(y)
ys = np.ravel(ys)
xs = np.array([xs])
ys = np.array([ys])

rx, ry = xs-960, ys-540
print(rx)
r = np.concatenate((rx, ry), 0)
r = np.linalg.norm(r, axis=0)
r = r / ((1080**2)+(1920**2)) ** 0.5 / factor
print(r)
theta = np.zeros(1920*1080)
theta[np.where(r == 0)] = 1.0

theta[np.where(r != 0)] = np.arctan(r[0]) / r[0]
sxs, sys = 960 + theta * rx, 540 + theta * ry

"""img = cv2.imread("/home/jaiama/Auto_Camera_Calibration/snap2.jpg", 0)
blank = np.zeros(img.shape, np.uint8)
for (sx,sy,x,y) in zip(sxs[0], sys[0], xs[0], ys[0]):
    if sx < 1920 and sy < 1080 and x < 1920 and y < 1080:
        blank[int(sy)][int(sx)] = img[y][x]

cv2.namedWindow("", cv2.WINDOW_NORMAL)
cv2.imshow("", blank)
cv2.waitKey(0)"""