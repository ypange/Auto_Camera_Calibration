import cv2
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from skimage.draw import line_aa

def computeDistortionCoefficients(mapx, mapy, K):
    x_points = (np.random.rand(100) * K[0][2] * 2).astype(int)
    y_points = (np.random.rand(100) * K[1][2] * 2).astype(int)
    x_distorted = mapx[y_points, x_points] - K[0][2]
    y_distorted = mapy[y_points, x_points] - K[1][2]
    x_undistorted = x_points - K[0][2]
    y_undistorted = y_points - K[1][2]
    r2 = (x_undistorted ** 2 + y_undistorted ** 2) / (K[0][0] ** 2)
    r4 = r2 ** 2
    r6 = r2 ** 3
    B = x_distorted / x_undistorted - 1
    A = np.array([r2, r4, r6]).transpose()
    k1,k2,k3 = np.linalg.lstsq(A,B)[0]
    return np.array([k1,k2,0.,0.,k3])


def computeVanishinPoint(xs, ys, h, w, factor):
    rx, ry = xs - w / 2, ys - h / 2
    r_ = (rx ** 2 + ry ** 2) ** 0.5
    dist = ((h ** 2) + (w ** 2)) ** 0.5
    r = r_ / (dist / ((factor + 1) / 100.))
    if r == 0:
        theta = 1.0
    else:
        theta = np.arctan(r) / r
    vx, vy = rx * theta + w / 2, ry * theta + h / 2
    return vx, vy


def orthoProjection(P_groundToCamera, K, img):
    H = np.zeros((3, 3), np.float32)
    H[:, :2] = P_groundToCamera[:, :2]
    H[:, 2] = P_groundToCamera[:, 3]
    H = np.matmul(K, H)
    H = np.linalg.inv(H)
    M = np.float32([[1, 0, 75000], [0, 1, 75000], [0, 0, 1]])
    M = np.matmul(M, H)
    dst = cv2.warpPerspective(img, M, (100000, 100000))
    dst = cv2.resize(dst, (20000, 20000))
    cv2.namedWindow("", cv2.WINDOW_NORMAL)
    #plt.imshow(dst)
    cv2.imshow("", dst)
    cv2.imwrite("ortho.jpg", dst)
    cv2.imwrite("original.jpg", img)
    cv2.waitKey(0)


def computeProjection(H, K, u1, v1, u2, v2):
    Kinv = np.linalg.inv(K)
    cx, cy = K[0][2], K[1][2]
    r1 = np.matmul(Kinv, np.array([[u1], [v1], [1.]])) / np.linalg.norm(np.matmul(Kinv, np.array([[u1], [v1], [1.]])))
    r2 = np.matmul(Kinv, np.array([[u2], [v2], [1.]])) / np.linalg.norm(np.matmul(Kinv, np.array([[u2], [v2], [1.]])))
    if (u1 - cx) < 0:
        r1 = -r1
    if (v2 - cy) > 0 :
        r2 = -r2
    r3 = np.cross(r1.transpose(), r2.transpose()).transpose() / (np.linalg.norm(np.cross(r1.transpose(), r2.transpose()).transpose()))
    R = np.concatenate((r1,r2), axis = 1)
    R = np.concatenate((R, r3), axis = 1)
    tz = - H / R[2][2]
    T = np.array([[0.],[0.],[tz]])
    P = np.concatenate((R,T), axis = 1)
    return P

file_name = "vlc-record-2019-05-24-10h20m44s-2017_0705_151023_001.MP4-.mp4"
factor = 150
outfile_name = 'outfiles/' + str(factor) + file_name

cap = cv2.VideoCapture("data_videos/" + file_name)
ret, frame = cap.read()
h, w, c = frame.shape
xs = list(range(w )) * (h )
ys = []
for i in range(h ):
    t = [i]
    y = t * (w )
    ys.append(y)
ys = np.ravel(ys)
xs = np.array(xs)
ys = np.array(ys)
rx, ry = xs - w / 2, ys - h / 2
r_ = (rx ** 2 + ry ** 2) ** 0.5
dist = ((h ** 2) + (w ** 2)) ** 0.5
r = r_ / (dist / ((factor + 1) / 100.))
theta = np.zeros(w * h )
theta[np.where(r == 0)] = 1.0
indices = np.where(r != 0)
theta[indices] = np.arctan(r[indices]) / r[indices]
sxs, sys = rx * theta + w / 2, ry * theta + h / 2
mapx = np.reshape(sxs, (-1, w )).astype(np.float32)
mapy = np.reshape(sys, (-1, w )).astype(np.float32)
dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

with open(outfile_name, 'rb') as fp:
    array = np.array(pkl.load(fp))

rhos, thetas = array[:, 0], array[:, 1]

bins, z, _ = plt.hist(thetas * 180 / np.pi, bins=180, range=(-90, 90))
data = bins.argsort()[::-1]
for i in range(1, len(data)):
    if 170 <= data[0] <= 179:
        if np.absolute(data[0] - data[i]) > 25 and data[0] - 165 < data[i]:
            p = data[0] - 90
            q = data[i] - 90
            print(p, q)
            indices1a = np.where(np.absolute(thetas * 180 / np.pi - p) <= 5)
            indices1b = np.where(np.absolute(thetas[np.where(thetas >= 0)] * 180 / np.pi - p) >= 85)
            indices2 = np.where(np.absolute(thetas * 180 / np.pi - q) <= 5)
            indices1 = np.concatenate((indices1a, indices1b), axis=1)[0]
            break
    elif 0 <= data[0] <= 4:
        if np.absolute(data[0] - data[i]) > 25 and 165 - data[0] >= data[i]:
            p = data[0] - 90
            q = data[i] - 90
            print(p, q)
            indices1a = np.where(np.absolute(thetas * 180 / np.pi - p) <= 5)
            indices1b = np.where(np.absolute(thetas[np.where(thetas >= 0)] * 180 / np.pi - p) >= 85)
            indices2 = np.where(np.absolute(thetas * 180 / np.pi - q) <= 5)
            indices1 = np.concatenate((indices1a, indices1b), axis=1)[0]
            break
    else:
        if np.absolute(data[0] - data[i]) > 25:
            p = data[0] - 90
            q = data[i] - 90
            print(p, q)
            indices1 = np.where(np.absolute(thetas * 180 / np.pi - p) <= 5)
            indices2 = np.where(np.absolute(thetas * 180 / np.pi - q) <= 5)
            break

canvas3 = np.zeros((h * 5, w * 5), np.int)
canvas4 = np.zeros((h * 5, w * 5), np.int)
h, w = h * 5, w * 5
rhos = array[indices1][:, 0]
thetas = array[indices1][:, 1]

for rho, theta in zip(rhos, thetas):
    a = np.cos(theta)
    b = np.sin(theta)
    rho = rho + (2. * h / 5) * b + (2. * w / 5) * a
    if b == 0:
        x1, x2 = rho / a, (rho - h * b) / a
        y1, y2 = rho, rho
    elif a == 0:
        y1, y2 = rho / b, (rho - w * a) / b
        x1, x2 = rho, rho
    else:
        x1, x2, y1, y2 = rho / a, (rho - h * b) / a, rho / b, (rho - w * a) / b

    cordinates = []
    if 0 <= x1 < w:
        cordinates.append((int(x1), 0))
    if 0 <= x2 < w:
        cordinates.append((int(x2), int(h - 1)))
    if 0 <= y1 < h:
        cordinates.append((0, int(y1)))
    if 0 <= y2 < h:
        cordinates.append((int(w - 1), (int(y2))))
    [(x1, y1), (x2, y2)] = cordinates
    rr, cc, val = line_aa(y1, x1, y2, x2)
    canvas3[rr, cc] = canvas3[rr, cc] + 255

rhos = array[indices2][:, 0]
thetas = array[indices2][:, 1]

for rho, theta in zip(rhos, thetas):
    a = np.cos(theta)
    b = np.sin(theta)
    rho = rho + (2. * h / 5) * b + (2. * w / 5) * a
    if b == 0:
        x1, x2 = rho / a, (rho - h * b) / a
        y1, y2 = rho, rho
    elif a == 0:
        y1, y2 = rho / b, (rho - w * a) / b
        x1, x2 = rho, rho
    else:
        x1, x2, y1, y2 = rho / a, (rho - h * b) / a, rho / b, (rho - w * a) / b

    cordinates = []
    if 0 <= x1 < w:
        cordinates.append((int(x1), 0))
    if 0 <= x2 < w:
        cordinates.append((int(x2), int(h - 1)))
    if 0 <= y1 < h:
        cordinates.append((0, int(y1)))
    if 0 <= y2 < h:
        cordinates.append((int(w - 1), (int(y2))))
    [(x1, y1), (x2, y2)] = cordinates
    rr, cc, val = line_aa(y1, x1, y2, x2)
    canvas4[rr, cc] = canvas4[rr, cc] + 1


alpha = np.max(canvas3) * 0.20
y1, x1 = np.where(canvas3 >= np.max(canvas3) - alpha)
alpha = np.max(canvas4) * 0.20
y2, x2 = np.where(canvas4 >= np.max(canvas4) - alpha)
cx, cy = w / 5 / 2, h / 5 / 2
u1m, u2m, v1m, v2m = np.sum(canvas3[y1,x1] * x1) / np.sum(canvas3[y1,x1]) - 2 * w / 5 - cx, np.sum(canvas4[y2,x2] * x2) / np.sum(canvas4[y2,x2]) - 2 * w / 5 - cx, np.sum(canvas3[y1,x1] * y1) / np.sum(canvas3[y1,x1]) - 2 * h / 5 - cy, np.sum(canvas4[y2,x2] * y2) / np.sum(canvas4[y2,x2]) - 2 * h / 5 - cy
plt.figure(1)
plt.imshow(canvas3)
plt.scatter(u1m - (- 2 * w / 5 - cx),v1m - (- 2 * h / 5 - cy),c="red")

plt.figure(2)
plt.imshow(canvas4)
plt.scatter(u2m - (- 2 * w / 5 - cx),v2m - (- 2 * h / 5 - cy),c="red")

if abs(u1m) > abs(u2m) :
    if u1m < 0:
        u1 = u1m - 2 * np.std(x1) + cx
    else:
        u1 = u1m + 2 * np.std(x1) + cx

    if canvas3[int(v1m - 2 * np.std(y1) + cy + h / 5 * 2)][int(u1 + w / 5 * 2)] > canvas3[int(v1m + 2 * np.std(y1) + cy + h / 5 * 2)][int(u1 + w / 5 * 2)]:
        v1 = v1m - 2 * np.std(y1) + cy
    else:
        v1 = v1m + 2 * np.std(y1) + cy

    if v2m < 0:
        v2 = v2m - 2 * np.std(y2) + cy
    else:
        v2 = v2m + 2 * np.std(y2) + cy

    if canvas4[int(v2 + h / 5 * 2)][int(u2m + 2 * np.std(x2) + cx + w / 5 * 2)] > canvas4[int(v2 + h / 5 * 2)][int(u2m - 2 * np.std(x2) + cx + w / 5 * 2)]:
        u2 = u2m + 2 * np.std(x2) + cx
    else:
        u2 = u2m - 2 * np.std(x2) + cx

else:
    if v1m < 0:
        v2 = v1m - 2 * np.std(y1) + cy
    else:
        v2 = v1m + 2 * np.std(y1) + cy
    if canvas3[int(v2 + h / 5 * 2)][int(u1m - 2 * np.std(x1) + cx + w / 5 * 2)] > canvas3[int(v2 + h / 5 * 2)][int(u1m + 2 * np.std(x1) + cx + w / 5* 2)]:
        u2 = u1m - 2 * np.std(x1) + cx
    else:
        u2 = u1m + 2 * np.std(x1) + cx

    if u2m < 0:
        u1 = u2m - 2 * np.std(x2) + cx
    else:
        u1 = u2m + 2 * np.std(x2) + cx

    if canvas4[int(v2m - 2 * np.std(y2) + cy + h / 5 * 2)][int(u1 + w / 5 * 2)] > canvas4[int(v2m + 2 * np.std(y2) + cy + h / 5 * 2)][int(u1 + w / 5 * 2)]:
        v1 = v2m - 2 * np.std(y2) + cy
    else:
        v1 = v2m + 2 * np.std(y2) + cy

f = ((u1 - cx) * (cx - u2) + (v1 - cy) * (cy - v2)) ** 0.5
K = np.array([[f, 0., cx], [0., f, cy], [0., 0., 1.]])
P = computeProjection(9000, K, u1, v1, u2, v2)
orthoProjection(P, K, dst)
print(K)
(u1,v1), (u2,v2) = computeVanishinPoint(u1,v1,h/5,w/5, factor), computeVanishinPoint(u2,v2,h/5,w/5, factor)
f = ((u1 - cx) * (cx - u2) + (v1 - cy) * (cy - v2)) ** 0.5
K2 = np.array([[f, 0., cx], [0., f, cy], [0., 0., 1.]])
P = computeProjection(9000, K2, u1, v1, u2, v2)
orthoProjection(P, K, frame)
print(K2)

"""
f70512, f30106, f145452
Out[121]: (495.61894113116557, 442.90798432615765, 398.3028604130947)
"""
