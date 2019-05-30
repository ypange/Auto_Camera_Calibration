import cv2
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import time
from skimage.draw import line_aa

file_name = 'outfiles/outfile0'
with open(file_name, 'rb') as fp:
    a = np.array(pkl.load(fp))
rhos, thetas = a[:, 0], a[:, 1]

# degrees = thetas * 180 / np.pi
# bins, z, _ = plt.hist(degrees, bins=180, range=(-90, 90))
# peaks, _ = find_peaks(bins, height=1000, distance = 20)

# [p, q, r, s] = bins.argsort()[::-1][:4]
# print(bins.argsort()[::-1])
# print(bins[p], bins[q], bins[r], bins[s])
# plt.show()
canvas = np.zeros((1080*5,1920*5), np.int)
cv2.namedWindow("", cv2.WINDOW_NORMAL)
start_time = time.time()

h, w = 1080. * 5, 1920. * 5

for rho, theta in zip(rhos, thetas):
    a = np.cos(theta)
    b = np.sin(theta)
    rho = rho + h / 5 * b + w / 5 * a
    if b == 0:
        x1, x2 = rho / a, (rho - h * b) / a
        y1, y2 = rho, rho
    elif a == 0:
        y1, y2 = rho / b, (rho - w * a) / b
        x1, x2 = rho, rho
    else:
        x1, x2, y1, y2 = rho / a, (rho - h * b) / a, rho / b, (rho - w * a) / b

    cordinates = []
    if 0 <= x1 <= w:
        cordinates.append((int(x1),0))
    if 0 <= x2 <= w:
        cordinates.append((int(x1), int(h)))
    if 0 <= y1 <= h:
        cordinates.append((0, int(y1)))
    if 0 <= y2 <= h:
        cordinates.append((int(w), (int(y2))))
    [(x1, y1), (x2, y2)] = cordinates
    rr, cc, val = line_aa(x1, y1, x2, y2)
    canvas[rr, cc] = canvas[rr, cc] + 1
# cv2.imwrite("canvas.jpg", canvas)
print("--- %s seconds ---" % (time.time() - start_time))
canvas = cv2.convertScaleAbs(canvas)
cv2.imshow("", canvas)
cv2.waitKey(0)

"""range1 = None
range2 = None

if np.absolute(p - q) < 2 and (bins[p] - bins[q]) / bins[p] < 0.2:
    range1 = (z[p] + z[q]) / 2.0, (z[p] + z[q]) / 2.0 + 10
    if np.absolute(r - s) < 2 and (bins[r] - bins[s]) / bins[r] < 0.2:
        range2 = (z[r] + z[s]) / 2.0, (z[r] + z[s]) / 2.0 + 10

    else:
        if (r == 0 and s == 17) or (r == 17 and s == 0):
            range2a = (-85, -90)
            range2b = (85, 90)
        else:
            range2 = z[r], z[r + 1]
else:
    if (p == 0 and q == 17) or (p == 17 and q == 0):
        range1a = (-85, -90)
        range1b = (85, 90)
        if np.absolute(r - s) < 2 and (bins[r] - bins[s]) / bins[r] < 0.2:
            range2 = (z[r] + z[s]) / 2.0, (z[r] + z[s]) / 2.0 + 10
        else:
            range2 = z[r], z[r + 1]
    else:
        range1 = z[p], z[p + 1]
        if np.absolute(q - r) < 2 and (bins[q] - bins[r]) / bins[q] < 0.2:
            range2 = (z[q] + z[r]) / 2.0, (z[q] + z[r]) / 2.0 + 10
        else:
            range2 = z[q], z[q + 1]

canvas1 = np.zeros(blank.shape, np.int)
canvas2 = np.zeros(blank.shape, np.int)
if range1 is not None:
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

h, w = blank.shape
h = int(h / 5)
w = int(w / 5)
for rho, theta in zip(rhos, thetas):
    if range1 is not None:
        if range1[0] <= theta * 180 / np.pi <= range1[1]:
            temp = np.zeros(canvas1.shape, np.uint8)
            x1 = int(w * 2 + rho / np.cos(theta) - (h * 0 - h * 2) * np.tan(theta))
            x2 = int(w * 2 + rho / np.cos(theta) - (h * 5 - h * 2) * np.tan(theta))
            cv2.line(temp, (x1, 0), (x2, h * 5), 1, 1)
            canvas1 = temp + canvas1
    else:
        if (range1a[0] <= theta * 180 / np.pi <= range1a[1]) or (range1b[0] <= theta * 180 / np.pi <= range1b[1]):
            temp = np.zeros(canvas1.shape, np.uint8)
            x1 = int(w * 2 + rho / np.cos(theta) - (h * 0 - h * 2) * np.tan(theta))
            x2 = int(w * 2 + rho / np.cos(theta) - (h * 5 - h * 2) * np.tan(theta))
            cv2.line(temp, (x1, 0), (x2, h * 5), 1, 1)
            canvas1 = temp + canvas1

    if range2 is not None:
        if range2[0] <= theta * 180 / np.pi <= range2[1]:
            temp = np.zeros(canvas2.shape, np.uint8)
            x1 = int(w * 2 + rho / np.cos(theta) - (h * 0 - h * 2) * np.tan(theta))
            x2 = int(w * 2 + rho / np.cos(theta) - (h * 5 - h * 2) * np.tan(theta))
            cv2.line(temp, (x1, 0), (x2, h * 5), 1, 1)
            canvas2 = temp + canvas2
    else:
        if (range2a[0] <= theta * 180 / np.pi <= range2a[1]) or (range2b[0] <= theta * 180 / np.pi <= range2b[1]):
            temp = np.zeros(canvas2.shape, np.uint8)
            x1 = int(w * 2 + rho / np.cos(theta) - (h * 0 - h * 2) * np.tan(theta))
            x2 = int(w * 2 + rho / np.cos(theta) - (h * 5 - h * 2) * np.tan(theta))
            cv2.line(temp, (x1, 0), (x2, h * 5), 1, 1)
            canvas2 = temp + canvas2

cv2.imwrite("vlc-record-2019-05-17-10h41m37s-GP0301061.jpg", canvas1)
cv2.imwrite("vlc-record-2019-05-17-10h41m37s-GP0301062.jpg", canvas2)
canvas = canvas1 + canvas2
plt.hist(canvas.ravel())
plt.show()
cv2.imwrite("vlc-record-2019-05-17-10h41m37s-GP030106_.jpg", canvas)
"""