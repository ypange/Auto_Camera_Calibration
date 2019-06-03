import cv2
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import time
from skimage.draw import line_aa

file_name = 'outfiles/200vlc-record-2019-05-21-12h50m30s-2017_0627_145452_001.MOV-.avi'
with open(file_name, 'rb') as fp:
    array = np.array(pkl.load(fp))
rhos, thetas = array[:, 0], array[:, 1]
bins, z, _ = plt.hist(thetas*180/np.pi, bins=180, range=(-90, 90))
data = bins.argsort()[::-1]
for i in range(1, len(data)):
    if 170 <= data[0] <= 179:
        if np.absolute(data[0] - data[i]) > 10 and data[0] - 170 < data[i]:
            p = data[0] - 90
            q = data[i] - 90
            print(p, q)
            indices1a = np.where(np.absolute(thetas * 180 / np.pi - p) <= 5)
            indices1b = np.where(np.absolute(thetas[np.where(alpha >= 0 )] * 180 / np.pi - p) >= 85)
            indices2 = np.where(np.absolute(thetas * 180 / np.pi - q) <= 5)
            indices1 = np.concatenate((indices1a, indices1b), axis=1)[0]
            break
    elif 0 <= data[0] <= 4:
        if np.absolute(data[0] - data[i]) > 10 and 170 - data[0] >= data[i]:
            p = data[0] - 90
            q = data[i] - 90
            print(p, q)
            indices1a = np.where(np.absolute(thetas * 180 / np.pi - p) <= 5)
            indices1b = np.where(np.absolute(thetas[np.where(alpha >= 0)] * 180 / np.pi - p) >= 85)
            indices2 = np.where(np.absolute(thetas * 180 / np.pi - q) <= 5)
            indices1 = np.concatenate((indices1a, indices1b), axis=1)[0]
            break
    else:
        if np.absolute(data[0] - data[i]) > 10:
            p = data[0] - 90
            q = data[i] - 90
            print(p, q)
            indices1 = np.where(np.absolute(thetas * 180 / np.pi - p) <= 5)
            indices2 = np.where(np.absolute(thetas * 180 / np.pi - q) <= 5)
            break

canvas1 = np.zeros((1080*5, 1920*5), np.int)
canvas2 = np.zeros((1080*5, 1920*5), np.int)

cv2.namedWindow("1", cv2.WINDOW_NORMAL)
cv2.namedWindow("2", cv2.WINDOW_NORMAL)
start_time = time.time()

h, w = 1080. * 5, 1920. * 5

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
    canvas1[rr, cc] = canvas1[rr, cc] + 255

dcanvas1 = cv2.convertScaleAbs(canvas1)
cv2.imshow("1", dcanvas1)

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
    canvas2[rr, cc] = canvas2[rr, cc] + 1

dcanvas2 = cv2.convertScaleAbs(canvas2)
cv2.imshow("2", dcanvas2)
cv2.waitKey(0)

alpha = np.max(canvas1) * 0.10
# if np.max(canvas1) * 0.10 > 20:
#     alpha = 20
y1, x1 = np.where(canvas1 >= np.max(canvas1) - alpha)
# y1, x1 = np.mean(y1), np.mean(x1)

alpha = np.max(canvas2) * 0.10
# if np.max(canvas2) * 0.10 > 20:
#     alpha = 20
y2, x2 = np.where(canvas2 >= np.max(canvas2) - alpha)
# y2, x2 = np.mean(y2), np.mean(x2)


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
"""
14 139 140 138 141 142  13 137 143 136 144 135 145  15 134 147  12 146
 148 133  17 149  16  11 150  18 151 132  10 152 154  19 153   9 155 159
 157 158 160 131 162  20   8 161 156 164   5 165 163   3 167   7 168 166
 174   0 172 170  22 169 178  21 177   6 176   4 173 175 171   2   1  24
  26  23  25 130  27 129  35  28  36  31 128  29  33  30  32  37  34 127
  38  45  39  81  70 123 121  52 118  44  43  40  73  42  71 126  51  41
 124  57 122 116  63 120  59 125 111 112 117  49  54  50  58  93  55 105
 104  56 114 115  76 100 113  78  46  48  91  62  82  75  92 110  47  72
 119  74 102  60  77  90  99 106  94  69  68  67  66  65  64  86  61  53
  85  89  79  83  87  88  95  98 101 107 109  80  96  97 103 108 179  84]
"""