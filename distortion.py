import cv2
import numpy as np


def on_trackbar(s):
    pass

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Distortion_Factor", "Frame", 200, 1000, on_trackbar)


img = cv2.imread("/home/jaiama/Auto_Camera_Calibration/snap1.jpg")
h,w,c = img.shape
xs = list(range(w * 2)) * (h * 2)
ys = []
for i in range(h * 2):
    t = [i]
    y = t * (w * 2)
    ys.append(y)
ys = np.ravel(ys)
xs = np.array(xs)
ys = np.array(ys)
rx, ry = xs - w, ys - h
r_ = (rx ** 2 + ry ** 2) ** 0.5
dist = ((h ** 2) + (w ** 2)) ** 0.5

cv2.imshow("Frame", img)

while 1:
    factor = (cv2.getTrackbarPos("Distortion_Factor", "Frame") + 1) / 100.0
    r = r_ / (dist / factor)
    theta = np.zeros(w * h * 4)
    theta[np.where(r == 0)] = 1.0
    indices = np.where(r != 0)
    theta[indices] = np.arctan(r[indices]) / r[indices]
    sxs, sys = rx * theta + w / 2, ry * theta + h / 2
    a = np.reshape(sxs, (-1, w * 2)).astype(np.float32)
    b = np.reshape(sys, (-1, w * 2)).astype(np.float32)
    dst = cv2.remap(img, a, b, cv2.INTER_LINEAR)
    cv2.imshow("Frame", dst)
    if cv2.waitKey(1) == 27:
        break

cap = cv2.VideoCapture("/home/jaiama/Auto_Camera_Calibration/data_videos/vlc-record-2019-05-17-10h41m37s-GP030106.MP4-.mp4")
while 1:
    ret, frame = cap.read()
    dst = cv2.remap(frame, a, b, cv2.INTER_LINEAR)
    cv2.imshow("Frame", dst)
    if cv2.waitKey(1) == 27:
        break

