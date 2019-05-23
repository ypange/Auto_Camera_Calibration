import cv2
import numpy as np
import pickle as pkl
count = 0
sift = cv2.xfeatures2d.SIFT_create()
blank = np.ones((540 * 5, 960 * 5), np.uint8)
cv2.namedWindow("MAP", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture('/home/aman/2017_0627_145452_001.MOV')
lines = []
while(1):
    ret, frame = cap.read()
    if ret is not True:
        break

    frame = cv2.resize(frame, (960,540))

    if count == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp1, des1 = sift.detectAndCompute(gray, None)
    elif count % 6 == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(gray, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        for ([[x1, y1]], [[x2, y2]]) in zip(src_pts, dst_pts):
            if 10 < (np.absolute(x1 - x2) + np.absolute(y1 - y2)) < 100:
                cv2.circle(frame, (x1, y1), 3, (255, 0, 0), -1)
                cv2.circle(frame, (x2, y2), 3, (0, 255, 0), -1)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                if y1 != y2:
                    theta = np.arctan(-(x2 - x1) / (y2 - y1))
                else:
                    if (x1 - x2) > 0:
                        theta = -np.pi / 2
                    else:
                        theta = np.pi / 2
                rho = y1 * np.sin(theta) + x1 * np.cos(theta)
                lines.append([rho, theta])
                for i in range(540 * 5):
                    x = int(960*2 + rho / np.cos(theta) - (i - 540*2) * np.tan(theta))
                    if 0 <= x < 960 * 5:
                        blank[i][x] = blank[i][x] + 1

	count = 0
        kp1 = kp2
        des1 = des2
    if count == 0:
        cv2.imshow("MAP", blank)
        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("blank_.jpg", blank)
            with open('outfiles', 'wb') as fp:
                pkl.dump(lines, fp)
	    break
    count = count + 1
    if (np.max(blank) == 254):
        cv2.imwrite("blank_.jpg", blank)
        with open('outfiles', 'wb') as fp:
            pkl.dump(lines, fp)
	break
cv2.imwrite("blank_.jpg", blank)
with open('outfiles', 'wb') as fp:
    pkl.dump(lines, fp)
	

