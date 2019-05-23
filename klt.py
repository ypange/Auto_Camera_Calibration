import numpy as np
import cv2 as cv

cap = cv.VideoCapture("vlc-record-2019-05-17-10h41m37s-GP030106.MP4-.mp4")
cv.namedWindow("frame", cv.WINDOW_NORMAL)
sift = cv.xfeatures2d.SIFT_create()

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 10,
                       qualityLevel = 0.03,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))



# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

kp = sift.detect(old_gray,None)
p0 = (np.array(list(map(lambda p: [p.pt], kp))).astype(int)).astype(np.float32)
# Create some random colors
color = np.random.randint(0,255,(p0.shape[0],3))

mask = np.zeros_like(old_frame)

while(1):

    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    print(p1)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)

    img = cv.add(frame,mask)
    cv.imshow('frame',mask)

    k = cv.waitKey(1) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
