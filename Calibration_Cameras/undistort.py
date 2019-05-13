import cv2
import numpy as np

DIM=(1920, 1080)
K=np.array([[1016.8568796410335, 0.0, 935.4467831575824], [0.0, 1028.1493391362328, 535.5004273222314], [0.0, 0.0, 1.0]])
D=np.array([[0.0007826377077388248], [-0.04180530068089383], [0.10629864103608097], [-0.0918388741870907]])

def undistort(img_path, balance=1.0, dim2=None, dim3=None):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("undistorted", cv2.WINDOW_NORMAL)
    cv2.imshow("original", img)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import os

file_names = os.listdir("/home/aman/Auto_Camera_Calibration/Calibration_Cameras")
for file_name in file_names:
	if file_name[-4:] == '.png':
        	undistort(file_name)
