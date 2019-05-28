from __future__ import division

from util import *
from darknet import Darknet
from preprocess import letterbox_image

import pickle as pkl
import argparse
import time
start_time = time.time()

# K = np.array([[584.95, 0.00, 635.12],[0.22, 584.42, 367.63], [0.00, 0.00, 1.00]])
# D = np.array([-0.26117408, 0.09197799, 0,0,0])
# K = np.array([[ 894.18, 0.0,    951.75], [0.0,  913.20, 573.04], [0., 0., 1.]])
# D = np.array([-0.35887,  0.16375, 0, 0, 0])


def undistort(img):
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (1920, 1080), 1)
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, (1920, 1080), 5)
    image = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    return image


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    x = x.cpu().numpy()
    c1 = tuple(x[1:3])
    c2 = tuple(x[3:5])
    cls = int(x[-1])
    classes = load_classes('data/coco.names')
    label = "{0}".format(classes[cls])
    objects = []
    if label in ['car','bus','truck', 'bicycle', 'motorbike' ]:
        objects.append(['car', (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1]))])
    else:
        objects.append(None)
    return objects


def arg_parse():
    """
    Parse arguements to the detect module

    """


    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')

    parser.add_argument("--video", dest = 'video', help =
                        "Video to run detection upon",
                        default = "video.avi", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.7)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help =
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help =
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    return parser.parse_args()


args = arg_parse()
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
CUDA = torch.cuda.is_available()


num_classes = 80

CUDA = torch.cuda.is_available()

bbox_attrs = 5 + num_classes
colors = pkl.load(open("pallete", "rb"))

print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

if CUDA:
    model.cuda()

model(get_test_input(inp_dim, CUDA), CUDA)

model.eval()


cv2.namedWindow(" ", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture("data_videos/vlc-record-2019-05-21-12h50m30s-2017_0627_145452_001.MOV-.avi")
ret, frame = cap.read()
h,w,c = frame.shape

count = 0
sift = cv2.xfeatures2d.SIFT_create()
canvas = np.zeros((h * 5, w * 5), np.uint8)
lines = []
while(1):
    ret, frame = cap.read()

    if ret is not True:
        break
    # frame = undistort(frame)
    foot_print = np.zeros(frame.shape[:2], np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if count == 0:

        img, orig_im, dim = prep_image(frame, inp_dim)
        im_dim = torch.FloatTensor(dim).repeat(1, 2)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = model(Variable(img), CUDA)

        output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        objects = list(map(lambda x: write(x, orig_im), output))
        for object in objects:
            if object[0] is not None:
                [[classifier, (x1, y1), (x2, y2)]] = object

                if classifier is 'car':
                    cv2.rectangle(foot_print, (max(x1 - 5, 0), max(y1 - 5, 0)),
                                  (min(x2 + 5, foot_print.shape[-1]), min(y2 + 5, foot_print.shape[0])), 255, -1)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        kp1, des1 = sift.detectAndCompute(gray, mask=foot_print)

    elif count % 3 == 0:

        img, orig_im, dim = prep_image(frame, inp_dim)
        im_dim = torch.FloatTensor(dim).repeat(1, 2)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = model(Variable(img), CUDA)

        output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        objects = list(map(lambda x: write(x, orig_im), output))

        for object in objects:
            if object[0] is not None:
                [[classifier, (x1, y1), (x2, y2)]] = object

                if classifier is 'car':
                    cv2.rectangle(foot_print, (max(x1 - 5, 0), max(y1 - 5, 0)),
                                  (min(x2 + 5, foot_print.shape[-1]), min(y2 + 5, foot_print.shape[0])), 255, -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        kp2, des2 = sift.detectAndCompute(gray, mask=foot_print)
        # print(len(des1), len(des2))
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        # FLANN_INDEX_KDTREE = 0
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # search_params = dict(checks=50)  # or pass empty dictionary
        #
        # flann = cv2.FlannBasedMatcher(index_params, search_params)
        #
        # matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.80 * n.distance:
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

                """temp = np.zeros(canvas.shape, np.uint8)
                x1 = np.int64(w * 2 + rho / np.cos(theta) - (h * 0 - h * 2) * np.tan(theta))
                x2 = np.int64(w * 2 + rho / np.cos(theta) - (h * 5 - h * 2) * np.tan(theta))
                cv2.line(temp, (x1,0), (x2, h * 5), 1, 1)
                canvas = temp + canvas
        cv2.imshow(" ", canvas)
        cv2.imwrite("fvlc-record-2019-05-21-12h50m30s-2017_0627_145452_001.jpg", canvas)"""
        count = 0
        kp1 = kp2
        des1 = des2

    if count == 0:
        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    count = count + 1

# When everything done, release the capture

with open('outfiles/outfile0', 'wb') as fp:
    pkl.dump(lines, fp)

cap.release()
cv2.destroyAllWindows()


"""
pt1 = 1073, 365
pt2  = 657, 424
https://drive.google.com/file/d/1E9nVX0c17-OzZioCYroHZVGj4ANdtDkD/view?usp=sharing
"""
