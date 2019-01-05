import numpy as np
import cv2

def plot_corners(corners, rgb_img, size=4, thickness=2):
    kp_lst = []
    brg_img = cv2.cvtColor(np.copy(rgb_img*256).astype('u1'), cv2.COLOR_RGB2BGR)
    for x,y in corners:
        brg_img = cv2.circle(brg_img, (y,x), size, thickness=thickness, color=(0, 0, 255)) # Color is in BGR
    rgb_img =cv2.cvtColor(brg_img, cv2.COLOR_BGR2RGB)
    return rgb_img

def plot_corr(corr, img1, img2, mask=None):
    pts1 = [cv2.KeyPoint(y,x,0) for x,y in corr[:,:2]]
    pts2 = [cv2.KeyPoint(y,x,0) for x,y in corr[:,2:]]
    matches = [cv2.DMatch(i,i,0) for i in range(corr.shape[0]) if mask is None or mask[i] > 0]
    return cv2.drawMatches(img1, pts1, img2, pts2, matches, None, flags=4)

# Draw the epipolar lines
# Reference from https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html
def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def draw_epipolar_lines(img1, img2, corrs, F):
    pts1, pts2 = [], []
    dists = []
    for i in range(corrs.shape[0]):
        x_1, y_1, x_2, y_2 = corrs[i]
        pts1.append((y_1, x_1))
        pts2.append((y_2, x_2))
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    img1 = (img1.copy()*256.).astype(np.uint8)
    img2 = (img2.copy()*256.).astype(np.uint8)

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img1_out,img2_out = drawlines(img1,img2,lines1,pts1,pts2)

    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img4_out,img3_out = drawlines(img2,img1,lines2,pts2,pts1)

    img1_out = cv2.cvtColor(img1_out, cv2.COLOR_BGR2RGB)
    img2_out = cv2.cvtColor(img2_out, cv2.COLOR_BGR2RGB)
    img3_out = cv2.cvtColor(img3_out, cv2.COLOR_BGR2RGB)
    img4_out = cv2.cvtColor(img4_out, cv2.COLOR_BGR2RGB)
    return img1_out, img2_out, img3_out, img4_out

