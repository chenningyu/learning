import cv2
import numpy as np
import glob


# 设置棋盘格尺寸和方块大小
chessboard_size = (10, 7)
square_size = 25  # mm

# 准备棋盘格点的3D坐标
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 存储对象点和图像点
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# 读取图像
images = glob.glob('chessboard_img/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 如果找到，添加对象点和图像点
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 绘制并显示角点
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# 执行相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 打印相机内参
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)