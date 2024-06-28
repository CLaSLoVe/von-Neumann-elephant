import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize

def get_curve(y_bias=0, projection='polar'):
    image = cv2.imread('elephant.jpg')
    if image is None:
        raise FileNotFoundError("图像文件未找到，请检查文件路径。")

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化图像
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 确认是否找到轮廓
    if not contours:
        raise ValueError("未找到任何轮廓，请检查二值化步骤。")

    # 假设只有一个轮廓，取第一个
    contour = contours[0]

    # 总点数
    total_points = len(contour)

    # 检查轮廓点数是否足够
    if total_points < 100:
        raise ValueError(f"轮廓点数不足，只有 {total_points} 个点。")

    # 采样100个点
    sampled_points = []
    for i in range(100):
        index = int(i * total_points / 100)
        sampled_points.append(contour[index][0])

    # 转换为NumPy数组
    sampled_points = np.array(sampled_points)

    x_coords = sampled_points[:, 0]
    y_coords = sampled_points[:, 1]

    # 计算质心
    x_c = np.mean(x_coords)
    y_c = np.mean(y_coords)-y_bias

    # 转换为相对于质心的坐标
    x_prime = x_coords - x_c
    y_prime = y_coords - y_c

    if projection == 'cartesian':
        return x_prime, y_prime
    elif projection == 'polar':
        # 转换为极坐标
        r0 = np.sqrt(x_prime**2 + y_prime**2)
        theta0 = np.arctan2(y_prime, x_prime) + np.pi / 2
        return theta0, r0

    return x_prime, y_prime



