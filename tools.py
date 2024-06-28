import cv2
import numpy as np


def get_curve(y_bias=0, projection='polar'):
    image = cv2.imread('elephant.jpg')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contour = contours[0]

    total_points = len(contour)

    sampled_points = []
    for i in range(100):
        index = int(i * total_points / 100)
        sampled_points.append(contour[index][0])

    sampled_points = np.array(sampled_points)

    x_coords = sampled_points[:, 0]
    y_coords = sampled_points[:, 1]

    x_c = np.mean(x_coords)
    y_c = np.mean(y_coords)-y_bias

    x_prime = x_coords - x_c
    y_prime = y_coords - y_c

    if projection == 'cartesian':
        return x_prime, y_prime
    elif projection == 'polar':
        r0 = np.sqrt(x_prime**2 + y_prime**2)
        theta0 = np.arctan2(y_prime, x_prime) + np.pi / 2
        return theta0, r0

    return x_prime, y_prime



