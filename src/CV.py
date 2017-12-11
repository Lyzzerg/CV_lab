from random import random

import cv2
import numpy as np
import pymp


def minus(variable):
    result = variable
    if variable < 0:
        result = 0
    return int(result)


def plus(variable1, variable2):
    result = variable1
    if variable1 >= variable2:
        result = variable2 - 1
    return int(result)


def processing(image_):
    edge_thresh = 12
    mask_size = cv2.DIST_MASK_5
    dist_type = cv2.DIST_L2
    coefficient = 3

    gray = cv2.cvtColor(image_, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, edge_thresh, 3 * edge_thresh, 3)

    _, threshold = cv2.threshold(edges, 100, 255, cv2.THRESH_TRUNC)

    distance = cv2.distanceTransform(threshold, dist_type, mask_size)

    integal = cv2.integral(gray)

    rows, columns = np.array(gray).shape

    output = pymp.shared.array((rows, columns), dtype='uint8')

    with pymp.Parallel() as p:
        for i in p.range(0, rows):
            for j in range(0, columns):
                dist = float(distance[i][j])
                if dist == 0:
                    dist = 2

                i_minus = minus(i - coefficient * dist / 2)
                j_minus = minus(j - coefficient * dist / 2)
                i_plus = plus((i + coefficient * dist / 2), rows)
                j_plus = plus((j + coefficient * dist / 2), columns)

                size = (coefficient * dist) * (coefficient * dist)

                output[i][j] = (integal[i_minus][j_minus] + integal[i_plus][j_plus] - integal[i_minus][j_plus]
                                - integal[i_plus][j_minus]) / size

    return gray, edges, threshold, distance, output


def reLU(value):

    if value < 0:
        value *= 0

    return value


def convolution_3d(tensor_, filter_):

    W, H, D = tensor_.shape
    w, h, d = filter_.shape

    out_ = np.empty((W, H))

    for x in range(0, W):
        for y in range(0, H):

            out_[x][y] = 0

            for z in range(0, D):

                for x_ in range(0, w):
                    for y_ in range(0, h):
                        for z_ in range(0, d):

                            if (x >= 0 + int(w / 2)) and (x < W - int(w / 2)) and \
                                    (y >= 0 + int(h / 2)) and (y < H - int(h / 2)) and \
                                    (z >= 0 + int(d / 2)) and (z < D - int(d / 2)):
                                out_[x][y] += tensor_[x - int(w / 2) + x_][y - int(h / 2) + y_][z - int(d / 2) + z_] *\
                                              filter_[x_][y_][z_]
                            else:
                                out_[x][y] += 0

            out_[x][y] = int(reLU(out_[x][y] / (w * h * d)))

    return out_


image = cv2.imread("../images/image1.png", cv2.IMREAD_COLOR)

gray_, edges_, threshold_, distance_, output_ = processing(image)

images = [gray_, edges_, threshold_, distance_, output_]
names = ['gray', 'edges', 'threshold', 'distance', 'output']

i = 0
for name in names:
    cv2.imwrite('../result/' + name + '.png', images[i])
    i += 1

W, H, D = 5, 7, 9
w, h, d = 3, 3, 5

tensor_ = np.empty((W, H, D))
filter_ = np.empty((w, h, d))

for x in range(0, W):
    for y in range(0, H):
        for z in range(0, D):
            tensor_[x][y][z] = int(random() * 100)
            if (x < w) & (y < h) & (z < d):
                filter_[x][y][z] = random() * 10/(w*h*d)

conv_ = convolution_3d(tensor_, filter_)

print('tensor')
print(tensor_)
print('filter')
print(filter_)
print('conv')
print(conv_)

cv2.waitKey(0)
cv2.destroyAllWindows()
