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
    gauss_sigma = 3
    mask_size = cv2.DIST_MASK_5
    dist_type = cv2.DIST_L2
    coefficient = 3

    gray = cv2.cvtColor(image_, cv2.COLOR_RGB2GRAY)

    gauss = cv2.GaussianBlur(gray, (3, 3), gauss_sigma)

    edges = cv2.Canny(gauss, edge_thresh, 3 * edge_thresh, 3)

    _, threshold = cv2.threshold(edges, 100, 255, cv2.THRESH_TRUNC)

    distance = cv2.distanceTransform(threshold, dist_type, mask_size)

    normal = distance
    cv2.normalize(distance, normal, 0, 1, cv2.NORM_MINMAX)

    rows, columns = np.array(gray).shape

    output = np.copy(gray)

    with pymp.Parallel() as p:
        for i in p.range(0, rows):
            for j in range(0, columns):
                dist = float(normal[i][j])
                if dist == 0:
                    dist = 2

                i_minus = minus(i - coefficient * dist/2)
                j_minus = minus(j - coefficient * dist/2)
                i_plus = plus((i + coefficient * dist / 2), rows)
                j_plus = plus((j + coefficient * dist / 2), columns)

                sum_, count = 0, 0

                for i_ in range(i_minus, i_plus+1):
                    for j_ in range(j_minus, j_plus+1):
                        sum_ += int(gray[i_][j_])
                        count += 1
                output[i][j] = int(sum_/count)

    return gray, gauss, edges, threshold, distance, normal, output


image_name = "image1"
image_type = ".png"

image = cv2.imread("images/"+image_name+image_type, cv2.IMREAD_COLOR)

gray_, gauss_, edges_, threshold_, distance_,  normal_, output_ = processing(image)

cv2.imshow('original', image)
cv2.imshow('gray', gray_)
cv2.imshow('gauss', gauss_)
cv2.imshow('edges', edges_)
cv2.imshow('result', output_)

cv2.waitKey(0)
cv2.destroyAllWindows()
