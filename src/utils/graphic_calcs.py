import colorsys
import numpy as np
from collections import Counter
from shapely.geometry import LineString, Polygon, Point


def get_forehead_color(imageBGR, landmarks):
    delta = int(landmarks[27][1] - (landmarks[19][1]+landmarks[24][1])/2)
    fp3 = (landmarks[24][0], max(landmarks[24][1]-delta, 0))
    fp4 = (landmarks[19][0], max(landmarks[19][1]-delta, 0))
    face_forehead_color = get_color_of_zone(imageBGR, [landmarks[19], landmarks[24], fp3, fp4])
    return face_forehead_color


def switch_rgb(image):
    imageBGR = np.ndarray(image.shape, dtype=np.uint8)
    imageBGR[:, :, 0] = image[:, :, 2]
    imageBGR[:, :, 1] = image[:, :, 1]
    imageBGR[:, :, 2] = image[:, :, 0]
    return imageBGR


def get_dominant_color_masked(img, mask):
    """Calcs one dominant color on the image with mask."""
    masked_pixels = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                masked_pixels.append(img[i, j])
    cntr = Counter(tuple(map(tuple, masked_pixels)))
    dc = cntr.most_common(1)[0][0]
    return colorsys.rgb_to_hsv(dc[2]/255, dc[1]/255, dc[0]/255)


def get_color_of_line(img, p1, p2, p3):
    """Calcs one dominant color on the line."""

    points = []
    det = abs(int(p2[0] - p1[0]))
    coefs = np.linspace(0, 1, num=det)
    for t in coefs:
        x = p1[0] + t*(p2[0]-p1[0])
        y = p1[1] + t*(p2[1]-p1[1])
        points.append((min(x, img.shape[1]-1), min(y, img.shape[0]-1)))
    det = abs(int(p3[0] - p2[0]))
    coefs = np.linspace(0, 1, num=det)
    for t in coefs:
        x = p2[0] + t*(p3[0]-p2[0])
        y = p2[1] + t*(p3[1]-p2[1])
        points.append((min(x, img.shape[1]-1), min(y, img.shape[0]-1)))
    points = np.round(points).astype(np.int32)
    colors = [img[y, x] for x, y in points]

    cntr = Counter(tuple(map(tuple, colors)))
    dc = cntr.most_common(1)[0][0]
    return colorsys.rgb_to_hsv(dc[2]/255, dc[1]/255, dc[0]/255)


def get_color_of_zone(img, points):
    """Calcs one dominant color on the line."""
    zone = Polygon([points[0], points[1], points[2], points[3]])
    colors_in_zone = []
    minx = min([points[0][0], points[1][0], points[2][0], points[3][0]])
    maxx = max([points[0][0], points[1][0], points[2][0], points[3][0]])
    maxx = min(maxx, img.shape[1]-1)
    miny = min([points[0][1], points[1][1], points[2][1], points[3][1]])
    maxy = max([points[0][1], points[1][1], points[2][1], points[3][1]])
    maxy = min(maxy, img.shape[0]-1)
    for i in range(minx, maxx+1):
        for j in range(miny, maxy+1):
            if zone.contains(Point(i, j)):
                colors_in_zone.append(img[j, i])
    cntr = Counter(tuple(map(tuple, colors_in_zone)))
    dc = cntr.most_common(1)[0][0]
    return colorsys.rgb_to_hsv(dc[2]/255, dc[1]/255, dc[0]/255)
