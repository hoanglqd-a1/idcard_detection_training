"""Image loading, perspective crops, and edge-based document refinement.

Images use RGB channel order; corner arrays have shape (4, 2) with (x, y) pairs.
Perspective transform adapted from:
https://github.com/KMKnation/Four-Point-Invoice-Transform-with-OpenCV/blob/master/four_point_object_extractor.py
"""

from itertools import combinations

import cv2
import numpy as np
from PIL import Image


def load_image(image_path, image_size: tuple[int, int]):
    """Load an RGB image and resize to (width, height)."""
    with Image.open(image_path) as image:
        return np.array(image.convert('RGB').resize(image_size))


def order_points(pts):
    """Order corners as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(pts, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError('Expected four finite (x, y) corners.')
    coordinate_sum = points.sum(axis=1)
    coordinate_difference = np.diff(points, axis=1).ravel()
    return points[[
        np.argmin(coordinate_sum), np.argmin(coordinate_difference),
        np.argmax(coordinate_sum), np.argmax(coordinate_difference),
    ]]


def four_point_transform(image, pts):
    """Warp a quadrilateral into a rectangle; reject degenerate corners."""
    rectangle = order_points(pts)
    top_left, top_right, bottom_right, bottom_left = rectangle
    width = int(max(np.linalg.norm(bottom_right - bottom_left),
                    np.linalg.norm(top_right - top_left)))
    height = int(max(np.linalg.norm(top_right - bottom_right),
                     np.linalg.norm(top_left - bottom_left)))
    if width < 2 or height < 2 or not cv2.isContourConvex(rectangle):
        raise ValueError('Corners must form a nondegenerate convex quadrilateral.')
    destination = np.array([
        [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(rectangle, destination)
    return cv2.warpPerspective(image, transform, (width, height))


def check_validity(points, image_size):
    """Keep the original contour-spacing rule for candidate quadrilaterals."""
    corners = np.asarray(points).reshape(4, 2)
    return all(
        abs(first[0] - second[0]) >= image_size[0] // 8
        or abs(first[1] - second[1]) >= image_size[1] // 8
        for first, second in combinations(corners, 2)
    )


def find_contours(image, thickness=3):
    contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, 255, thickness)
    return contour_image, contours, hierarchy


def get_corners(raw_image, mask_image):
    """Return valid four-corner contours, ordered by decreasing area."""
    if raw_image is None:
        raise ValueError('raw_image must not be None.')
    _, contours, _ = find_contours(mask_image)
    candidates = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        perimeter = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(corners) == 4 and check_validity(corners, mask_image.shape):
            candidates.append(corners)
    return candidates


def expand_corners(shape, corners, expand_rate=0.05):
    """Expand corners around their center and clip to the image bounds."""
    corners = np.asarray(corners)
    center = np.mean(corners, axis=0)
    expanded = center + (corners - center) * (1 + expand_rate)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, shape[1] - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, shape[0] - 1)
    return expanded


def crop_image(raw_image, corners, expand_rate=0.05):
    expanded = expand_corners(raw_image.shape, corners, expand_rate)
    return four_point_transform(raw_image, expanded)


def extract_card(image, corners):
    if len(corners) < 4:
        return None
    return four_point_transform(image, np.asarray(corners, dtype=np.int32))


def auto_canny(image, sigma=0.33):
    median = np.median(image)
    lower = int(max(0, (1 - sigma) * median))
    upper = int(min(255, (1 + sigma) * median))
    return cv2.Canny(image, lower, upper, L2gradient=True)


def find_intersections(lines, im):
    """Find unique line intersections within a 10% margin of the image."""
    height, width = im.shape[:2]
    intersections = []
    for (rho1, theta1), (rho2, theta2) in combinations(lines, 2):
        coefficients = [[np.cos(theta1), np.sin(theta1)],
                        [np.cos(theta2), np.sin(theta2)]]
        try:
            x, y = np.round(np.linalg.solve(coefficients, [rho1, rho2])).astype(int)
        except np.linalg.LinAlgError:
            continue  # Parallel lines do not have a unique intersection.
        point = (int(x), int(y))
        if -width / 10 < x < width * 1.1 and -height / 10 < y < height * 1.1:
            if point not in intersections:
                intersections.append(point)
    return intersections


def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)


def draw_lines(image, lines):
    """Draw Hough lines onto image in place and return it."""
    for rho, theta in lines:
        cosine, sine = np.cos(theta), np.sin(theta)
        x, y = cosine * rho, sine * rho
        start = (int(x - 1000 * sine), int(y + 1000 * cosine))
        end = (int(x + 1000 * sine), int(y - 1000 * cosine))
        cv2.line(image, start, end, (0, 0, 255), 2)
    return image


def get_lines(edges):
    """Keep approximately horizontal/vertical Hough lines and merge neighbors."""
    detected = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if detected is None:
        return []
    groups = []
    for rho, theta in detected[:, 0, :]:
        if np.pi / 6 < theta < np.pi / 3 or np.pi * 2 / 3 < theta < np.pi * 5 / 6:
            continue
        if theta > np.pi * 5 / 6:
            rho, theta = -rho, theta - np.pi
        for group in groups:
            mean_rho, mean_theta = np.mean(group, axis=0)
            if abs(rho - mean_rho) < 50 and abs(theta - mean_theta) < 0.3:
                group.append((rho, theta))
                break
        else:
            groups.append([(rho, theta)])
    return [tuple(np.mean(group, axis=0)) for group in groups]


def document_detect(cropped_image):
    """Refine a crop using border edges; return None when no valid crop exists."""
    inverted = cv2.bitwise_not(cropped_image)
    gray = cv2.cvtColor(inverted, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 15)
    edges = auto_canny(blurred, sigma=0.5)

    height, width = edges.shape
    border_mask = np.full_like(edges, 255)
    cv2.rectangle(border_mask, (width // 8, height // 8),
                  (width * 7 // 8, height * 7 // 8), 0, -1)
    lines = get_lines(apply_mask(edges, border_mask))
    corners = find_intersections(lines, cropped_image)[:4]
    try:
        return extract_card(cropped_image, corners)
    except ValueError:
        return None


def convert_rec2corners(rec):
    """Convert (left, top, right, bottom) into four (x, y) corners."""
    left, top, right, bottom = rec
    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
