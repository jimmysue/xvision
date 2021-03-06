import cv2
import colorsys
import numpy as np
import random

def draw_bbox(image, bbox, color=None, thickness=1):
    bbox = bbox.reshape(-1, 2, 2).astype(np.int32)
    random_color = color is None
    for box in bbox:
        if random_color:
            color = tuple([random.randint(0, 255) for _ in range(3)])
        pt1 = tuple(box[0, :].tolist())
        pt2 = tuple(box[1, :].tolist())
        cv2.rectangle(image, pt1, pt2, color, thickness=thickness)

    return image


def draw_points(image, points, color=(0, 0, 255), radius=1, plot_index=False):
    draw_shiftbits = 4
    draw_multiplier = 1 << 4

    # points = (points * draw_multiplier).reshape(-1, 2).astype(np.int32)
    points = points.reshape(-1, 2)
    for i, p in enumerate(points):
        pt = (p * draw_multiplier).astype(np.int32)
        pt = tuple(pt.tolist())
        cv2.circle(image, pt, radius, color,
                   thickness=radius, shift=draw_shiftbits)
        if plot_index:
            pt =tuple( p.astype(np.int32).tolist() )
            txt = f'{i}'
            cv2.putText(image, txt, pt, cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))

    return image


def draw_shapes(image, shapes, radius=None):
    """draw point shape, assign different color for each index points
    """
    n = shapes.shape[0]
    shapes = shapes.reshape(n, -1, 2)
    num_pts = shapes.shape[1]
    if radius is None:
        h, w = image.shape[:2]
        radius = (h + w) * 0.5 * 0.01
        radius = max(1, int(radius))

    hue = np.linspace(0, 1, num_pts + 1)[:-1].tolist()
    value = [0.8] * num_pts
    saturation = [1.0] * num_pts

    colors = []
    for h, s, v in zip(hue, saturation, value):
        bgr = colorsys.hsv_to_rgb(h, s, v)
        colors.append(bgr)
    
    colors = (np.array(colors) * 255).astype(np.int32)
    shapes = np.transpose(shapes, [1, 0, 2])
    for i, pts in enumerate(shapes):
        color = tuple(colors[i].tolist())
        draw_points(image, pts,  color=color, radius=radius)
    return image