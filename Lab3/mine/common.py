from numpy import math


def norm(v):
    return int(v * 255)


def hsv2rgb(h, s, v):
    if s == 0.0:
        return (v, v, v)
    i = int(h * 6.)  # XXX assume int() truncates!
    f = (h * 6.) - i
    p, q, t = v * (1. - s), v * (1. - s * f), v * (1. - s * (1. - f))
    i %= 6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    if i == 5: return (v, p, q)


def linear_points_interpolation(rgb1, rgb2, c):
    rgb = ((1.0 - c) * a + c * b for a, b in zip(rgb1, rgb2))
    return tuple(rgb)


def linear_interpolation(colors, x):
    max_index = (len(colors) - 1)
    interval = (1. / max_index)
    ci1 = math.floor(x / interval)
    ci2 = min(math.ceil(x / interval), max_index)

    return linear_points_interpolation(colors[ci1], colors[ci2], (x - interval * ci1) / interval)


def hsv_angle(angle):
    return angle / 360.0
