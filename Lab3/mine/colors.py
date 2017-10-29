#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division  # Division in Python 2.7

import matplotlib

from mine.common import hsv2rgb, hsv_angle, linear_interpolation

matplotlib.use('Agg')  # So that we can render files without GUI
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np


def plot_color_gradients(gradients, names):
    # For pretty latex fonts (commented out, because it does not work on some machines)
    # rc('text', usetex=True)
    # rc('font', family='serif', serif=['Times'], size=10)
    rc('legend', fontsize=10)

    column_width_pt = 400  # Show in latex using \the\linewidth
    pt_per_inch = 72
    size = column_width_pt / pt_per_inch

    fig, axes = plt.subplots(nrows=len(gradients), sharex=True, figsize=(size, 0.75 * size))
    fig.subplots_adjust(top=1.00, bottom=0.05, left=0.25, right=0.95)

    for ax, gradient, name in zip(axes, gradients, names):
        # Create image with two lines and draw gradient on it
        img = np.zeros((2, 1024, 3))
        for i, v in enumerate(np.linspace(0, 1, 1024)):
            img[:, i] = gradient(v)

        im = ax.imshow(img, aspect='auto')
        im.set_extent([0, 1, 0, 1])
        ax.yaxis.set_visible(False)

        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.25
        y_text = pos[1] + pos[3] / 2.
        fig.text(x_text, y_text, name, va='center', ha='left', fontsize=10)

    fig.savefig('my-gradients.pdf')


def gradient_rgb_bw(v):
    return (v, v, v)


def gradient_rgb_gbr(v):
    if v <= 0.5:
        return 0, 1 - (v * 2), v * 2
    else:
        v -= 0.5
        return (v * 2), 0, 1 - (v * 2)


def gradient_rgb_gbr_full(v):
    r, g, b = gradient_rgb_gbr(v)
    multip = 2

    def lim(v):
        return min(1, v)

    return lim(multip * r), lim(multip * g), lim(multip * b)


steps = 7


def gradient_rgb_wb_custom(v):
    step = 1 / steps
    if v < step:
        return 1, decrease_val(v), 1
    elif v < step * 2:
        v -= step
        return decrease_val(v), 0, 1
    elif v < step * 3:
        v -= step * 2
        return 0, increase_val(v), 1
    elif v < step * 4:
        v -= step * 3
        return 0, 1, decrease_val(v)
    elif v < step * 5:
        v -= step * 4
        return increase_val(v), 1, 0
    elif v < step * 6:
        v -= step * 5
        return 1, decrease_val(v), 0
    else:
        v -= step * 6
        return decrease_val(v), 0, 0


def decrease_val(v):
    return min(1, max(1 - (v * steps), 0))


def increase_val(v):
    return min(1, v * steps)


def gradient_hsv_bw(v):
    return hsv2rgb(0, 0, v)


def gradient_hsv_gbr(v):
    hsv_colors = ((hsv_angle(120), 1, 1), (hsv_angle(180), 1, 1), (hsv_angle(240), 1, 1),
                  (hsv_angle(300), 1, 1), (hsv_angle(360), 1, 1))
    h, s, v = linear_interpolation(hsv_colors, v)
    return hsv2rgb(h, s, v)


def gradient_hsv_unknown(v):
    colors = ((hsv_angle(120), 1 / 2, 1), (hsv_angle(0), 1 / 2, 1))
    h, s, v = linear_interpolation(colors, v)
    return hsv2rgb(h, s, v)


def random_creator(amount):
    root = []
    step = min(1, max(0, 360 / amount))

    for i in range(amount):
        if i % 2 == 0:
            angle = i * step
        else:
            angle = 360 - i * step
        root.append((hsv_angle(angle), i / amount, 1))
    return root


def gradient_hsv_custom(v):
    colors = random_creator(4)
    h, s, v = linear_interpolation(colors, v)
    return hsv2rgb(h, s, v)


if __name__ == '__main__':
    def toname(g):
        return g.__name__.replace('gradient_', '').replace('_', '-').upper()


    gradients = (gradient_rgb_bw, gradient_rgb_gbr, gradient_rgb_gbr_full, gradient_rgb_wb_custom,
                 gradient_hsv_bw, gradient_hsv_gbr, gradient_hsv_unknown, gradient_hsv_custom)

    plot_color_gradients(gradients, [toname(g) for g in gradients])
