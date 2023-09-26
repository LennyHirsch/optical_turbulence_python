#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import propagation_functions as prop
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import argparse
from typing import List, Tuple
from math import sqrt


def calculate_circle_positions(x: int, y: int, radius: float =3.0) -> List[Tuple[int, int]]:
    """ create a small collection of points in a neighborhood of some point 
    """
    neighborhood = []

    X = int(radius)
    for i in range(-X, X + 1):
        Y = int(pow((radius) * (radius) - i * i, 1/2))
        for j in range(-Y, Y + 1):
            neighborhood.append((x + i, y + j))

    return neighborhood


def plotter(sigma=3, posx=0, posy=0):
    """ Plot a binary image """    
    # arr = np.zeros([sigma * 2 + 1] * 2)
    arr = np.zeros((512,512))

    points = calculate_circle_positions(int(posx), int(posy), sigma)

    # flip pixel value if it lies inside (or on) the circle
    for p in points:
        arr[p] = 1

    # fig = plt.figure(0)
    # ax  = fig.add_subplot(111, aspect='equal')

    plt.imshow(-arr, interpolation='none', cmap='gray')
    plt.show()
    return points

def plot_circles(grid):
    arr = np.ones((2048,2048))
    for x in range(2000):
        for y in range(2000):
            if grid[x][y] == 0:
                points = calculate_circle_positions(int(x), int(y), 11)
                for p in points:
                    arr[p] = 0

    plt.imshow(arr, interpolation='none', cmap='gray')
    plt.show()
    return arr

absorbers = prop.AbsorberScreen(4, 2048, 11, 100)
grid = absorbers.generate_absorbers()
arr = plot_circles(grid)
print('bloop')