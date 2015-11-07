# -*- coding: utf-8 -*-
# Mathieu Bouchard && Pierre Gerard

import numpy


def minkowski(x, y, p=2.):
    """
    Calcule la distance Minkowski entre un vecteur x et une vecteur Y
    """
    if type(x) == float or type(x) == int:
        x = numpy.array([x])
    if type(y) == float or type(y) == int:
        y = numpy.array([y])
    if type(y) == numpy.float64:
        return (numpy.abs(x - y))
    if type(x) == numpy.float64:
        return (numpy.abs(x - y))
    else:
        sum = 0
        for i in range(len(x)):
            sum += numpy.abs(x[i] - y[i]) ** p
        return sum ** (1.0 / p)
