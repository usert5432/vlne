"""
This module contains functions to make various plots.

Notes
-----
During the module initialization it sets the default plot parameters.
"""

import os

import matplotlib.style
import matplotlib as mpl

from cycler import cycler

mpl.rcParams['axes.prop_cycle'] = cycler(color = 'krgbcmy')

