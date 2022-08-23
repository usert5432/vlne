from collections import namedtuple

HistData = namedtuple(
    'HistData', [ 'values', 'weights', 'label', 'color' ]
)

PlotData = namedtuple(
    'PlotData', [ 'pred', 'true', 'weights', 'label', 'color' ]
)


