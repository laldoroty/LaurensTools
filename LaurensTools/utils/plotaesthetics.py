"""
Use:
> import LaurensTools.utils.plotaesthetics

to set plot style globally.

"""

import matplotlib as mpl
from matplotlib import rcParams

def update_rcParams(key, val):
    if key in rcParams:
        rcParams[key] = val

update_rcParams('font.size', 20)
update_rcParams('font.family', 'serif')
update_rcParams('xtick.major.size', 8)
update_rcParams('xtick.labelsize', 'large')
update_rcParams('xtick.direction', "in")
update_rcParams('xtick.minor.visible', True)
update_rcParams('xtick.top', True)
update_rcParams('ytick.major.size', 8)
update_rcParams('ytick.labelsize', 'large')
update_rcParams('ytick.direction', "in")
update_rcParams('ytick.minor.visible', True)
update_rcParams('ytick.right', True)
update_rcParams('xtick.minor.size', 4)
update_rcParams('ytick.minor.size', 4)
update_rcParams('xtick.major.pad', 10)
update_rcParams('ytick.major.pad', 10)
update_rcParams('legend.numpoints', 1)
update_rcParams('mathtext.fontset', 'cm')
update_rcParams('mathtext.rm', 'serif')
update_rcParams('axes.labelsize', 'x-large')
update_rcParams('lines.markersize', 10)
update_rcParams('lines.markeredgewidth', 1)
update_rcParams('lines.markeredgecolor', 'k')

cycle_colors = ['darkorchid','darkorange','darkturquoise','6FADFA','7D7D7D','black']
# cycle_colors = ['9F6CE6','FF984A','538050','6FADFA','7D7D7D','black']
cycle_markers = ['o','^','*','s','X','d']
# cycle_colors = ['darkorchid','darkorange','darkturquoise']
# cycle_markers = ['o','^','*']
update_rcParams('axes.prop_cycle', mpl.cycler(color=cycle_colors) + mpl.cycler(marker=cycle_markers))
# update_rcParams('axes.prop_cycle', mpl.cycler(color=cycle_colors))
