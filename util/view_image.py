iname = '/home/nugrinovic/Documents/code/30_depth_from_monocular/midas_v3/MiDaS/results_depthOrder_w800_sc2/mupots-3d/TS13/img_000284_resized/depth_map.jpg'

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from misc import plot
img = plt.imread(iname)
plot(img)
