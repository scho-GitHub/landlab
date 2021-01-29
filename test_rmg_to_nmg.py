# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:49:07 2021

@author: sahrendt
"""

import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import pandas as pd

#landlab modules
from landlab.grid.network import NetworkModelGrid
from landlab.plot import graph
from landlab import RasterModelGrid
from landlab.io import read_esri_ascii
from landlab import imshow_grid_at_node
from landlab.components import FlowAccumulator, ChannelProfiler
from landlab.components import DepressionFinderAndRouter
# Package for plotting raster data
from landlab.plot.imshow import imshow_grid, imshow_grid_at_node
from landlab.grid.create_network import create_network_from_raster



# tutorial_dir = os.path.dirname(os.getcwd())
# raster_fn = os.path.join(tutorial_dir, 'overland_flow/hugo_site.asc') #mac
# # raster_fn = os.path.join(tutorial_dir, 'overland_flow\\hugo_site.asc') #windows

raster_fn = r'C:\Users\sahrendt\Documents\GitHub\landlab\tests\io\test_read_esri_ascii\hugo_site.asc'

# Import the Hugo Site as a RasterModelGrid and visualize:


rmg, z = read_esri_ascii(raster_fn, name='topographic__elevation')
rmg.status_at_node[rmg.nodes_at_right_edge] = rmg.BC_NODE_IS_FIXED_VALUE
rmg.status_at_node[np.isclose(z, -9999.)] = rmg.BC_NODE_IS_CLOSED

nmg = create_network_from_raster(rmg, node_spacing=25,
                                 fields=['drainage_area', 'topographic__elevation'])

# Package for plotting networks
from landlab.plot import graph

## Plot nodes
plt.figure(figsize=(18,5))
plt.subplot(1,2,1)
graph.plot_nodes(nmg)
plt.title("Nodes")

## Plot nodes + links
plt.subplot(1,2,2)
graph.plot_nodes(nmg,with_id=False,markersize=4)
graph.plot_links(nmg)
plt.title("Links")
plt.show()


# In[74]:


# Plot topography
plt.figure(figsize=(12,8))
graph.plot_nodes(nmg,with_id=False,markersize=4)
graph.plot_links(nmg,with_id=False)
imshow_grid(rmg, 'topographic__elevation',            plot_name="Basin topography Overlayed with Network Model Grid",            color_for_closed=None,            colorbar_label="$z$ [m]")
plt.show()
