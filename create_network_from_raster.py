#!/usr/bin/env python
# coding: utf-8

# ## Create A Network Grid from Raster Grid
# 
# This notebook takes an .asc raster file representing a DEM and extracts a channel network from the topography using the [ChannelProfiler](https://landlab.readthedocs.io/en/master/reference/components/channel_profiler.html). It then uses this extracted channel network to create a NetworkModelGrid.

# In[72]:


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




# ## Import a DEM
# Setup this notebook to draw DEM data for the Hugo Site from the overland_flow tutorial folder.

# In[2]:


# tutorial_dir = os.path.dirname(os.getcwd())
# raster_fn = os.path.join(tutorial_dir, 'overland_flow/hugo_site.asc') #mac
# # raster_fn = os.path.join(tutorial_dir, 'overland_flow\\hugo_site.asc') #windows

raster_fn = r'C:\Users\sahrendt\Documents\GitHub\landlab\tests\io\test_read_esri_ascii\hugo_site.asc'

# Import the Hugo Site as a RasterModelGrid and visualize:

# In[3]:


rmg, z = read_esri_ascii(raster_fn, name='topographic__elevation')
rmg.status_at_node[rmg.nodes_at_right_edge] = rmg.BC_NODE_IS_FIXED_VALUE
rmg.status_at_node[np.isclose(z, -9999.)] = rmg.BC_NODE_IS_CLOSED


plt.figure()
imshow_grid_at_node(rmg, z, colorbar_label='Elevation (m)')
plt.show()


# ## Find flow accumulation:
# 
# Find where water will flow using the FlowAccumulator package:

# In[4]:


fa = FlowAccumulator(rmg, 
                     'topographic__elevation',
                     flow_director='FlowDirectorSteepest')
fa.run_one_step()

imshow_grid_at_node(rmg,
                    rmg.at_node['surface_water__discharge'],
                    colorbar_label='discharge')


# Yuck, that doesn't look like a very well defined channel network, let's fill in those depressions that are affecting the flow and then reroute it:

# In[5]:


df_4 = DepressionFinderAndRouter(rmg)
df_4.map_depressions()


# In[6]:


imshow_grid_at_node(rmg,
                    rmg.at_node['surface_water__discharge'],
                    colorbar_label='discharge')


# That looks better! 
# 
# ## Use the channel profiler to extract channels:
# * note, toggle minimum channel threshhold to refine network!

# In[43]:


min_channel_thresh = 10000

profiler = ChannelProfiler(
    rmg,
    number_of_watersheds=1,
    minimum_channel_threshold=min_channel_thresh,
    main_channel_only=False,
    cmap='jet',
)
profiler.run_one_step()

plt.figure(figsize = (6,6))
profiler.plot_profiles_in_map_view(colorbar_label='elevation')


# In[44]:


profiler.plot_profiles(xlabel='distance upstream (m)',
                  ylabel='elevation (m)')


# ## Get network-grid nodes from channel.

# Now we will get nodes and links from the ChannelProfiler dataset so we can create a network_grid. This network is composed of nodes and links, meaning it will work as a one-dimensional grid. We iterate through the watershed segments of the Channel profiler to define nodes at  intervals of 'd_node_spacing'
# 
# * Note: both of the following should be considered when setting node spacing
#      * d_node_spacing cannot be longer than shortest link segment (you don't want to skip channel segments! If you do want longer grid spacings, adjust the 'min_channel_thresh' variable in the ChannelProfiler function above in order to exclude smaller channel segments.
#      * d_node_spacing should be greater than the grid resolution (it doesn't make sense to have nodes that are closer together than channels can be resolved by the underlying DEM)

# In[59]:


d_node_spacing = 25 #units to space nodes, must be greater than grid resolution
x_of_nodes = np.array([]) #empty list to store x locations of network model grid nodes
y_of_nodes = np.array([]) #empty list to store y locations of network model grid nodes
rmg_nodes = np.array([]) #empty list to store raster model grid node corresponding to each network model grid node
colors_nodes = [] #empty list to store colors corresponding to each channel segment
nodes_per_segment = [] #empty list to store number of nodes in each channel segment

#DEV ONLY: keep track of nodes per seg as explicit x and y locations (only toggle if using below)
# x_of_nodes = []
# y_of_nodes = []
links = []
node_xy = []

#would be ideal to also create dictionary here that stores rmg cells along each link to tie rmg properties to nmg
#nmg_cells_on_link = {}

#obtain watershed key
ws_keys = [k for k in profiler.data_structure.keys()]

#sent message if there is more than 1 watershed (code is not currently set up to handle multipe watersheds)
if len(ws_keys) > 1:
    print('more than one watershed in DEM, unable to deliniate network properly')

#access number of channel segments
n_channel_segs = len(profiler.data_structure[ws_keys[0]])
#generate random colors to keep track of channel segs for plotting
colors = [list(np.random.random(3)) for i in range(n_channel_segs)] 

#find the nodes of the start and end index of each segment
ws_start =[ws_seg[0] for ws_seg in profiler.data_structure[ws_keys[0]].keys()] 
ws_end = [ws_seg[1] for ws_seg in profiler.data_structure[ws_keys[0]].keys()]

#loop through each segment
for i, seg_key in enumerate(profiler.data_structure[ws_keys[0]].keys()):
    #access individual segment
    seg_i = profiler.data_structure[ws_keys[0]][seg_key] 
    #print(seg_i)
    # create array of idealized node locations using previously specified distance between nodes
    node_locs = np.arange(seg_i['distances'][0], seg_i['distances'][-1], d_node_spacing)
    n_possible_nodes = len(node_locs)
    nodes_per_segment.append(n_possible_nodes)
    
    #Find the index of the nearest channel cells to idealized node locations:
    idx_nodes = [np.abs(seg_i['distances'] - loc).argmin() for loc in node_locs]
    
    #obtain list of raster model grid nodes corresponding to channel nodes
    rmg_nodes_i = seg_i['ids'][idx_nodes]
    rmg_nodes = np.append(rmg_nodes, rmg_nodes_i) #append to list that will connect rmg node values to nmg
    
    #append x,y values to list of network model grid nodes from raster model grid
    x_of_nodes = np.append(x_of_nodes, rmg.x_of_node[rmg_nodes_i])
    y_of_nodes = np.append(y_of_nodes, rmg.y_of_node[rmg_nodes_i])
    
    #find links for x,y values
    for n in range(0,n_possible_nodes-1):

        head_node_xy = (rmg.x_of_node[seg_i['ids'][idx_nodes[n]]], rmg.y_of_node[seg_i['ids'][idx_nodes[n]]])
        tail_node_xy = (rmg.x_of_node[seg_i['ids'][idx_nodes[n+1]]], rmg.y_of_node[seg_i['ids'][idx_nodes[n+1]]])
            
        #the code below is taken from the read_shapefile landlab code
            
        # we should expect that the head node and tail node of later links will
        # already be part of the model grid. So we check, and add the nodes,
        # if they don't already exist.

        if head_node_xy not in node_xy:
            node_xy.append(head_node_xy)

        if tail_node_xy not in node_xy:
            node_xy.append(tail_node_xy)

        # get the index of the head and tail node index.
        head_node__node_id = node_xy.index(head_node_xy)
        tail_node__node_id = node_xy.index(tail_node_xy)

        # append the head and tail node ids to the link array
        links.append((head_node__node_id, tail_node__node_id))
            
        # check to see if the last node needs to be connected to the start of another segment
        if n == n_possible_nodes-2:
            if seg_i['ids'][-1] in ws_start:
                #Find the start segment to figured out the tail_node 
                seg_n = profiler.data_structure[ws_keys[0]][(ws_start[ws_start.index(seg_i['ids'][-1])],
                                                             ws_end[ws_start.index(seg_i['ids'][-1])])]
                
                head_node_xy = (rmg.x_of_node[seg_i['ids'][idx_nodes[n+1]]], rmg.y_of_node[seg_i['ids'][idx_nodes[n+1]]])
                tail_node_xy = (rmg.x_of_node[seg_n['ids'][0]], rmg.y_of_node[seg_n['ids'][0]])
                
                #the code below is taken from the read_shapefile landlab code
            
                # we should expect that the head node and tail node of later links will
                # already be part of the model grid. So we check, and add the nodes,
                # if they don't already exist.

                if head_node_xy not in node_xy:
                    node_xy.append(head_node_xy)

                if tail_node_xy not in node_xy:
                    node_xy.append(tail_node_xy)

                # get the index of the head and tail node index.
                head_node__node_id = node_xy.index(head_node_xy)
                tail_node__node_id = node_xy.index(tail_node_xy)

                # append the head and tail node ids to the link array
                links.append((head_node__node_id, tail_node__node_id))
       
    #DEV ONLY: keep track of nodes per seg as explicit x and y locations
#     x_of_nodes.append(rmg.x_of_node[rmg_nodes_i])
#     y_of_nodes.append(rmg.y_of_node[rmg_nodes_i])
    
    #add colors to correspond to each segment (this is just for dev plotting and can be removed later)
    for n in range(len(rmg_nodes_i)):
        colors_nodes.append(colors[i])

print(links)     
#get unique nodes
xy_df = pd.DataFrame({'x': x_of_nodes, 'y': y_of_nodes})
uniq_x_of_nodes = xy_df.drop_duplicates()['x'].values
uniq_y_of_nodes = xy_df.drop_duplicates()['y'].values

#print(x_of_nodes)
#print(y_of_nodes)


# ### Helpful plots for developing correct link connections from ChannelProfiler:
# * Plot nodes colored by channel segment:

# In[69]:


plt.figure(figsize = (10,8))
profiler.plot_profiles_in_map_view(    colorbar_label='Elevation [m]',shrink=0.65,    color_for_closed=None,output=False)
plt.scatter(x_of_nodes, y_of_nodes, c=colors_nodes, s=30, zorder=2)

#hacky way to label channel segments with #s -------
node_id = 0 #dummy counter
idx_x = [] #store indexes to acces x,y location of last node on channel
#loop through channels, get node index
for node_len in nodes_per_segment:
    node_id+=node_len
    idx_x.append(node_id-1)
#plot numbers at last node for each channel seg
for i in range(n_channel_segs):
    plt.text(x_of_nodes[idx_x[i]]+20,
             y_of_nodes[idx_x[i]]+20,
             '%d'%i,
             color='white',backgroundcolor=colors[i],
             fontweight='bold',fontsize="large")
plt.text(10,10,'#: channel segment',
         color='white', backgroundcolor="black",
         fontweight='bold')
plt.title('Network Nodes colored by channel segment')
plt.show()


# ## Sort our nodes and links to put into the NetworkModelGrid using the NetworkGraph:

# In[67]:


from landlab.graph.graph import NetworkGraph
# Create a Network Model Grid.
x_of_node, y_of_node = zip(*node_xy)

# We want to ensure that we maintain sorting, so start by creating an
# unsorted network graph and sorting.
# The sorting is important to ensure that the fields are assigned to
# the correct links.
graph_net = NetworkGraph((y_of_node, x_of_node), links=links, sort=False)
sorted_nodes, sorted_links, sorted_patches = graph_net.sort()

# use the sorting information to make a new network model grid.
nmg = NetworkModelGrid(
    (np.asarray(y_of_node)[sorted_nodes], np.asarray(x_of_node)[sorted_nodes]),
    np.vstack((graph_net.node_at_link_head, graph_net.node_at_link_tail)).T,
)

nmg.at_node['rmg_node_value'] = rmg_nodes[sorted_nodes]
# ###  Plot Network nodes and links

# In[68]:


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


# ### Another way to plot unique network nodes:
# * primarily for troubleshooting if the NetworkModelGrid throws and error

# In[75]:


plt.figure(figsize = (15,15))
profiler.plot_profiles_in_map_view(colorbar_label='elevation')
plt.scatter(uniq_x_of_nodes, uniq_y_of_nodes, s=30, c='white', zorder=2)
for i, (x,y) in enumerate(zip(uniq_x_of_nodes, uniq_y_of_nodes)):
    plt.text(x-10,y+5,
            '%d'%i,
             color='white',
            fontweight='bold')
#---------------------------------------------------
plt.title('Unique Network Nodes')


# In[ ]:




