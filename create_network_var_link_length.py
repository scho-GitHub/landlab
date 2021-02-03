# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:18:07 2021

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

raster_fn = r'C:\Users\sahrendt\Documents\GitHub\landlab\tests\io\test_read_esri_ascii\hugo_site.asc'

# Import the Hugo Site as a RasterModelGrid and visualize:

# In[3]:
# width to area scaling parameters
# from Frasson et al. 2019 GRL
# units for relationship convert area in km to width in m
# could calibrate using remote sensing measurements from site 
a = 9.68
b = 0.32 
n_widths = 30

rmg, z = read_esri_ascii(raster_fn, name='topographic__elevation')
rmg.status_at_node[rmg.nodes_at_right_edge] = rmg.BC_NODE_IS_FIXED_VALUE
rmg.status_at_node[np.isclose(z, -9999.)] = rmg.BC_NODE_IS_CLOSED


plt.figure()
imshow_grid_at_node(rmg, z, colorbar_label='Elevation (m)')
plt.show()


fa = FlowAccumulator(rmg, 
                     'topographic__elevation',
                     flow_director='D8',
                     depression_finder='DepressionFinderAndRouter')
fa.run_one_step()

imshow_grid_at_node(rmg,
                    rmg.at_node['surface_water__discharge'],
                    colorbar_label='discharge')


min_channel_thresh = 8000

#should specify outlet node

profiler = ChannelProfiler(
    rmg,
    number_of_watersheds=1,
    minimum_channel_threshold=min_channel_thresh,
    # outlet_node = ,
    main_channel_only=False,
    cmap='jet',
)
profiler.run_one_step()

plt.figure(figsize = (6,6))
profiler.plot_profiles_in_map_view(colorbar_label='elevation')



plt.figure(figsize = (6,6))
profiler.plot_profiles(xlabel='distance upstream (m)',
                  ylabel='elevation (m)')



#obtain watershed key (should only be one)
wtrshd_key = [k for k in profiler.data_structure.keys()][0]

#sent message if there is more than 1 watershed (code is not currently set up to handle multipe watersheds)
# if len(ws_keys) > 1:
#     raise ValueError('more than one watershed in DEM, unable to deliniate network properly')

 #access number of channel segments
n_channel_segs = len(profiler.data_structure[wtrshd_key])

channel_segment_keys = profiler.data_structure[wtrshd_key].keys()

# IDENTIFY SEGMENT CONNECTIVITY ------------------
# obtain node ids for start and end of every channel segments
seg_starts =[seg[0] for seg in profiler.data_structure[wtrshd_key].keys()] 
seg_ends = [seg[1] for seg in profiler.data_structure[wtrshd_key].keys()]

# add a connectivity key if 
for i, seg_key in enumerate(channel_segment_keys):
    connectivity_key = None
    seg_i = profiler.data_structure[wtrshd_key][seg_key]
    # if the start of a segment is the end of another segment
    if seg_key[0] in seg_ends:
        #find first segment downstream that should connect to last node of
        connect_to_channel_idx = np.argmax(seg_key[0]==seg_ends)
        connectivity_key = (seg_starts[connect_to_channel_idx],
                       seg_ends[connect_to_channel_idx])

    seg_i['connectivity_key'] = connectivity_key

# x_of_nodes = [] #empty list to store x locations of network model grid nodes
# y_of_nodes = [] #empty list to store y locations of network model grid nodes
node_xy = [] #empty list to store paired x,y locations of nmg nodes
rmg_nodes = [] #empty list to store raster model grid node corresponding to each network model grid node    
links = [] # empty list to store link connections between nodes

for i, seg_key in enumerate(channel_segment_keys):
    print(i)
    # access data of channel segments
    seg_i = profiler.data_structure[wtrshd_key][seg_key]
    # create list to hold rmg node ids where nmg nodes are located
    nmg_nodes = []
    # identify rmg value of first node in segment
    # first nodes of channel segments will always be included in network 
    # if they don't already exist
    idx_node = 0
    rmg_node = seg_i['ids'][idx_node]
    
    # if necessary, add link connecting first node of segment
    # to downstream node
    if seg_i['connectivity_key'] is not None:
        channel_key = seg_i['connectivity_key']
        connecting_seg = profiler.data_structure[wtrshd_key][channel_key]
        # check to make sure there are nmg nodes on downstream segment
        if len(connecting_seg['ids_nmg']) > 0:
            connect_node = connecting_seg['ids_nmg'][-1]
        # if there are no nmg nodes on the downstream segment
        # it must be too short for calculated node spacing
        # if this is the case, connect upstream segment to first node in 
        # dowsntream connecting seg
        else:
            connect_node = connecting_seg['ids'][0]
        # add a link for this connection
        # add head node as first node of segment
        head_node_xy = (rmg.x_of_node[rmg_node],
                        rmg.y_of_node[rmg_node])
        # add tail node as last node of downstream segment
        tail_node_xy = (rmg.x_of_node[connect_node],
                        rmg.y_of_node[connect_node])
        # if these nodes don't already exist in the array of node xy vals from
        # another channel segment, add them
        if head_node_xy not in node_xy:
            node_xy.append(head_node_xy)
        if tail_node_xy not in node_xy:
            node_xy.append(tail_node_xy)
        # get the index of the head and tail node from our node_xy list
        # must do this in case they weren't added with above if statements
        head_node__node_id = node_xy.index(head_node_xy)
        tail_node__node_id = node_xy.index(tail_node_xy)
        # append the head and tail node ids to the link array
        links.append((head_node__node_id, tail_node__node_id))
                
    # iterate over segment adding new nodes as long as there are upstream nodes
    # that can be placed on network model grid based upon node spacing
    upstrm_node=True
    while upstrm_node is True:
        # identify x and y of this node
        # x_of_node, y_of_node = rmg.x_of_node[rmg_node], rmg.y_of_node[rmg_node]
        
        # if we haven't already stored the rmg id value for this new node
        # add it to our master list of rmg nodes and sub-list of nmg nodes
        if rmg_node not in rmg_nodes:
            rmg_nodes.append(rmg_node)
            nmg_nodes.append(rmg_node)
            # x_of_nodes.append(x_of_node)
            # y_of_nodes.append(y_of_node)
            
        #calculate drainage area contributing to this node
        da_node = rmg.at_node['drainage_area'][rmg_node]
        #relate drainage area to river width (convert area to km, width in m)
        # from Frasson et al. 2019 GRL
        w_channel = (a*da_node/(1000**2))**b
        #calculate upstream node spacing, n_widths_defines stable node spacing
        node_spacing = n_widths*w_channel
        # if stable node spacing is greater than raster grid resolution
        if node_spacing > rmg.dx:
            #optimal along-channel node location based upon node spacing
            opt_loc = seg_i['distances'][idx_node] + node_spacing
            # define tolerance to not add extra node if opt loc is within half  
            # a node spacing away from end of segment 
            buffer_tol = 0.5*node_spacing
            # if we can fit another node on the channel segment
            if opt_loc < (seg_i['distances'][-1] - buffer_tol):
                # find id of node closest to this location
                idx_next_node = np.abs(seg_i['distances'] - opt_loc).argmin()
                # update rmg node with whatever this next node should be
                rmg_next_node = seg_i['ids'][idx_next_node]
                
                #LINKS: add link from this upstream node to the current node
                head_node_xy = (rmg.x_of_node[rmg_next_node],
                                rmg.y_of_node[rmg_next_node])
                tail_node_xy = (rmg.x_of_node[rmg_node],
                                rmg.y_of_node[rmg_node])
                # add these node xy locations to our list if we haven't already
                if head_node_xy not in node_xy:
                    node_xy.append(head_node_xy)
                if tail_node_xy not in node_xy:
                    node_xy.append(tail_node_xy)
                # get the index of the head and tail node index.
                head_node__node_id = node_xy.index(head_node_xy)
                tail_node__node_id = node_xy.index(tail_node_xy)
                # append the head and tail node ids to the link array
                links.append((head_node__node_id, tail_node__node_id))

                # update idx_node and rmg node for next loop
                rmg_node = rmg_next_node
                idx_node = idx_next_node
            
            # if no more nodes can be placed on this segment, 
            # move to next segment
            else:
                upstrm_node = False
                # add last node in segment to list of node xys
                last_node_xy = (rmg.x_of_node[rmg_node],
                                rmg.y_of_node[rmg_node])
                if last_node_xy not in node_xy:
                    node_xy.append(last_node_xy)

        # if no more nodes have stable locations on this segment
        # move to next segment
        else:
            upstrm_node = False
            # add last node in segment to list of node xys
            last_node_xy = (rmg.x_of_node[rmg_node],
                            rmg.y_of_node[rmg_node])
            if last_node_xy not in node_xy:
                node_xy.append(last_node_xy)
    
    # store location of network nodes as raster model ids in channel profiler
    # datastructure. this will be used for joining channel segments later
    seg_i['ids_nmg'] = np.array(nmg_nodes)

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
    np.vstack((graph_net.node_at_link_head, graph_net.node_at_link_tail)).T
)

nmg.at_node['rmg_node_value'] = np.array(rmg_nodes)[sorted_nodes]

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


#plotting dev checks
# ncol=58     
# transect = elevs[:, ncol]
# ma_trans = np.ma.masked_where(transect < -9998, transect)
# plt.plot(ma_trans)
# f, ax = plt.subplots()
# imshow_grid_at_node(rmg, z, colorbar_label='Elevation (m)')
# ax.axvline(ncol*rmg.dx)
# plt.show()

plt.figure(figsize = (10,8))
profiler.plot_profiles_in_map_view(colorbar_label='Elevation [m]',shrink=0.65,    color_for_closed=None,output=False)
plt.scatter(x_of_node, y_of_node, s=30, zorder=2)