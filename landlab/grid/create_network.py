# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:16:53 2021

@author: sahrendt
"""
import numpy as np
import pandas as pd #delete after removing pd.drop

from ..components import ChannelProfiler, FlowAccumulator, DepressionFinderAndRouter
from ..graph import NetworkGraph
from .raster import RasterModelGrid
from .network import NetworkModelGrid


def create_network_from_raster(
        rmg, node_spacing=None, min_channel_thresh=10000, fields=None):
#optional arguments to add: 'do_flow_routing'
#add to docstring: minimum channel thresh in units of 'm' 
    
    if node_spacing is None:
        node_spacing = 3*rmg.dx
    
    if 'drainage_area' not in rmg.at_node:
        
        # run flow accumulator for ChannelProfiler
        fa = FlowAccumulator(rmg, 
                             'topographic__elevation',
                             flow_director='D8',
                             depression_finder='DepressionFinderAndRouter')
        fa.run_one_step()
    

    #delinate channel
    profiler = ChannelProfiler(
        rmg,
        number_of_watersheds=1,
        minimum_channel_threshold=min_channel_thresh,
        main_channel_only=False,
    )
    profiler.run_one_step()    
    
    x_of_nodes = np.array([]) #empty list to store x locations of network model grid nodes
    y_of_nodes = np.array([]) #empty list to store y locations of network model grid nodes
    rmg_nodes = np.array([]) #empty list to store raster model grid node corresponding to each network model grid node    
    links = []
    node_xy = []
    nodes_per_segment = [] #empty list to store number of nodes in each channel segment

    
    #obtain watershed key
    ws_keys = [k for k in profiler.data_structure.keys()]
    
    #sent message if there is more than 1 watershed (code is not currently set up to handle multipe watersheds)
    if len(ws_keys) > 1:
        raise ValueError('more than one watershed in DEM, unable to deliniate network properly')
    
    #access number of channel segments
    n_channel_segs = len(profiler.data_structure[ws_keys[0]])
    
    #find the nodes of the start and end index of each segment
    ws_start =[ws_seg[0] for ws_seg in profiler.data_structure[ws_keys[0]].keys()] 
    ws_end = [ws_seg[1] for ws_seg in profiler.data_structure[ws_keys[0]].keys()]
    
    #loop through each segment
    for i, seg_key in enumerate(profiler.data_structure[ws_keys[0]].keys()):
        
        #access individual segment
        seg_i = profiler.data_structure[ws_keys[0]][seg_key] 
        
        # create array of idealized node locations using previously specified distance between nodes
        node_locs = np.arange(seg_i['distances'][0], seg_i['distances'][-1], node_spacing)
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
              
    #get unique nodes
    xy_df = pd.DataFrame({'x': x_of_nodes, 'y': y_of_nodes})
    uniq_x_of_nodes = xy_df.drop_duplicates()['x'].values
    uniq_y_of_nodes = xy_df.drop_duplicates()['y'].values
    
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

    #TODO: add extra fields to network model grid: elevation, RMG node locations,
    # drainage area from flow accumulator
    nmg.at_node['rmg_node_value'] = rmg_nodes[sorted_nodes].astype('int')
    if fields is None:
        fields = []
    for field in fields:
        nmg.at_node[field] = rmg.at_node[field][nmg.at_node['rmg_node_value']]
        
    return(nmg)