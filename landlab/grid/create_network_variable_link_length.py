# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:50:09 2021

@author: sahrendt
"""

import numpy as np

from ..components import ChannelProfiler, FlowAccumulator, DepressionFinderAndRouter
from ..graph import NetworkGraph
from .raster import RasterModelGrid
from .network import NetworkModelGrid

def create_network_from_raster(
        rmg, min_channel_thresh=10000, outlet_nodes=None,
        a=9.68, b=0.32, n_widths=20,
        fields=None):


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
        outlet_nodes=outlet_nodes,
        main_channel_only=False,
    )
    profiler.run_one_step()

    #obtain watershed key (should only be one)
    wtrshd_key = [k for k in profiler.data_structure.keys()][0]
    # obtain keys for channel segments, keys are raster nodes formated as 
    # tuple for (start, end) for channel segment start and end locations
    channel_segment_keys = profiler.data_structure[wtrshd_key].keys()
    
    # IDENTIFY SEGMENT CONNECTIVITY ------------------
    # obtain node ids for start and end of every channel segments
    seg_starts =[seg[0] for seg in profiler.data_structure[wtrshd_key].keys()] 
    seg_ends = [seg[1] for seg in profiler.data_structure[wtrshd_key].keys()]
    
    # identify channel connectivity and how to connect channels
    # currently identifies the key of the channel seg just downstream 
    # and connects first node of upstream channel to downstream channel
    for i, seg_key in enumerate(channel_segment_keys):
        connectivity = []
        connectivity_key = None
        seg_i = profiler.data_structure[wtrshd_key][seg_key]
        # if the start of a segment is the end of another segment
        if seg_key[0] in seg_ends:
            connectivity.append('connected downstream')
            #find first segment downstream that should connect to last node of
            connect_to_channel_idx = np.argmax(seg_key[0]==seg_ends)
            connectivity_key = (seg_starts[connect_to_channel_idx],
                                seg_ends[connect_to_channel_idx])
        # if end of segment is in start of another segment
        if seg_key[-1] in seg_starts:
            connectivity.append('connected upstream')
        
        seg_i['connectivity'] = connectivity
        seg_i['connectivity_key'] = connectivity_key
    
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
                
            #calculate drainage area contributing to this node
            da_node = rmg.at_node['drainage_area'][rmg_node]
            #relate drainage area to river width (convert area to km, width in m)
            # from Frasson et al. 2019 GRL
            w_channel = (a*da_node/(1000**2))**b
            #calculate upstream node spacing, n_widths_defines stable node spacing
            node_spacing = n_widths*w_channel
            # print(node_spacing)
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
                # if we are seeing links on main stem channels that are smaller
                # then grid resolution, flag this and break code
                if 'connected upstream' in seg_i['connectivity']:
                    raise ValueError(
                        'main stem link lengths are smaller than grid res.'\
                        'try increasing n_widths or changing a and b params')
                # add last node in segment to list of node xys
                last_node_xy = (rmg.x_of_node[rmg_node],
                                rmg.y_of_node[rmg_node])
                if last_node_xy not in node_xy:
                    node_xy.append(last_node_xy)
        
        # store location of network nodes as raster model ids in channel profiler
        # datastructure. this will be used for joining channel segments later
        seg_i['ids_nmg'] = np.array(nmg_nodes)
    
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
    
    #add extra fields to network model grid: elevation, RMG node locations,
    # drainage area from flow accumulator
    nmg.at_node['rmg_node_value'] = np.array(rmg_nodes)[sorted_nodes]
    if fields is None:
        fields = []
    for field in fields:
        nmg.at_node[field] = rmg.at_node[field][nmg.at_node['rmg_node_value']]
        
    return(nmg)
    
