#! /usr/bin/env python
"""Map values from one grid element to another.

Grid mapping functions
+++++++++++++++++++++++

.. autosummary::
    :toctree: generated/

    ~landlab.grid.mappers.map_link_head_node_to_link
    ~landlab.grid.mappers.map_link_tail_node_to_link
    ~landlab.grid.mappers.map_min_of_link_nodes_to_link
    ~landlab.grid.mappers.map_max_of_link_nodes_to_link
    ~landlab.grid.mappers.map_mean_of_link_nodes_to_link
    ~landlab.grid.mappers.map_value_at_min_node_to_link
    ~landlab.grid.mappers.map_value_at_max_node_to_link
    ~landlab.grid.mappers.map_node_to_cell
    ~landlab.grid.mappers.map_min_of_node_links_to_node
    ~landlab.grid.mappers.map_max_of_node_links_to_node
    ~landlab.grid.mappers.map_upwind_node_link_max_to_node
    ~landlab.grid.mappers.map_downwind_node_link_max_to_node
    ~landlab.grid.mappers.map_upwind_node_link_mean_to_node
    ~landlab.grid.mappers.map_downwind_node_link_mean_to_node
    ~landlab.grid.mappers.map_value_at_upwind_node_link_max_to_node
    ~landlab.grid.mappers.map_value_at_downwind_node_link_max_to_node
    ~landlab.grid.mappers.dummy_func_to_demonstrate_docstring_modification

Each link has a *tail* and *head* node. The *tail* nodes are located at the
start of a link, while the head nodes are located at end of a link.

Below, the numbering scheme for links in `RasterModelGrid` is illustrated
with an example of a four-row by five column grid (4x5). In this example,
each * (or X) is a node, the lines represent links, and the ^ and > symbols
indicate the direction and *head* of each link. Link heads in the
`RasterModelGrid` always point in the cardinal directions North (N) or East
(E).::

    *--27-->*--28-->*--29-->*--30-->*
    ^       ^       ^       ^       ^
   22      23      24      25      26
    |       |       |       |       |
    *--18-->*--19-->*--20-->*--21-->*
    ^       ^       ^       ^       ^
    13      14      15      16     17
    |       |       |       |       |
    *---9-->*--10-->X--11-->*--12-->*
    ^       ^       ^       ^       ^
    4       5       6       7       8
    |       |       |       |       |
    *--0--->*---1-->*--2--->*---3-->*

For example, node 'X' has four link-neighbors. From south and going clockwise,
these neighbors are [6, 10, 15, 11]. Both link 6 and link 10 have node 'X' as
their 'head' node, while links 15 and 11 have node 'X' as their tail node.
"""
from __future__ import division

import numpy as np
from landlab.grid.base import BAD_INDEX_VALUE, CLOSED_BOUNDARY, INACTIVE_LINK


def map_node_to_cell(grid, var_name, out=None):
    """Map values for nodes to cells.

    map_node_to_cell iterates across the grid and
    identifies the all node values of 'var_name'.

    This function takes node values of 'var_name' and mapes that value to the
    corresponding cell area for each node.

    Parameters
    ----------
    grid : ModelGrid
        A landlab ModelGrid.
    var_name : array or field name
        Values defined at nodes.
    out : ndarray, optional
        Buffer to place mapped values into or `None` to create a new array.

    Returns
    -------
    ndarray
        Mapped values at cells.

    Examples
    --------
    >>> import numpy as np
    >>> from landlab.grid.mappers import map_node_to_cell
    >>> from landlab import RasterModelGrid

    >>> rmg = RasterModelGrid((3, 4))
    >>> _ = rmg.add_field('node', 'z', np.arange(12.))
    >>> map_node_to_cell(rmg, 'z')
    array([ 5.,  6.])

    >>> values_at_cells = rmg.empty(at='cell')
    >>> rtn = map_node_to_cell(rmg, 'z', out=values_at_cells)
    >>> values_at_cells
    array([ 5.,  6.])
    >>> rtn is values_at_cells
    True

    LLCATS: CINF NINF MAP
    """
    if out is None:
        out = grid.empty(at='cell')

    if type(var_name) is str:
        var_name = grid.at_node[var_name]
    out[:] = var_name[grid.node_at_cell]

    return out


def map_mean_of_patch_nodes_to_patch(grid, var_name, ignore_closed_nodes=True,
                                     out=None):
    """Map the mean value of nodes around a patch to the patch.

    Parameters
    ----------
    grid : ModelGrid
        A landlab ModelGrid.
    var_name : array or field name
        Values defined at nodes.
    ignore_closed_nodes : bool
        If True, do not incorporate closed nodes into calc. If all nodes are
        masked at a patch, record zero if out is None or leave the existing
        value if out.
    out : ndarray, optional
        Buffer to place mapped values into or `None` to create a new array.

    Returns
    -------
    ndarray
        Mapped values at patches.

    Examples
    --------
    >>> import numpy as np
    >>> from landlab.grid.mappers import map_mean_of_patch_nodes_to_patch
    >>> from landlab import RasterModelGrid, CLOSED_BOUNDARY

    >>> rmg = RasterModelGrid((3, 4))
    >>> rmg.at_node['vals'] = np.array([5., 4., 3., 2.,
    ...                                 5., 4., 3., 2.,
    ...                                 3., 2., 1., 0.])
    >>> map_mean_of_patch_nodes_to_patch(rmg, 'vals')
    array([ 4.5, 3.5, 2.5,
            3.5, 2.5, 1.5])

    >>> rmg.at_node['vals'] = np.array([5., 4., 3., 2.,
    ...                                 5., 4., 3., 2.,
    ...                                 3., 2., 1., 0.])
    >>> rmg.status_at_node[rmg.node_x > 1.5] = CLOSED_BOUNDARY
    >>> ans = np.zeros(6, dtype=float)
    >>> _ = map_mean_of_patch_nodes_to_patch(rmg, 'vals', out=ans)
    >>> ans # doctest: +NORMALIZE_WHITESPACE
    array([ 4.5, 4. , 0. ,
            3.5, 3. , 0. ])

    LLCATS: PINF NINF MAP
    """
    if out is None:
        out = np.zeros(grid.number_of_patches, dtype=float)

    if type(var_name) is str:
        var_name = grid.at_node[var_name]
    values_at_nodes = var_name[grid.nodes_at_patch]
    if ignore_closed_nodes:
        values_at_nodes = np.ma.masked_where(grid.status_at_node[
            grid.nodes_at_patch] == CLOSED_BOUNDARY,
            values_at_nodes, copy=False)
        meanvals = np.mean(values_at_nodes, axis=1)
        if type(meanvals.mask) is not np.bool_:
            gooddata = np.logical_not(meanvals.mask)
            out[gooddata] = meanvals.data[gooddata]
        else:
            if not meanvals.mask:
                out[:] = meanvals.data
    else:
        np.mean(values_at_nodes, axis=1, out=out)

    return out


def map_max_of_patch_nodes_to_patch(grid, var_name, ignore_closed_nodes=True,
                                    out=None):
    """Map the maximum value of nodes around a patch to the patch.

    Parameters
    ----------
    grid : ModelGrid
        A landlab ModelGrid.
    var_name : array or field name
        Values defined at nodes.
    ignore_closed_nodes : bool
        If True, do not incorporate closed nodes into calc. If all nodes are
        masked at a patch, record zero if out is None or leave the existing
        value if out.
    out : ndarray, optional
        Buffer to place mapped values into or `None` to create a new array.

    Returns
    -------
    ndarray
        Mapped values at patches.

    Examples
    --------
    >>> import numpy as np
    >>> from landlab.grid.mappers import map_max_of_patch_nodes_to_patch
    >>> from landlab import RasterModelGrid, CLOSED_BOUNDARY

    >>> rmg = RasterModelGrid((3, 4))
    >>> rmg.at_node['vals'] = np.array([5., 4., 3., 2.,
    ...                                 3., 4., 3., 2.,
    ...                                 3., 2., 1., 0.])
    >>> map_max_of_patch_nodes_to_patch(rmg, 'vals')
    array([ 5., 4., 3.,
            4., 4., 3.])

    >>> rmg.at_node['vals'] = np.array([5., 4., 3., 2.,
    ...                                 3., 4., 3., 2.,
    ...                                 3., 2., 1., 0.])
    >>> rmg.status_at_node[rmg.node_x > 1.5] = CLOSED_BOUNDARY
    >>> ans = np.zeros(6, dtype=float)
    >>> _ = map_max_of_patch_nodes_to_patch(rmg, 'vals', out=ans)
    >>> ans # doctest: +NORMALIZE_WHITESPACE
    array([ 5., 4., 0.,
            4., 4., 0.])

    LLCATS: PINF NINF MAP
    """
    if out is None:
        out = np.zeros(grid.number_of_patches, dtype=float)

    if type(var_name) is str:
        var_name = grid.at_node[var_name]
    values_at_nodes = var_name[grid.nodes_at_patch]
    if ignore_closed_nodes:
        values_at_nodes = np.ma.masked_where(grid.status_at_node[
            grid.nodes_at_patch] == CLOSED_BOUNDARY,
            values_at_nodes, copy=False)
        maxvals = values_at_nodes.max(axis=1)
        if type(maxvals.mask) is not np.bool_:
            gooddata = np.logical_not(maxvals.mask)
            out[gooddata] = maxvals.data[gooddata]
        else:
            if not maxvals.mask:
                out[:] = maxvals.data
    else:
        np.amax(values_at_nodes, axis=1, out=out)

    return out


def map_min_of_patch_nodes_to_patch(grid, var_name, ignore_closed_nodes=True,
                                    out=None):
    """Map the minimum value of nodes around a patch to the patch.

    Parameters
    ----------
    grid : ModelGrid
        A landlab ModelGrid.
    var_name : array or field name
        Values defined at nodes.
    ignore_closed_nodes : bool
        If True, do not incorporate closed nodes into calc. If all nodes are
        masked at a patch, record zero if out is None or leave the existing
        value if out.
    out : ndarray, optional
        Buffer to place mapped values into or `None` to create a new array.

    Returns
    -------
    ndarray
        Mapped values at patches.

    Examples
    --------
    >>> import numpy as np
    >>> from landlab.grid.mappers import map_min_of_patch_nodes_to_patch
    >>> from landlab import RasterModelGrid, CLOSED_BOUNDARY

    >>> rmg = RasterModelGrid((3, 4))
    >>> rmg.at_node['vals'] = np.array([5., 4., 3., 2.,
    ...                                 5., 4., 3., 2.,
    ...                                 3., 2., 1., 0.])
    >>> map_min_of_patch_nodes_to_patch(rmg, 'vals')
    array([ 4., 3., 2.,
            2., 1., 0.])

    >>> rmg.at_node['vals'] = np.array([5., 4., 3., 2.,
    ...                                 5., 4., 3., 2.,
    ...                                 3., 2., 1., 0.])
    >>> rmg.status_at_node[rmg.node_x > 1.5] = CLOSED_BOUNDARY
    >>> ans = np.zeros(6, dtype=float)
    >>> _ = map_min_of_patch_nodes_to_patch(rmg, 'vals', out=ans)
    >>> ans # doctest: +NORMALIZE_WHITESPACE
    array([ 4., 4., 0.,
            2., 2., 0.])

    LLCATS: PINF NINF MAP
    """
    if out is None:
        out = np.zeros(grid.number_of_patches, dtype=float)

    if type(var_name) is str:
        var_name = grid.at_node[var_name]
    values_at_nodes = var_name[grid.nodes_at_patch]
    if ignore_closed_nodes:
        values_at_nodes = np.ma.masked_where(grid.status_at_node[
            grid.nodes_at_patch] == CLOSED_BOUNDARY,
            values_at_nodes, copy=False)
        minvals = values_at_nodes.min(axis=1)
        if type(minvals.mask) is not np.bool_:
            gooddata = np.logical_not(minvals.mask)
            out[gooddata] = minvals.data[gooddata]
        else:
            if not minvals.mask:
                out[:] = minvals.data
    else:
        np.amin(values_at_nodes, axis=1, out=out)

    return out


def map_link_vector_sum_to_patch(grid, var_name, ignore_inactive_links=True,
                                 out=None):
    """Map the vector sum of links around a patch to the patch.

    The resulting vector is returned as a length-2 list, with the two
    items being arrays of the x component and the y component of the resolved
    vectors at the patches, respectively.

    Parameters
    ----------
    grid : ModelGrid
        A landlab ModelGrid.
    var_name : array or field name
        Values defined at links.
    ignore_inactive_links : bool
        If True, do not incorporate inactive links into calc. If all links are
        inactive at a patch, record zero if out is None or leave the existing
        value if out.
    out : len-2 list of npatches-long arrays, optional
        Buffer to place mapped values into or `None` to create a new array.

    Returns
    -------
    len-2 list of arrays
        [x_component_of_link_vals_at_patch, y_component_of_link_vals_at_patch].

    Examples
    --------
    >>> import numpy as np
    >>> from landlab.grid.mappers import map_link_vector_sum_to_patch
    >>> from landlab import HexModelGrid
    >>> from landlab import CLOSED_BOUNDARY, CORE_NODE, INACTIVE_LINK

    >>> mg = HexModelGrid(4, 3)
    >>> interior_nodes = mg.status_at_node == CORE_NODE
    >>> exterior_nodes = mg.status_at_node != CORE_NODE

    Add a ring of closed nodes at the edge:

    >>> mg.status_at_node[exterior_nodes] = CLOSED_BOUNDARY

    This gives us 5 core nodes, 7 active links, and 3 present patches

    >>> (mg.number_of_core_nodes == 5 and mg.number_of_active_links == 7)
    True
    >>> A = mg.add_ones('link', 'vals')
    >>> A.fill(9.)  # any old values on the inactive links
    >>> A[mg.active_links] = np.array([ 1., -1.,  1., -1., -1., -1., -1.])

    This setup should give present patch 0 pure east, patch 1 zero (vorticity),
    and patch 2 westwards and downwards components.

    >>> xcomp, ycomp = map_link_vector_sum_to_patch(mg, 'vals')
    >>> np.allclose(xcomp[[6, 9, 10]], [2., 0., -1])
    True
    >>> np.allclose(ycomp[[6, 9, 10]]/np.sqrt(3.), [0., 0., -1.])
    True

    These are the patches with INACTIVE_LINKs on all three sides:

    >>> absent_patches = np.array([0, 1, 2, 4, 8, 11, 12, 15, 16, 17, 18])
    >>> np.allclose(xcomp[absent_patches], 0.)
    True
    >>> np.allclose(ycomp[absent_patches], 0.)
    True

    Now demonstrate the remaining functionality:

    >>> A = mg.at_link['vals'].copy()
    >>> A.fill(1.)
    >>> _ = map_link_vector_sum_to_patch(mg, A, ignore_inactive_links=False,
    ...                                  out=[xcomp, ycomp])
    >>> np.allclose(xcomp[absent_patches], 0.)
    False
    >>> np.allclose(ycomp[absent_patches], 0.)
    False

    LLCATS: PINF LINF MAP
    """
    if out is None:
        out = [np.zeros(grid.number_of_patches, dtype=float),
               np.zeros(grid.number_of_patches, dtype=float)]
    else:
        assert len(out) == 2

    if type(var_name) is str:
        var_name = grid.at_link[var_name]
    angles_at_links = grid.angle_of_link  # CCW round tail
    hoz_cpt = np.cos(angles_at_links)
    vert_cpt = np.sin(angles_at_links)
    hoz_vals = var_name * hoz_cpt
    vert_vals = var_name * vert_cpt
    hoz_vals_at_patches = hoz_vals[grid.links_at_patch]
    vert_vals_at_patches = vert_vals[grid.links_at_patch]
    if ignore_inactive_links:
        linkmask = grid.status_at_link[grid.links_at_patch] == INACTIVE_LINK
        hoz_vals_at_patches = np.ma.array(hoz_vals_at_patches, mask=linkmask,
                                          copy=False)
        vert_vals_at_patches = np.ma.array(vert_vals_at_patches, mask=linkmask,
                                           copy=False)
        hoz_sum = np.sum(hoz_vals_at_patches, axis=1)
        vert_sum = np.sum(vert_vals_at_patches, axis=1)
        if type(hoz_sum.mask) is not np.bool_:  # the 2 comps have same mask
            gooddata = np.logical_not(hoz_sum.mask)
            out[0][gooddata] = hoz_sum.data[gooddata]
            out[1][gooddata] = vert_sum.data[gooddata]
        else:
            if not hoz_sum.mask:
                out[0][:] = hoz_sum.data
                out[1][:] = vert_sum.data

    else:
        hoz_sum = np.sum(hoz_vals_at_patches, axis=1, out=out[0])
        vert_sum = np.sum(vert_vals_at_patches, axis=1, out=out[1])

    return out


def dummy_func_to_demonstrate_docstring_modification(grid, some_arg):
    """A dummy function to demonstrate automated docstring changes.

    Parameters
    ----------
    grid : ModelGrid
        A Landlab modelgrid.
    some_arg : whatever
        A dummy argument.

    Examples
    --------
    ...

    LLCATS: DEPR MAP
    """
    pass
