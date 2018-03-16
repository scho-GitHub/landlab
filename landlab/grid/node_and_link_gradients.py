#! /usr/bin/env python
"""Calculate gradients of quantities over links.

Gradient calculation functions
++++++++++++++++++++++++++++++

.. autosummary::
    :toctree: generated/

    ~landlab.grid.gradients.calc_grad_at_active_link
    ~landlab.grid.gradients.calc_grad_at_link
    ~landlab.grid.gradients.calculate_gradients_at_faces
    ~landlab.grid.gradients.calculate_diff_at_links
    ~landlab.grid.gradients.calculate_diff_at_active_links

"""

import numpy as np
from landlab.utils.decorators import use_field_name_or_array, deprecated
from landlab.core.utils import radians_to_degrees
from landlab.grid.base import CLOSED_BOUNDARY


@use_field_name_or_array('node')
def calc_grad_at_link(grid, node_values, out=None):
    """Calculate gradients of node values at links.

    Calculates the gradient in `node_values` at each link in the grid,
    returning an array of length `number_of_links`.

    Construction::

        calc_grad_at_link(grid, node_values, out=None)

    Parameters
    ----------
    grid : ModelGrid
        A ModelGrid.
    node_values : ndarray or field name (x number of nodes)
        Values at grid nodes.
    out : ndarray, optional (x number of links)
        Buffer to hold the result.

    Returns
    -------
    ndarray
        Gradients across active links.

    Examples
    --------
    >>> from landlab import RasterModelGrid
    >>> rg = RasterModelGrid(3, 4, 10.0)
    >>> z = rg.add_zeros('node', 'topographic__elevation')
    >>> z[5] = 50.0
    >>> z[6] = 36.0
    >>> calc_grad_at_link(rg, z)  # there are 17 links
    array([ 0. ,  0. ,  0. ,  0. ,  5. ,  3.6,  0. ,  5. , -1.4, -3.6,  0. ,
           -5. , -3.6,  0. ,  0. ,  0. ,  0. ])

    >>> from landlab import HexModelGrid
    >>> hg = HexModelGrid(3, 3, 10.0)
    >>> z = hg.add_zeros('node', 'topographic__elevation', noclobber=False)
    >>> z[4] = 50.0
    >>> z[5] = 36.0
    >>> calc_grad_at_link(hg, z)  # there are 11 faces
    array([ 0. ,  0. ,  0. ,  5. ,  5. ,  3.6,  3.6,  0. ,  5. , -1.4, -3.6,
            0. , -5. , -5. , -3.6, -3.6,  0. ,  0. ,  0. ])

    LLCATS: LINF GRAD
    """
    if out is None:
        out = grid.empty(at='link')
    return np.divide(node_values[grid.node_at_link_head] -
                     node_values[grid.node_at_link_tail],
                     grid.length_of_link, out=out)


@deprecated(use='calc_grad_at_link', version='1.0beta')
def calc_grad_of_active_link(grid, node_values, out=None):
    """Calculate gradients at active links.

    Examples
    --------
    >>> import numpy as np
    >>> from landlab import RasterModelGrid
    >>> grid = RasterModelGrid((3, 4))
    >>> z = np.array([0., 0., 0., 0.,
    ...               1., 1., 1., 1.,
    ...               3., 3., 3., 3.])
    >>> grid.calc_grad_of_active_link(z)
    array([ 1.,  1.,  0.,  0.,  0.,  2.,  2.])

    This method is *deprecated*. Instead, use ``calc_grad_at_link``.

    >>> vals = grid.calc_grad_at_link(z)
    >>> vals[grid.active_links]
    array([ 1.,  1.,  0.,  0.,  0.,  2.,  2.])

    LLCATS: DEPR
    """
    return calc_grad_at_active_link(grid, node_values, out)


@deprecated(use='calc_grad_at_link', version='1.0beta')
@use_field_name_or_array('node')
def calc_grad_at_active_link(grid, node_values, out=None):
    """Calculate gradients of node values over active links.

    Calculates the gradient in *quantity* node values at each active link in
    the grid.

    Construction::

        calc_grad_at_active_link(grid, node_values, out=None)

    Parameters
    ----------
    grid : ModelGrid
        A ModelGrid.
    node_values : ndarray or field name
        Values at grid nodes.
    out : ndarray, optional
        Buffer to hold the result.

    Returns
    -------
    ndarray
        Gradients across active links.

    LLCATS: DEPR LINF GRAD
    """
    if out is None:
        out = np.empty(len(grid.active_links), dtype=float)
    return np.divide(
        np.diff(node_values[grid.nodes_at_link[grid.active_links]], axis=1).flatten(),
        grid.length_of_link[grid.active_links], out=out)


@use_field_name_or_array('node')
def calc_diff_at_link(grid, node_values, out=None):
    """Calculate differences of node values over links.

    Calculates the difference in quantity *node_values* at each link in the
    grid.

    Construction::

        calc_diff_at_link(grid, node_values, out=None)

    Parameters
    ----------
    grid : ModelGrid
        A ModelGrid.
    node_values : ndarray or field name
        Values at grid nodes.
    out : ndarray, optional
        Buffer to hold the result.

    Returns
    -------
    ndarray
        Differences across links.

    Examples
    --------
    >>> import numpy as np
    >>> from landlab import RasterModelGrid
    >>> rmg = RasterModelGrid((3, 3))
    >>> z = np.zeros(9)
    >>> z[4] = 1.
    >>> rmg.calc_diff_at_link(z)
    array([ 0.,  0.,  0.,  1.,  0.,  1., -1.,  0., -1.,  0.,  0.,  0.])

    LLCATS: LINF GRAD
    """
    if out is None:
        out = grid.empty(at='link')
    node_values = np.asarray(node_values)
    return np.subtract(node_values[grid.node_at_link_head],
                       node_values[grid.node_at_link_tail], out=out)


@deprecated(use='calc_diff_at_link', version='1.0beta')
@use_field_name_or_array('node')
def calculate_diff_at_links(grid, node_values, out=None):
    """Calculate differences of node values over links.

    Examples
    --------
    >>> import numpy as np
    >>> from landlab import RasterModelGrid

    >>> grid = RasterModelGrid((3, 3))
    >>> z = np.zeros(9)
    >>> z[4] = 1.

    >>> grid.calculate_diff_at_links(z)
    array([ 0.,  0.,  0.,  1.,  0.,  1., -1.,  0., -1.,  0.,  0.,  0.])

    >>> grid.calc_diff_at_link(z)
    array([ 0.,  0.,  0.,  1.,  0.,  1., -1.,  0., -1.,  0.,  0.,  0.])

    LLCATS: DEPR LINF GRAD
    """
    return calc_diff_at_link(grid, node_values, out)


@deprecated(use='calc_diff_at_link', version='1.0beta')
@use_field_name_or_array('node')
def calculate_diff_at_active_links(grid, node_values, out=None):
    """Calculate differences of node values over active links.

    Calculates the difference in quantity *node_values* at each active link
    in the grid.

    Construction::

        calculate_diff_at_active_links(grid, node_values, out=None)

    Parameters
    ----------
    grid : ModelGrid
        A ModelGrid.
    node_values : ndarray or field name
        Values at grid nodes.
    out : ndarray, optional
        Buffer to hold the result.

    Returns
    -------
    ndarray
        Differences across active links.

    LLCATS: DEPR LINF GRAD
    """
    if out is None:
        out = np.empty(len(grid.active_links), dtype=float)
    node_values = np.asarray(node_values)
    node_values = node_values[grid.nodes_at_link[grid.active_links]]
    return np.subtract(node_values[:, 1], node_values[:, 0], out=out)
