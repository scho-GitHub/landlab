#! /usr/env/python
"""
Python implementation of ModelGrid, a base class used to create and manage
grids for 2D numerical models.

Do NOT add new documentation here. Grid documentation is now built in a semi-
automated fashion. To modify the text seen on the web, edit the files
`docs/text_for_[gridfile].py.txt`.
"""

import numpy
import numpy as np
import warnings
from time import time

import six
from six.moves import range

from landlab.grid.skeleton import SkeletonGrid
from landlab.testing.decorators import track_this_method
from landlab.utils import count_repeated_values
from landlab.core.utils import argsort_points_by_x_then_y
from landlab.utils.decorators import make_return_array_immutable, deprecated
from landlab.field import ModelDataFields, ModelDataFieldsMixIn
from landlab.field.scalar_data_fields import FieldError
from . import grid_funcs as gfuncs
from ..core.utils import as_id_array
from ..core.utils import add_module_functions_to_class
from .decorators import (override_array_setitem_and_reset, return_id_array,
                         return_readonly_id_array)
from ..utils.decorators import cache_result_in_object
from ..layers.eventlayers import EventLayersMixIn
from .nodestatus import (CORE_NODE, FIXED_VALUE_BOUNDARY,
                         FIXED_GRADIENT_BOUNDARY, LOOPED_BOUNDARY,
                         CLOSED_BOUNDARY)
from .linkstatus import ACTIVE_LINK, FIXED_LINK, INACTIVE_LINK
from .linkstatus import set_status_at_link

#: Indicates an index is, in some way, *bad*.
BAD_INDEX_VALUE = -1
# DEJH thinks the user should be able to override this value if they want

# Map names grid elements to the ModelGrid attribute that contains the count
# of that element in the grid.
_ARRAY_LENGTH_ATTRIBUTES = {
    'node': 'number_of_nodes',
    'patch': 'number_of_patches',
    'link': 'number_of_links',
    'corner': 'number_of_corners',
    'face': 'number_of_faces',
    'cell': 'number_of_cells',
    'active_link': 'number_of_active_links',
    'active_face': 'number_of_active_faces',
    'core_node': 'number_of_core_nodes',
    'core_cell': 'number_of_core_cells',
}

# Fields whose sizes can not change.
_SIZED_FIELDS = {'node', 'link', 'patch', 'corner', 'face', 'cell', }


def _sort_points_into_quadrants(x, y, nodes):
    """Divide x, y points into quadrants.

    Divide points with locations given in the *x*, and *y* arrays into north,
    south, east, and west quadrants. Returns nodes contained in quadrants
    (west, east, north, south).

    Parameters
    ----------
    x : array_like
        X-coordinates of points.
    y : array_like
        Y-coordinates of points.
    nodes : array_like
        Nodes associated with points.

    Returns
    -------
    tuple of array_like
        Tuple of nodes in each coordinate. Nodes are grouped as
        (*east*, *north*, *west*, *south*).

    Examples
    --------
    >>> import numpy as np
    >>> from landlab.grid.base import _sort_points_into_quadrants
    >>> x = np.array([0, 1, 0, -1])
    >>> y = np.array([1, 0, -1, 0])
    >>> nodes = np.array([1, 2, 3, 4])
    >>> _sort_points_into_quadrants(x, y, nodes)
    (array([2]), array([1]), array([4]), array([3]))
    """
    above_x_axis = y > 0
    right_of_y_axis = x > 0
    closer_to_y_axis = numpy.abs(y) >= numpy.abs(x)

    north_nodes = nodes[above_x_axis & closer_to_y_axis]
    south_nodes = nodes[(~ above_x_axis) & closer_to_y_axis]
    east_nodes = nodes[right_of_y_axis & (~ closer_to_y_axis)]
    west_nodes = nodes[(~ right_of_y_axis) & (~ closer_to_y_axis)]

    return (east_nodes, north_nodes, west_nodes, south_nodes)

def find_true_vector_from_link_vector_pair(L1, L2, b1x, b1y, b2x, b2y):
    r"""Separate a pair of links with vector values into x and y components.

    The concept here is that a pair of adjacent links attached to a node are
    projections of a 'true' but unknown vector. This function finds and returns
    the x and y components of this true vector. The trivial case is the
    situation in which the two links are orthogonal and aligned with the grid
    axes, in which case the vectors of these two links *are* the x and y
    components.

    Parameters
    ----------
    L1, L2 : float
        Values (magnitudes) associated with the two links
    b1x, b1y, b2x, b2y : float
        Unit vectors of the two links

    Returns
    -------
    ax, ay : float
        x and y components of the 'true' vector

    Notes
    -----
    The function does an inverse vector projection. Suppose we have a given
    'true' vector :math:`a`, and we want to project it onto two other lines
    with unit vectors (b1x,b1y) and (b2x,b2y). In the context of Landlab,
    the 'true' vector is some unknown vector quantity, which might for
    example represent the local water flow velocity. The lines represent two
    adjacent links in the grid.

    Let :math:`\mathbf{a}` be the true vector, :math:`\mathbf{B}` be a
    different vector with unit vector :math:`\mathbf{b}`, and :math:`L`
    be the scalar projection of *a* onto *B*. Then,

    ..math::

        L = \mathbf{a} \dot \mathbf{b} = a_x b_x + a_y b_y,

    where :math:`(a_x,a_y)` are the components of **a** and :math:`(b_x,b_y)`
    are the components of the unit vector **b**.

    In this case, we know *b* (the link unit vector), and we want to know the
    *x* and *y* components of **a**. The problem is that we have one equation
    and two unknowns (:math:`a_x` and :math:`a_y`). But we can solve this if
    we have *two* vectors, both of which are projections of **a**. Using the
    subscripts 1 and 2 to denote the two vectors, we can obtain equations for
    both :math:`a_x` and :math:`a_y`:

    ..math::

        a_x = L_1 / b_{1x} - a_y b_{1y} / b_{1x}

        a_y = L_2 / b_{2y} - a_x b_{2x} / b_{2y}

    Substituting the second into the first,

    ..math::

        a_x = [L_1/b_{1x}-L_2 b_{1y}/(b_{1x} b_{2y})] / [1-b_{1y} b_{2x}/(b_{1x} b_{2y})]

    Hence, we find the original vector :math:`(a_x,a_y)` from two links with
    unit vectors :math:`(b_{1x},b_{1y})` and :math:`(b_{2x},b_{2y})` and
    associated values :math:`L_1` and :math:`L_2`.

    Note that the above equations require that :math:`b_{1x}>0` and
    :math:`b_{2y}>0`. If this isn't the case, we invert the order of the two
    links, which requires :math:`b_{2x}>0` and :math:`b_{1y}>0`. If none of
    these conditions is met, then we have a degenerate case.

    Examples
    --------
    The following example represents the active links in a 7-node hexagonal
    grid, with just one core node. The 'true' vector has a magnitude of 5 units
    and an orientation of 30 degrees, pointing up and to the right (i.e., the
    postive-x and postive-y quadrant), so that its vector components are 4 (x)
    and 3 (y) (in other words, it is a 3-4-5 triangle). The values assigned to
    L below are the projection of that true vector onto the six link
    vectors. The algorithm should recover the correct vector component
    values of 4 and 3. The FOR loop examines each pair of links in turn.

    >>> import numpy as np
    >>> from landlab.grid.base import find_true_vector_from_link_vector_pair
    >>> bx = np.array([0.5, -0.5, -1., -0.5, 1., 0.5])
    >>> by = np.array([0.866, 0.866, 0., -0.866, 0., -0.866])
    >>> L = np.array([4.6, 0.6, -4., -4.6, 4., -0.6])
    >>> for i in range(5):
    ...     ax, ay = find_true_vector_from_link_vector_pair(
    ...         L[i], L[i+1], bx[i], by[i], bx[i+1], by[i+1])
    ...     round(ax,1), round(ay,1)
    (4.0, 3.0)
    (4.0, 3.0)
    (4.0, 3.0)
    (4.0, 3.0)
    (4.0, 3.0)
    """
    assert ((b1x != 0 and b2y != 0) or (b2x != 0 and b1y != 0)), \
        'Improper unit vectors'

    if b1x != 0. and b2y != 0.:
        ax = (L1 / b1x - L2 * (b1y / (b1x * b2y))) / \
            (1. - (b1y * b2x) / (b1x * b2y))
        ay = L2 / b2y - ax * (b2x / b2y)
    elif b2x != 0. and b1y != 0.:
        ax = (L2 / b2x - L1 * (b2y / (b2x * b1y))) / \
            (1. - (b2y * b1x) / (b2x * b1y))
        ay = L1 / b1y - ax * (b1x / b1y)

    return ax, ay


class ModelGrid(SkeletonGrid):
    """Base class for 2D structured or unstructured grids for numerical models.

    The idea is to have at least two inherited
    classes, RasterModelGrid and DelaunayModelGrid, that can create and
    manage grids. To this might be added a GenericModelGrid, which would
    be an unstructured polygonal grid that doesn't necessarily obey or
    understand the Delaunay triangulation, but rather simply accepts
    an input grid from the user. Also a :class:`~.HexModelGrid` for hexagonal.

    Attributes
    ----------
    at_node : dict-like
        Values at nodes.
    at_cell : dict-like
        Values at cells.
    at_link : dict-like
        Values at links.
    at_face : dict-like
        Values at faces.
    at_grid: dict-like
        Global values
    Other Parameters
    ----------------
    axis_name : tuple, optional
        Name of axes
    axis_units : tuple, optional
        Units of coordinates
    """
    BC_NODE_IS_CORE = CORE_NODE
    BC_NODE_IS_FIXED_VALUE = FIXED_VALUE_BOUNDARY
    BC_NODE_IS_FIXED_GRADIENT = FIXED_GRADIENT_BOUNDARY
    BC_NODE_IS_LOOPED = LOOPED_BOUNDARY
    BC_NODE_IS_CLOSED = CLOSED_BOUNDARY

    BC_LINK_IS_ACTIVE = ACTIVE_LINK
    BC_LINK_IS_FIXED = FIXED_LINK
    BC_LINK_IS_INACTIVE = INACTIVE_LINK

    # Debugging flags (if True, activates some output statements)
    _DEBUG_VERBOSE = False
    _DEBUG_TRACK_METHODS = False

    at_node = {}  # : Values defined at nodes
    at_link = {}  # : Values defined at links
    at_patch = {}  # : Values defined at patches
    at_corner = {}  # : Values defined at corners
    at_face = {}  # : Values defined at faces
    at_cell = {}  # : Values defined at cells

    def __init__(self, **kwds):
        super(ModelGrid, self).__init__()

        self._link_length = None
        self._all_node_distances_map = None
        self._all_node_azimuths_map = None
        self.bc_set_code = 0

        # Sort links according to the x and y coordinates of their midpoints.
        # Assumes 1) node_at_link_tail and node_at_link_head have been
        # created, and 2) so have node_x and node_y.
        # self._sort_links_by_midpoint()

        for loc in _SIZED_FIELDS:
            if loc not in ('node', 'link'):
                size = self.number_of_elements(loc)
                ModelDataFields.new_field_location(self, loc, size=size)
    
    def number_of_elements(self, name):
        """Number of instances of an element.

        Get the number of instances of a grid element in a grid.

        Parameters
        ----------
        name : {'node', 'cell', 'link', 'face', 'core_node', 'core_cell',
                'active_link', 'active_face'}
            Name of the grid element.

        Returns
        -------
        int
            Number of elements in the grid.

        Examples
        --------
        >>> from landlab import RasterModelGrid, CLOSED_BOUNDARY
        >>> mg = RasterModelGrid((4, 5), 1.)
        >>> mg.number_of_elements('node')
        20
        >>> mg.number_of_elements('core_cell')
        6
        >>> mg.number_of_elements('link')
        31
        >>> mg.number_of_elements('active_link')
        17
        >>> mg.status_at_node[8] = CLOSED_BOUNDARY
        >>> mg.number_of_elements('link')
        31
        >>> mg.number_of_elements('active_link')
        13

        LLCATS: GINF
        """
        try:
            return getattr(self, _ARRAY_LENGTH_ATTRIBUTES[name])
        except KeyError:
            raise TypeError(
                '{name}: element name not understood'.format(name=name))
            
    @property
    def node_at_cell(self):
        """Node ID associated with grid cells.

        Examples
        --------
        >>> from landlab import RasterModelGrid, BAD_INDEX_VALUE
        >>> grid = RasterModelGrid((4, 5))
        >>> grid.node_at_cell # doctest: +NORMALIZE_WHITESPACE
        array([ 6,  7,  8,
               11, 12, 13])

        LLCATS: NINF CINF CONN
        """
        return self._node_at_cell

    @property
    def cell_at_node(self):
        """Node ID associated with grid cells.

        Examples
        --------
        >>> from landlab import RasterModelGrid, BAD_INDEX_VALUE
        >>> grid = RasterModelGrid((4, 5))
        >>> ids = grid.cell_at_node
        >>> ids[ids == BAD_INDEX_VALUE] = -1
        >>> ids # doctest: +NORMALIZE_WHITESPACE
        array([-1, -1, -1, -1, -1,
               -1,  0,  1,  2, -1,
               -1,  3,  4,  5, -1,
               -1, -1, -1, -1, -1])

        LLCATS: CINF NINF CONN
        """
        return self._cell_at_node

    @property
    @return_readonly_id_array
    @cache_result_in_object()
    def active_faces(self):
        """Get array of active faces.

        Examples
        --------
        >>> from landlab import RasterModelGrid
        >>> grid = RasterModelGrid((3, 4))
        >>> grid.active_faces
        array([0, 1, 2, 3, 4, 5, 6])

        >>> from landlab import CLOSED_BOUNDARY
        >>> grid.status_at_node[6] = CLOSED_BOUNDARY
        >>> grid.active_faces
        array([0, 2, 5])

        LLCATS: FINF BC
        """
        return self.face_at_link[self.active_links]

    @property
    @cache_result_in_object()
    @return_readonly_id_array
    def node_at_core_cell(self):
        """Get array of nodes associated with core cells.

        Examples
        --------
        >>> from landlab import RasterModelGrid
        >>> grid = RasterModelGrid((4, 5), 1.)

        Initially each cell's node is core.

        >>> grid.node_at_core_cell
        array([ 6,  7,  8,
               11, 12, 13])

        Setting a node to closed causes means its cell is also
        "closed".

        >>> grid.status_at_node[8] = grid.BC_NODE_IS_CLOSED
        >>> grid.node_at_core_cell
        array([ 6,  7, 11, 12, 13])

        LLCATS: NINF CINF BC CONN
        """
        return numpy.where(self.status_at_node == CORE_NODE)[0]

    @property
    @make_return_array_immutable
    @cache_result_in_object()
    def core_cells(self):
        """Get array of core cells.

        Examples
        --------
        >>> from landlab import RasterModelGrid
        >>> grid = RasterModelGrid((4, 5), 1.)

        Initially all of the cells are "core".

        >>> grid.core_cells
        array([0, 1, 2,
               3, 4, 5])

        Setting a node to closed causes its cell to no longer be core.

        >>> grid.status_at_node[8] = grid.BC_NODE_IS_CLOSED
        >>> grid.core_cells
        array([0, 1, 3, 4, 5])

        LLCATS: CINF BC
        """
        return self.cell_at_node[self.core_nodes]

    @property
    def face_at_link(self):
        """Get array of faces associated with links.

        Examples
        --------
        >>> import numpy as np
        >>> from landlab import RasterModelGrid, BAD_INDEX_VALUE
        >>> mg = RasterModelGrid((4, 5), 1.)
        >>> mg.face_at_link[5:7]
        array([0, 1])
        >>> np.all(mg.face_at_link[:5]==BAD_INDEX_VALUE)
        True

        LLCATS: FINF LINF CONN
        """
        try:
            return self._face_at_link
        except AttributeError:
            return self._create_face_at_link()

    @property
    def link_at_face(self):
        """Get array of links associated with faces.

        Examples
        --------
        >>> from landlab import RasterModelGrid
        >>> mg = RasterModelGrid((4, 5), 1.)
        >>> mg.link_at_face[0:3]
        array([5, 6, 7])

        LLCATS: LINF FINF CONN
        """
        try:
            return self._link_at_face
        except AttributeError:
            return self._create_link_at_face()

    @property
    def number_of_corners(self):
        """Total number of corners.

        Examples
        --------
        >>> from landlab import RasterModelGrid
        >>> grid = RasterModelGrid((4, 5))
        >>> grid.number_of_corners
        12

        LLCATS: CNINF
        """
        return self.number_of_patches

    @property
    def number_of_cells(self):
        """Total number of cells.

        Examples
        --------
        >>> from landlab import RasterModelGrid
        >>> grid = RasterModelGrid((4, 5))
        >>> grid.number_of_cells
        6

        LLCATS: CINF
        """
        return len(self._node_at_cell)

    @property
    def number_of_faces(self):
        """Total number of faces.

        Returns
        -------
        int
            Total number of faces in the grid.

        Examples
        --------
        >>> from landlab import RasterModelGrid
        >>> grid = RasterModelGrid((3, 4))
        >>> grid.number_of_faces
        7

        LLCATS: FINF
        """
        return len(self.link_at_face)

    @property
    def number_of_active_faces(self):
        """Total number of active faces.

        Returns
        -------
        int
            Total number of active faces in the grid.

        Examples
        --------
        >>> from landlab import RasterModelGrid
        >>> grid = RasterModelGrid((3, 4))
        >>> grid.number_of_active_faces
        7

        The number of active faces is updated when a node status changes.

        >>> from landlab import CLOSED_BOUNDARY
        >>> grid.status_at_node[6] = CLOSED_BOUNDARY
        >>> grid.number_of_active_faces
        3

        LLCATS: FINF BC
        """
        return self.active_faces.size

    @property
    def number_of_core_cells(self):
        """Number of core cells.

        A core cell excludes all boundary cells.

        Examples
        --------
        >>> from landlab import RasterModelGrid, CLOSED_BOUNDARY
        >>> grid = RasterModelGrid((4, 5))
        >>> grid.number_of_core_cells
        6

        >>> grid.status_at_node[7] = CLOSED_BOUNDARY
        >>> grid.number_of_core_cells
        5

        LLCATS: CINF BC
        """
        return self.core_cells.size

    @property
    @make_return_array_immutable
    def x_of_cell(self):
        """Get array of the x-coordinates of nodes at cells.

        Examples
        --------
        >>> from landlab import RasterModelGrid
        >>> mg = RasterModelGrid((4, 5), (2., 3.))
        >>> mg.x_of_cell.reshape((2, 3))
        array([[  3.,   6.,   9.],
               [  3.,   6.,   9.]])

        LLCATS: CINF MEAS
        """
        return self.x_of_node[self.node_at_cell]

    @property
    @make_return_array_immutable
    def y_of_cell(self):
        """Get array of the y-coordinates of nodes at cells.

        Examples
        --------
        >>> from landlab import RasterModelGrid
        >>> mg = RasterModelGrid((4, 5), (2., 3.))
        >>> mg.y_of_cell.reshape((2, 3))
        array([[ 2.,  2.,  2.],
               [ 4.,  4.,  4.]])

        LLCATS: CINF MEAS
        """
        return self.y_of_node[self.node_at_cell]

    @property
    @cache_result_in_object()
    @make_return_array_immutable
    def x_of_face(self):
        """Get array of the x-coordinates of face midpoints.

        Examples
        --------
        >>> from landlab import RasterModelGrid
        >>> mg = RasterModelGrid((4, 5), (2., 3.))
        >>> mg.x_of_face # doctest: +NORMALIZE_WHITESPACE
        array([  3. ,   6. ,   9. ,   1.5,   4.5,   7.5,  10.5,
                 3. ,   6. ,   9. ,   1.5,   4.5,   7.5,  10.5,
                 3. ,   6. ,   9. ])

        LLCATS: FINF MEAS
        """
        return self.x_of_link[self.link_at_face]

    @property
    @cache_result_in_object()
    @make_return_array_immutable
    def y_of_face(self):
        """Get array of the y-coordinates of face midpoints.

        Examples
        --------
        >>> from landlab import RasterModelGrid
        >>> mg = RasterModelGrid((4, 5), (2., 3.))
        >>> mg.y_of_face # doctest: +NORMALIZE_WHITESPACE
        array([ 1.,  1.,  1.,  2.,  2.,  2.,  2.,  3.,  3.,  3.,
                4.,  4.,  4.,  4.,  5.,  5.,  5.])

        LLCATS: FINF MEAS
        """
        return self.y_of_link[self.link_at_face]

    @property
    @return_readonly_id_array
    def link_at_face(self):
        """Get links associated with faces.

        Returns an array of the link IDs for the links that intersect
        faces.

        Examples
        --------
        >>> from landlab import RasterModelGrid
        >>> mg = RasterModelGrid((3, 4))
        >>> mg.link_at_face
        array([ 4,  5,  7,  8,  9, 11, 12])

        LLCATS: LINF FINF MEAS
        """
        try:
            return self._link_at_face
        except AttributeError:
            return self._create_link_at_face()

    @property
    def faces_at_cell(self):
        """Return array containing face IDs at each cell.

        Creates array if it doesn't already exist.

        Examples
        --------
        >>> from landlab import HexModelGrid, RasterModelGrid
        >>> mg = RasterModelGrid((4, 5))
        >>> mg.faces_at_cell
        array([[ 4,  7,  3,  0],
               [ 5,  8,  4,  1],
               [ 6,  9,  5,  2],
               [11, 14, 10,  7],
               [12, 15, 11,  8],
               [13, 16, 12,  9]])
        >>> mg = HexModelGrid(3, 4)
        >>> mg.faces_at_cell
        array([[ 7, 11, 10,  6,  0,  1],
               [ 8, 13, 12,  7,  2,  3],
               [ 9, 15, 14,  8,  4,  5]])

        LLCATS: FINF CINF CONN
        """
        try:
            return self._faces_at_cell
        except AttributeError:
            self._create_faces_at_cell()
            return self._faces_at_cell

    def number_of_faces_at_cell(self):
        """Number of faces attached to each cell.

        Examples
        --------
        >>> from landlab import HexModelGrid
        >>> hg = HexModelGrid(3, 3)
        >>> hg.number_of_faces_at_cell()
        array([6, 6])

        LLCATS: FINF CINF CONN
        """
        num_faces_at_cell = np.zeros(self.number_of_cells, dtype=np.int)
        node_at_link_tail = self.node_at_link_tail
        node_at_link_head = self.node_at_link_head
        for ln in range(self.number_of_links):
            cell = self.cell_at_node[node_at_link_tail[ln]]
            if cell != BAD_INDEX_VALUE:
                num_faces_at_cell[cell] += 1
            cell = self.cell_at_node[node_at_link_head[ln]]
            if cell != BAD_INDEX_VALUE:
                num_faces_at_cell[cell] += 1
        return num_faces_at_cell

    def _sort_faces_at_cell_by_angle(self):
        """Sort the faces_at_cell array by angle.

        Assumes links_at_node and link_dirs_at_node created.
        """
        for cell in range(self.number_of_cells):
            sorted_links = self.links_at_node[self.node_at_cell[cell], :]
            sorted_faces = self._faces_at_cell[cell, :] = self.face_at_link[
                sorted_links]
            self._faces_at_cell[cell, :] = sorted_faces

    def _create_faces_at_cell(self):
        """Construct faces_at_cell array.

        Examples
        --------
        >>> from landlab import HexModelGrid
        >>> hg = HexModelGrid(3, 3)
        >>> hg._create_faces_at_cell()
        >>> hg._faces_at_cell
        array([[ 5,  8,  7,  4,  0,  1],
               [ 6, 10,  9,  5,  2,  3]])
        """
        num_faces = self.number_of_faces_at_cell()
        self._faces_at_cell = np.zeros((self.number_of_cells,
                                        np.amax(num_faces)), dtype=int)
        num_faces[:] = 0  # Zero out and count again, to use as index
        node_at_link_tail = self.node_at_link_tail
        node_at_link_head = self.node_at_link_head
        for ln in range(self.number_of_links):
            cell = self.cell_at_node[node_at_link_tail[ln]]
            if cell != BAD_INDEX_VALUE:
                self._faces_at_cell[cell, num_faces[cell]] = \
                    self.face_at_link[ln]
                num_faces[cell] += 1
            cell = self.cell_at_node[node_at_link_head[ln]]
            if cell != BAD_INDEX_VALUE:
                self._faces_at_cell[cell, num_faces[cell]] = \
                    self.face_at_link[ln]
                num_faces[cell] += 1
        self._sort_faces_at_cell_by_angle()

    @property
    @make_return_array_immutable
    def patches_present_at_node(self):
        """
        A boolean array, False where a patch has a closed node or is missing.

        The array is the same shape as :func:`patches_at_node`, and is designed
        to mask it.

        Note that in cases where patches may have more than 3 nodes (e.g.,
        rasters), a patch is considered still present as long as at least 3
        open nodes are present.

        Examples
        --------
        >>> from landlab import RasterModelGrid, CLOSED_BOUNDARY
        >>> mg = RasterModelGrid((3, 3))
        >>> mg.status_at_node[mg.nodes_at_top_edge] = CLOSED_BOUNDARY
        >>> mg.patches_at_node
        array([[ 0, -1, -1, -1],
               [ 1,  0, -1, -1],
               [-1,  1, -1, -1],
               [ 2, -1, -1,  0],
               [ 3,  2,  0,  1],
               [-1,  3,  1, -1],
               [-1, -1, -1,  2],
               [-1, -1,  2,  3],
               [-1, -1,  3, -1]])
        >>> mg.patches_present_at_node
        array([[ True, False, False, False],
               [ True,  True, False, False],
               [False,  True, False, False],
               [False, False, False,  True],
               [False, False,  True,  True],
               [False, False,  True, False],
               [False, False, False, False],
               [False, False, False, False],
               [False, False, False, False]], dtype=bool)
        >>> 1 in mg.patches_at_node * mg.patches_present_at_node
        True
        >>> 2 in mg.patches_at_node * mg.patches_present_at_node
        False

        LLCATS: PINF NINF
        """
        try:
            return self._patches_present_mask
        except AttributeError:
            self.patches_at_node
            self._reset_patch_status()
            return self._patches_present_mask

    @property
    @make_return_array_immutable
    def patches_present_at_link(self):
        """
        A boolean array, False where a patch has a closed node or is missing.

        The array is the same shape as :func:`patches_at_link`, and is designed
        to mask it.

        Examples
        --------
        >>> from landlab import RasterModelGrid, CLOSED_BOUNDARY
        >>> mg = RasterModelGrid((3, 3))
        >>> mg.status_at_node[mg.nodes_at_top_edge] = CLOSED_BOUNDARY
        >>> mg.patches_at_link
        array([[ 0, -1],
               [ 1, -1],
               [ 0, -1],
               [ 0,  1],
               [ 1, -1],
               [ 0,  2],
               [ 1,  3],
               [ 2, -1],
               [ 2,  3],
               [ 3, -1],
               [ 2, -1],
               [ 3, -1]])
        >>> mg.patches_present_at_link
        array([[ True, False],
               [ True, False],
               [ True, False],
               [ True,  True],
               [ True, False],
               [ True, False],
               [ True, False],
               [False, False],
               [False, False],
               [False, False],
               [False, False],
               [False, False]], dtype=bool)
        >>> 1 in mg.patches_at_link * mg.patches_present_at_link
        True
        >>> 2 in mg.patches_at_link * mg.patches_present_at_link
        False

        LLCATS: PINF LINF
        """
        try:
            return self._patches_present_link_mask
        except AttributeError:
            self.patches_at_node
            self._reset_patch_status()
            return self._patches_present_link_mask

    @property
    @make_return_array_immutable
    def number_of_patches_present_at_node(self):
        """Return the number of patches at a node without a closed node.

        Examples
        --------
        >>> from landlab import RasterModelGrid, CLOSED_BOUNDARY
        >>> mg = RasterModelGrid((3, 3))
        >>> mg.status_at_node[mg.nodes_at_top_edge] = CLOSED_BOUNDARY
        >>> mg.patches_present_at_node
        array([[ True, False, False, False],
               [ True,  True, False, False],
               [False,  True, False, False],
               [False, False, False,  True],
               [False, False,  True,  True],
               [False, False,  True, False],
               [False, False, False, False],
               [False, False, False, False],
               [False, False, False, False]], dtype=bool)
        >>> mg.number_of_patches_present_at_node
        array([1, 2, 1, 1, 2, 1, 0, 0, 0])

        LLCATS: PINF NINF BC
        """
        try:
            return self._number_of_patches_present_at_node
        except AttributeError:
            self.patches_at_node
            self._reset_patch_status()
            return self._number_of_patches_present_at_node

    @property
    @make_return_array_immutable
    def number_of_patches_present_at_link(self):
        """Return the number of patches at a link without a closed node.

        Examples
        --------
        >>> from landlab import RasterModelGrid, CLOSED_BOUNDARY
        >>> mg = RasterModelGrid((3, 3))
        >>> mg.status_at_node[mg.nodes_at_top_edge] = CLOSED_BOUNDARY
        >>> mg.patches_present_at_link
        array([[ True, False],
               [ True, False],
               [ True, False],
               [ True,  True],
               [ True, False],
               [ True, False],
               [ True, False],
               [False, False],
               [False, False],
               [False, False],
               [False, False],
               [False, False]], dtype=bool)
        >>> mg.number_of_patches_present_at_link
        array([1, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0])

        LLCATS: PINF LINF BC
        """
        try:
            return self._number_of_patches_present_at_link
        except AttributeError:
            self.patches_at_node
            self._reset_patch_status()
            return self._number_of_patches_present_at_link

    def _reset_patch_status(self):
        """
        Creates the array which stores patches_present_at_node.

        Call whenever boundary conditions are updated on the grid.
        """
        from landlab import RasterModelGrid, VoronoiDelaunayGrid
        node_status_at_patch = self.status_at_node[self.nodes_at_patch]
        if isinstance(self, RasterModelGrid):
            max_nodes_at_patch = 4
        elif isinstance(self, VoronoiDelaunayGrid):
            max_nodes_at_patch = 3
        else:
            max_nodes_at_patch = (self.nodes_at_patch > -1).sum(axis=1)
        any_node_at_patch_closed = (node_status_at_patch ==
                                    CLOSED_BOUNDARY).sum(axis=1) > (
                                        max_nodes_at_patch - 3)
        absent_patches = any_node_at_patch_closed[self.patches_at_node]
        bad_patches = numpy.logical_or(absent_patches,
                                       self.patches_at_node == -1)
        self._patches_present_mask = numpy.logical_not(
            bad_patches)
        self._number_of_patches_present_at_node = numpy.sum(
            self._patches_present_mask, axis=1)
        absent_patches = any_node_at_patch_closed[self.patches_at_link]
        bad_patches = numpy.logical_or(absent_patches,
                                       self.patches_at_link == -1)
        self._patches_present_link_mask = numpy.logical_not(
            bad_patches)
        self._number_of_patches_present_at_link = numpy.sum(
            self._patches_present_link_mask, axis=1)

    def calc_hillshade_at_node(self, alt=45., az=315., slp=None, asp=None,
                               unit='degrees', elevs='topographic__elevation'):
        """Get array of hillshade.

        .. codeauthor:: Katy Barnhart <katherine.barnhart@colorado.edu>

        Parameters
        ----------
        alt : float
            Sun altitude (from horizon) - defaults to 45 degrees
        az : float
            Sun azimuth (CW from north) - defaults to 315 degrees
        slp : float
            slope of cells at surface - optional
        asp : float
            aspect of cells at surface (from north) - optional (with slp)
        unit : string
            'degrees' (default) or 'radians' - only needed if slp and asp
                                                are not provided

        If slp and asp are both not specified, 'elevs' must be provided as
        a grid field name (defaults to 'topographic__elevation') or an
        nnodes-long array of elevation values. In this case, the method will
        calculate local slopes and aspects internally as part of the hillshade
        production.

        Returns
        -------
        ndarray of float
            Hillshade at each cell.

        Notes
        -----
        code taken from GeospatialPython.com example from December 14th, 2014
        DEJH found what looked like minor sign problems, and adjusted to follow
        the ArcGIS algorithm: http://help.arcgis.com/en/arcgisdesktop/10.0/
        help/index.html#/How_Hillshade_works/009z000000z2000000/ .

        Remember when plotting that bright areas have high values. cmap='Greys'
        will give an apparently inverted color scheme. *cmap='gray'* has white
        associated with the high values, so is recommended for plotting.

        Examples
        --------
        >>> import numpy as np
        >>> from landlab import RasterModelGrid

        >>> mg = RasterModelGrid((5, 5), 1.)
        >>> z = mg.x_of_node * np.tan(60. * np.pi / 180.)
        >>> mg.calc_hillshade_at_node(elevs=z, alt=30., az=210.)
        array([ 0.625,  0.625,  0.625,  0.625,  0.625,  0.625,  0.625,  0.625,
                0.625,  0.625,  0.625,  0.625,  0.625,  0.625,  0.625,  0.625,
                0.625,  0.625,  0.625,  0.625,  0.625,  0.625,  0.625,  0.625,
                0.625])

        LLCATS: NINF SURF
        """
        if slp is not None and asp is not None:
            if unit == 'degrees':
                (alt, az, slp, asp) = (numpy.radians(alt), numpy.radians(az),
                                       numpy.radians(slp), numpy.radians(asp))
            elif unit == 'radians':
                if alt > numpy.pi / 2. or az > 2. * numpy.pi:
                    six.print_(
                        'Assuming your solar properties are in degrees, '
                        'but your slopes and aspects are in radians...')
                    (alt, az) = (numpy.radians(alt), numpy.radians(az))
                    # ...because it would be super easy to specify radians,
                    # but leave the default params alone...
            else:
                raise TypeError("unit must be 'degrees' or 'radians'")
        elif slp is None and asp is None:
            if unit == 'degrees':
                (alt, az) = (numpy.radians(alt), numpy.radians(az))
            elif unit == 'radians':
                pass
            else:
                raise TypeError("unit must be 'degrees' or 'radians'")
            slp, slp_comps = self.calc_slope_at_node(
                elevs, return_components=True)

            asp = self.calc_aspect_at_node(slope_component_tuple=slp_comps,
                                           unit='radians')
        else:
            raise TypeError('Either both slp and asp must be set, or neither!')

        shaded = (
            numpy.sin(alt) * numpy.cos(slp) +
            numpy.cos(alt) * numpy.sin(slp) * numpy.cos(az - asp)
        )

        return shaded.clip(0.)

    @property
    @make_return_array_immutable
    def cell_area_at_node(self):
        """Cell areas in a nnodes-long array.

        Zeros are entered at all perimeter nodes, which lack cells.

        Returns
        -------
        ndarray
            Cell areas as an n_nodes-long array.

        Examples
        --------
        >>> from landlab import RasterModelGrid
        >>> grid = RasterModelGrid((4, 5), spacing=(3, 4))
        >>> grid.status_at_node[7] = CLOSED_BOUNDARY
        >>> grid.cell_area_at_node
        array([  0.,   0.,   0.,   0.,   0.,
                 0.,  12.,  12.,  12.,   0.,
                 0.,  12.,  12.,  12.,   0.,
                 0.,   0.,   0.,   0.,   0.])

        LLCATS: CINF NINF CONN
        """
        try:
            return self._cell_area_at_node
        except AttributeError:
            return self._create_cell_areas_array_force_inactive()

    @property
    @deprecated(use='width_of_face', version=1.0)
    def face_width(self):
        """
        LLCATS: DEPR FINF MEAS
        """
        return self.width_of_face

    @property
    @make_return_array_immutable
    def width_of_face(self):
        """Width of grid faces.

        Examples
        --------
        >>> from landlab import RasterModelGrid, HexModelGrid
        >>> mg = RasterModelGrid((3, 4), (1., 2.))
        >>> mg.width_of_face
        array([ 2.,  2.,  2.,  1.,  1.,  1.,  1.])
        >>> mg = HexModelGrid(3, 3)
        >>> np.allclose(mg.width_of_face, 0.57735027)
        True

        LLCATS: FINF MEAS
        """
        try:
            return self._face_width
        except AttributeError:
            return self._create_face_width()

    def _create_face_at_link(self):
        """Set up face_at_link array.

        Examples
        --------
        >>> from landlab import HexModelGrid, BAD_INDEX_VALUE
        >>> hg = HexModelGrid(3, 3)

        >>> face_at_link = hg.face_at_link.copy()
        >>> face_at_link[face_at_link == BAD_INDEX_VALUE] = -1
        >>> face_at_link # doctest: +NORMALIZE_WHITESPACE
        array([-1, -1, -1,  0,  1,  2,  3, -1,  4,  5,  6, -1,  7,  8,  9, 10,
               -1, -1, -1])
        """
        self._face_at_link = numpy.full(self.number_of_links, BAD_INDEX_VALUE,
                                        dtype=int)
        face_id = 0
        node_at_link_tail = self.node_at_link_tail
        node_at_link_head = self.node_at_link_head
        for link in range(self.number_of_links):
            tc = self.cell_at_node[node_at_link_tail[link]]
            hc = self.cell_at_node[node_at_link_head[link]]
            if tc != BAD_INDEX_VALUE or hc != BAD_INDEX_VALUE:
                self._face_at_link[link] = face_id
                face_id += 1

        return self._face_at_link

    def _create_link_at_face(self):
        """Set up link_at_face array.

        Examples
        --------
        >>> from landlab import HexModelGrid
        >>> hg = HexModelGrid(3, 3)
        >>> hg.link_at_face
        array([ 3,  4,  5,  6,  8,  9, 10, 12, 13, 14, 15])
        """
        num_faces = len(self.width_of_face)
        self._link_at_face = numpy.empty(num_faces, dtype=int)
        face_id = 0
        node_at_link_tail = self.node_at_link_tail
        node_at_link_head = self.node_at_link_head
        for link in range(self.number_of_links):
            tc = self.cell_at_node[node_at_link_tail[link]]
            hc = self.cell_at_node[node_at_link_head[link]]
            if tc != BAD_INDEX_VALUE or hc != BAD_INDEX_VALUE:
                self._link_at_face[face_id] = link
                face_id += 1

        return self._link_at_face

    def _create_cell_areas_array_force_inactive(self):
        """Set up an array of cell areas that is n_nodes long.

        Sets up an array of cell areas that is nnodes long. Nodes that have
        cells receive the area of that cell. Nodes which do not, receive
        zeros.
        """
        _cell_area_at_node_zero = numpy.zeros(self.number_of_nodes,
                                              dtype=float)
        _cell_area_at_node_zero[self.node_at_cell] = self.area_of_cell
        self._cell_area_at_node = _cell_area_at_node_zero
        return self._cell_area_at_node

    @property
    @make_return_array_immutable
    def area_of_cell(self):
        """Get areas of grid cells.

        Examples
        --------
        >>> from landlab import RasterModelGrid
        >>> grid = RasterModelGrid((4, 5), spacing=(2, 3))
        >>> grid.area_of_cell # doctest: +NORMALIZE_WHITESPACE
        array([ 6.,  6.,  6.,
                6.,  6.,  6.])

        LLCATS: CINF MEAS
        """
        return self._area_of_cell

    def reset_status_at_node(self):
        attrs = ['_active_link_dirs_at_node', '_status_at_link',
                 '_active_links', '_fixed_links', '_activelink_fromnode',
                 '_activelink_tonode', '_active_faces', '_core_nodes',
                 '_core_cells', '_fixed_links',
                 '_active_adjacent_nodes_at_node',
                 '_fixed_value_boundary_nodes', '_node_at_core_cell',
                 '_link_status_at_node', ]

        for attr in attrs:
            try:
                del self.__dict__[attr]
            except KeyError:
                pass
        try:
            self.bc_set_code += 1
        except AttributeError:
            self.bc_set_code = 0
        try:
            del self.__dict__['__node_active_inlink_matrix']
        except KeyError:
            pass
        try:
            del self.__dict__['__node_active_outlink_matrix']
        except KeyError:
            pass

    @deprecated(use='node_is_boundary', version=1.0)
    def is_boundary(self, ids, boundary_flag=None):
        """
        LLCATS: DEPR NINF BC
        """
        return self.node_is_boundary(ids, boundary_flag=boundary_flag)

    def _assign_boundary_nodes_to_grid_sides(self):
        """Assign boundary nodes to a quadrant.

        For each boundary node, determines whether it belongs to the left,
        right, top or bottom of the grid, based on its distance from the grid's
        centerpoint (mean (x,y) position). Returns lists of nodes on each of
        the four grid sides. Assumes self.status_at_node, self.number_of_nodes,
        self.boundary_nodes, self._node_x, and self._node_y have been
        initialized.

        Returns
        -------
        tuple of array_like
            Tuple of nodes in each coordinate. Nodes are grouped as
            (*east*, *north*, *west*, *south*).

        Examples
        --------
        >>> import landlab as ll
        >>> m = ll.HexModelGrid(5, 3, 1.0)
        >>> [r,t,l,b] = m._assign_boundary_nodes_to_grid_sides()
        >>> l
        array([ 7, 12,  3])
        >>> r
        array([11, 15,  6])
        >>> t
        array([16, 18, 17])
        >>> b
        array([0, 2, 1])
        """
        # Calculate x and y distance from centerpoint
        diff_x = self.node_x[self.boundary_nodes] - numpy.mean(self.node_x)
        diff_y = self.node_y[self.boundary_nodes] - numpy.mean(self.node_y)

        return _sort_points_into_quadrants(diff_x, diff_y, self.boundary_nodes)

add_module_functions_to_class(ModelGrid, 'mappers.py', pattern='map_*')
# add_module_functions_to_class(ModelGrid, 'gradients.py',
#                               pattern='calculate_*')
add_module_functions_to_class(ModelGrid, 'gradients.py', pattern='calc_*')
add_module_functions_to_class(ModelGrid, 'divergence.py', pattern='calc_*')


if __name__ == '__main__':
    import doctest
    doctest.testmod()
