# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19.

@author: dejh
"""

from landlab import Component
import richdem as rd
import numpy as np
from landlab.utils.decorators import use_field_name_or_array

@use_field_name_or_array('node')
def _return_surface(grid, surface):
    """
    Private function to return the surface to direct flow over.

    This function exists to take advantange of the 'use_field_name_or_array
    decorator which permits providing the surface as a field name or array.
    """
    return surface

class DepressionFiller(Component):
    """

    Construction::

        DepressionFiller(grid, epsilon=True)
    Parameters
    ----------
    grid : ModelGrid
        A landlab grid.
    epsilon : boolean
        

    Examples
    --------
   
    """
    _name = 'DepressionFiller'

    _input_var_names = ('topographic__elevation',
                        )

    _output_var_names = ('topographic__elevation',
                         'sediment_fill__depth',
                         )

    _var_units = {'topographic__elevation': 'm',
                  'sediment_fill__depth': 'm',
                  }

    _var_mapping = {'topographic__elevation': 'node',
                    'sediment_fill__depth': 'node',
                    }

    _var_doc = {'topographic__elevation': 'Surface topographic elevation',
                'sediment_fill__depth': 'Depth of sediment added at each' +
                                        'node',
                }

    def __init__(self, 
                 grid, 
                 dem_to_fill='topographic__elevation', 
                 filled_dem='topographic__elevation', 
                 epsilon=True, 
                 no_data=-9999.):
        
        super(DepressionFiller, self).__init__(grid)

        self._grid = grid
        
        self._epsilon = epsilon

        # get dem
        self._dem_to_fill = _return_surface(grid, dem_to_fill)
        if filled_dem != dem_to_fill:
            self._filled_dem = _return_surface(grid, filled_dem)
            self._in_place = False
        else:
            self._filled_dem = None
            self._in_place = True
        
        # create the only new output field:
        if 'sediment_fill__depth' not in grid.at_node:
            self.sed_fill_depth = self._grid.add_zeros('node',
                                                       'sediment_fill__depth',
                                                       noclobber=False)
        else:
            self.sed_fill_depth = grid.at_node['sediment_fill__depth']
            
        # this is the ideal use
        self._rd_dem_to_fill = rd.rdarray(self._dem_to_fill.reshape(self._grid.shape), no_data=no_data)

#        # this works, but does not point to the memory location
#        self.rd_elev = rd._richdem.Array2D_float()
#        self.rd_elev.fromArray(self._elev.reshape(self._grid.shape))
        
        # in a flow accumulation sense we may hvae some more work to do because
        # assinging the richedem output into the landlab numpy array will 
        # result in a memory copy. This may be fixable, if we point richdem
        # to the dem and to the array used for accumulation. 
        # ideally we have something like:
        # rd.FlowAccumulation(dem, accum, method="Tarboton")
        # instead of the present:
        # accum = rd.FlowAccumulation(dem,   method="Tarboton")

    def run_one_step(self):
        """
        This is the main method. Call it to fill depressions in a starting
        topography.
        """
    
        self._original_dem_to_fill = self._dem_to_fill.copy()
        
        if self._in_place:
            rd.FillDepressions(self._rd_dem_to_fill, epsilon=self._epsilon, in_place=self._in_place) # in place
            self.sed_fill_depth = np.array(self._rd_dem_to_fill).flatten() - self._original_dem_to_fill
        else:
            self._filled_dem[:] = rd.FillDepressions(self._rd_dem_to_fill, epsilon=self._epsilon, in_place=self._in_place).flatten()
            self.sed_fill_depth = self._filled_dem - self._original_dem_to_fill
