import numpy as np
from landlab import Component
from landlab import CLOSED_BOUNDARY

class HybridAlluvium(Component):
    """
    Stream Power with Alluvium Conservation and Entrainment (SPACE)
    
    Algorithm developed by G. Tucker, summer 2016.
    Component written by C. Shobe, begun 11/28/2016.
    """
    
    _name= 'HybridAlluvium'
    
    _input_var_names = (
        'flow__receiver_node',
        'flow__upstream_node_order',
        'topographic__steepest_slope',
        'drainage_area',
        'soil__depth'
    )
    
    _output_var_names = (
        'topographic__elevation'
        'soil__depth'
    )
    
    _var_units = {
        'flow__receiver_node': '-',
        'flow__upstream_node_order': '-',
        'topographic__steepest_slope': '-',
        'drainage_area': 'm**2',
        'soil__depth': 'm',
        'topographic__elevation': 'm',
    }
    
    _var_mapping = {
        'flow__receiver_node': 'node',
        'flow__upstream_node_order': 'node',
        'topographic__steepest_slope': 'node',
        'drainage_area': 'node',
        'soil__depth': 'node',
        'topographic__elevation': 'node',
    }
    
    _var_doc = {
        'flow__receiver_node':
            'Node array of receivers (node that receives flow from current '
            'node)',
        'flow__upstream_node_order':
            'Node array containing downstream-to-upstream ordered list of '
            'node IDs',
        'topographic__steepest_slope':
            'Topographic slope at each node',
        'drainage_area':
            "Upstream accumulated surface area contributing to the node's "
            "discharge",
        'soil__depth': 
            'Depth of sediment above bedrock',
        'topographic__elevation': 
            'Land surface topographic elevation',
    }
    
    def __init__(self, grid, K_sed=None, K_br=None, F_f=None, 
                 phi=None, H_star=None, v_s=None, 
                 m_sp=None, n_sp=None, sp_crit_sed=None, 
                 sp_crit_br=None, method=None, discharge_method=None, 
                 area_field=None, discharge_field=None, **kwds):
        """Initialize the HybridAlluvium model.
        
        Parameters
        ----------
        grid : ModelGrid
            Landlab ModelGrid object
        K_sed : float
            Erodibility constant for sediment (units vary).
        K_br : float
            Erodibility constant for bedrock (units vary).
        F_f : float
            Fraction of permanently suspendable fines in bedrock [-].
        phi : float
            Sediment porosity [-].
        H_star : float
            Sediment thickness required for full entrainment [L].
        v_s : float
            Effective settling velocity for chosen grain size metric [L/T].
        m_sp : float
            Drainage area exponent (units vary)
        n_sp : float
            Slope exponent (units vary)
        sp_crit_sed : float
            Critical stream power to erode sediment [E/(TL^2)]
        sp_crit_br : float
            Critical stream power to erode rock [E/(TL^2)]
        method : string
            Either "simple_stream_power", "threshold_stream_power", or
            "stochastic_hydrology". Method for calculating sediment
            and bedrock entrainment/erosion.
        discharge_method : string
            Either "area_field" or "discharge_field". If using stochastic
            hydrology, determines whether component is supplied with
            drainage area or discharge.
        area_field : string or array
            Used if discharge_method = 'area_field'. Either field name or
            array of length(number_of_nodes) containing drainage areas [L^2].
        discharge_field : string or array
            Used if discharge_method = 'discharge_field'.Either field name or
            array of length(number_of_nodes) containing drainage areas [L^2/T].
            
        Examples
        ---------
        >>> import numpy as np
        >>> from landlab import RasterModelGrid
        >>> from landlab.components.flow_routing import FlowRouter
        >>> from landlab.components import DepressionFinderAndRouter
        >>> from landlab.components import HybridAlluvium
        >>> from landlab.components import FastscapeEroder
        >>> np.random.seed(seed = 5000)
        
        Define grid and initial topography:
            -5x5 grid with baselevel in the lower left corner
            -all other boundary nodes closed
            -Initial topography is plane tilted up to the upper right + noise
        
        >>> nr = 5
        >>> nc = 5
        >>> dx = 10
        >>> mg = RasterModelGrid((nr, nc), 10.0)
        >>> _ = mg.add_zeros('node', 'topographic__elevation')
        >>> mg['node']['topographic__elevation'] += mg.node_y/10 + \
                mg.node_x/10 + np.random.rand(len(mg.node_y)) / 10
        >>> mg.set_closed_boundaries_at_grid_edges(bottom_is_closed=True,\
                                                       left_is_closed=True,\
                                                       right_is_closed=True,\
                                                       top_is_closed=True)
        >>> mg.set_watershed_boundary_condition_outlet_id(0,\
                mg['node']['topographic__elevation'], -9999.)
        >>> fsc_dt = 100. 
        >>> hybrid_dt = 100.
        
        Instantiate Fastscape eroder, flow router, and depression finder        
        
        >>> fsc = FastscapeEroder(mg, K_sp=.001, m_sp=.5, n_sp=1)
        >>> fr = FlowRouter(mg) #instantiate
        >>> df = DepressionFinderAndRouter(mg)
        
        Burn in an initial drainage network using the Fastscape eroder:
        
        >>> for x in range(100): 
        ...     fr.run_one_step()
        ...     df.map_depressions()
        ...     flooded = np.where(df.flood_status==3)[0]
        ...     fsc.run_one_step(dt = fsc_dt, flooded_nodes=flooded)
        ...     mg.at_node['topographic__elevation'][0] -= 0.001 #uplift
        
        Add some soil to the drainage network:        
        
        >>> _ = mg.add_zeros('soil__depth', at='node', dtype=float)
        >>> mg.at_node['soil__depth'] += 0.5
        >>> mg.at_node['topographic__elevation'] += mg.at_node['soil__depth']
        
        Instantiate the hybrid component:        
        
        >>> ha = HybridAlluvium(mg, K_sed=0.00001, K_br=0.00000000001,\
                                F_f=0.5, phi=0.1, H_star=1., v_s=0.001,\
                                m_sp=0.5, n_sp = 1.0, sp_crit_sed=0,\
                                sp_crit_br=0, method='simple_stream_power',\
                                discharge_method=None, area_field=None,\
                                discharge_field=None)
                                
        Now run the hybrid component for 2000 short timesteps:                            
                                
        >>> for x in range(2000): #hybrid component loop
        ...     fr.run_one_step()
        ...     df.map_depressions()
        ...     flooded = np.where(df.flood_status==3)[0]
        ...     ha.run_one_step(dt = hybrid_dt, flooded_nodes=flooded)
        ...     mg.at_node['bedrock__elevation'][0] -= 2e-6 * hybrid_dt
        
        Now we test to see if soil depth and topography are right:
        
        >>> mg.at_node['soil__depth'] # doctest: +NORMALIZE_WHITESPACE
        array([ 0.50003494,  0.5       ,  0.5       ,  0.5       ,  0.5       ,
               0.5       ,  0.11887907,  0.16197065,  0.21999913,  0.5       ,
               0.5       ,  0.1619594 ,  0.1504285 ,  0.21048366,  0.5       ,
               0.5       ,  0.21974902,  0.21047882,  0.21695828,  0.5       ,
               0.5       ,  0.5       ,  0.5       ,  0.5       ,  0.5       ])
        
        >>> mg.at_node['topographic__elevation'] # doctest: +NORMALIZE_WHITESPACE
        array([ 0.52313972,  1.53606698,  2.5727653 ,  3.51126678,  4.56077707,
                1.58157495,  0.54749666,  0.59765511,  0.66640713,  5.50969486,
                2.54008677,  0.5976575 ,  0.58613448,  0.65805841,  6.52641123,
                3.55874171,  0.66646161,  0.65805951,  0.68238831,  7.55334077,
                4.55922478,  5.5409473 ,  6.57035008,  7.5038935 ,  8.51034357])
        """
        #assign class variables to grid fields; create necessary fields
        self.flow_receivers = grid.at_node['flow__receiver_node']
        self.stack = grid.at_node['flow__upstream_node_order']
        self.topographic__elevation = grid.at_node['topographic__elevation']
        self.slope = grid.at_node['topographic__steepest_slope']
        
        self.is_not_closed = grid.status_at_node < CLOSED_BOUNDARY

        try:
            self.soil__depth = grid.at_node['soil__depth']
        except KeyError:
            self.soil__depth = grid.add_zeros(
                'soil__depth', at='node', dtype=float)
        self.link_lengths = grid._length_of_link_with_diagonals
        try:
            self.bedrock__elevation = grid.at_node['bedrock__elevation']
        except KeyError:
            self.bedrock__elevation = grid.add_zeros(
                'bedrock__elevation', at='node', dtype=float)
        self.bedrock__elevation[:] += self.topographic__elevation.copy()
        try:
            self.qs = grid.at_node['sediment__flux']
        except KeyError:
            self.qs = grid.add_zeros(
                'sediment__flux', at='node', dtype=float)
        try:
            self.q = grid.at_node['surface_water__discharge']
        except KeyError:
            self.q = grid.add_zeros(
                'surface_water__discharge', at='node', dtype=float)
                
        self._grid = grid #store grid
        
        #store other constants
        self.m_sp = float(m_sp)
        self.n_sp = float(n_sp)
        self.F_f = float(F_f)
        self.phi = float(phi)
        self.H_star = float(H_star)
        self.v_s = float(v_s)
        
        #K's and critical values can be floats, grid fields, or arrays
        if type(K_sed) is str:
            self.K_sed = self._grid.at_node[K_sed]
        elif type(K_sed) in (float, int):  # a float
            self.K_sed = float(K_sed)
        elif len(K_sed) == self.grid.number_of_nodes:
            self.K_sed = np.array(K_sed)
        else:
            raise TypeError('Supplied type of K_sed ' +
                            'was not recognised, or array was ' +
                            'not nnodes long!') 
                            
        if type(K_br) is str:
            self.K_br = self._grid.at_node[K_br]
        elif type(K_br) in (float, int):  # a float
            self.K_br = float(K_br)
        elif len(K_br) == self.grid.number_of_nodes:
            self.K_br = np.array(K_br)
        else:
            raise TypeError('Supplied type of K_br ' +
                            'was not recognised, or array was ' +
                            'not nnodes long!') 
        
        if sp_crit_sed is not None:
            if type(sp_crit_sed) is str:
                self.sp_crit_sed = self._grid.at_node[sp_crit_sed]
            elif type(sp_crit_sed) in (float, int):  # a float
                self.sp_crit_sed = float(sp_crit_sed)
            elif len(sp_crit_sed) == self.grid.number_of_nodes:
                self.sp_crit_sed = np.array(sp_crit_sed)
            else:
                raise TypeError('Supplied type of sp_crit_sed ' +
                                'was not recognised, or array was ' +
                                'not nnodes long!') 
            
        if sp_crit_br is not None:                
            if type(sp_crit_br) is str:
                self.sp_crit_br = self._grid.at_node[sp_crit_br]
            elif type(sp_crit_br) in (float, int):  # a float
                self.sp_crit_br = float(sp_crit_br)
            elif len(sp_crit_br) == self.grid.number_of_nodes:
                self.sp_crit_br = np.array(sp_crit_br)
            else:
                raise TypeError('Supplied type of sp_crit_br ' +
                                'was not recognised, or array was ' +
                                'not nnodes long!') 
                                
        #go through erosion methods to ensure correct hydrology
        self.method = str(method)
        self.discharge_method = str(discharge_method) 
        self.area_field = str(area_field)
        self.discharge_field = str(discharge_field)
        
        if self.method == 'simple_stream_power':
            self.simple_stream_power()
        elif self.method == 'threshold_stream_power':
            self.threshold_stream_power()
        elif self.method == 'stochastic_hydrology':
            self.stochastic_hydrology()
        else:
            raise ValueError('Specify erosion method (simple stream power,\
                            threshold stream power, or stochastic hydrology)!')
            
    def simple_stream_power(self):
        """Calculate required fields under simple stream power hydrology."""
        if self.method == 'simple_stream_power' and self.discharge_method == None:
            self.q[:] = np.power(self.grid.at_node['drainage_area'], self.m_sp)
        elif self.method == 'simple_stream_power' and self.discharge_method is not None: 
            if self.discharge_method == 'drainage_area':
                if self.area_field is not None:
                    if type(self.area_field) is str:
                        self.drainage_area = self._grid.at_node[self.area_field]
                    elif len(self.area_field) == self.grid.number_of_nodes:
                        self.drainage_area = np.array(self.area_field)
                    else:
                        raise TypeError('Supplied type of area_field ' +
                                'was not recognised, or array was ' +
                                'not nnodes long!')  
                self.q[:] = np.power(self.drainage_area, self.m_sp)
            elif self.discharge_method == 'discharge_field':
                if self.discharge_field is not None:                    
                    if type(self.discharge_field) is str:
                        self.q[:] = self._grid.at_node[self.discharge_field]
                    elif len(self.discharge_field) == self.grid.number_of_nodes:
                        self.q[:] = np.array(self.discharge_field)
                    else:
                        raise TypeError('Supplied type of discharge_field ' +
                                'was not recognised, or array was ' +
                                'not nnodes long!')
        self.Es = self.K_sed * self.q * np.power(self.slope, self.n_sp) * \
            (1.0 - np.exp(-self.soil__depth / self.H_star))
        self.Er = self.K_br * self.q * np.power(self.slope, self.n_sp) * \
            np.exp(-self.soil__depth / self.H_star)
        self.sed_erosion_term = self.K_sed * self.q * \
            np.power(self.slope, self.n_sp)
        self.br_erosion_term = self.K_br * self.q * \
            np.power(self.slope, self.n_sp)
            
    def threshold_stream_power(self):
        """Calculate required fields under threshold stream power hydrology."""
        if self.method == 'threshold_stream_power' and self.discharge_method == None:
            self.q[:] = np.power(self.grid.at_node['drainage_area'], self.m_sp)
        elif self.method == 'threshold_stream_power' and self.discharge_method is not None:
            if self.discharge_method == 'drainage_area':
                if self.area_field is not None:
                    if type(self.area_field) is str:
                        self.drainage_area = self._grid.at_node[self.area_field]
                    elif len(self.area_field) == self.grid.number_of_nodes:
                        self.drainage_area = np.array(self.area_field)
                    else:
                        raise TypeError('Supplied type of area_field ' +
                                'was not recognised, or array was ' +
                                'not nnodes long!')  
                self.q[:] = np.power(self.drainage_area, self.m_sp)
            elif self.discharge_method == 'discharge_field':
                if self.discharge_field is not None:                    
                    if type(self.discharge_field) is str:
                        self.q[:] = self._grid.at_node[self.discharge_field]
                    elif len(self.discharge_field) == self.grid.number_of_nodes:
                        self.q[:] = np.array(self.discharge_field)
                    else:
                        raise TypeError('Supplied type of discharge_field ' +
                                'was not recognised, or array was ' +
                                'not nnodes long!')
        omega_sed = self.K_sed * self.q * \
            np.power(self.slope, self.n_sp)
        omega_br = self.K_br * self.q * \
            np.power(self.slope, self.n_sp)
        self.Es = (omega_sed - self.sp_crit_sed * (1 - np.exp(-omega_sed /\
            self.sp_crit_sed))) * \
            (1.0 - np.exp(-self.soil__depth / self.H_star))
        self.Er = (omega_br - self.sp_crit_br * (1 - np.exp(-omega_br /\
            self.sp_crit_br))) * \
            np.exp(-self.soil__depth / self.H_star)
        self.sed_erosion_term = omega_sed - self.sp_crit_sed * \
            (1 - np.exp(-omega_sed / self.sp_crit_sed))
        self.br_erosion_term = omega_br - self.sp_crit_br * \
            (1 - np.exp(-omega_br / self.sp_crit_br))
            
    def stochastic_hydrology(self):
        """Calculate required fields under stochastic hydrology."""

        if self.method == 'stochastic_hydrology' and self.discharge_method == None:
            raise TypeError('Supply a discharge method to use stoc. hydro!')
        elif self.discharge_method is not None:
            if self.discharge_method == 'drainage_area':
                if self.area_field is not None:
                    if type(self.area_field) is str:
                        self.drainage_area = self._grid.at_node[self.area_field]
                    elif len(self.area_field) == self.grid.number_of_nodes:
                        self.drainage_area = np.array(self.area_field)
                    else:
                        raise TypeError('Supplied type of area_field ' +
                                'was not recognised, or array was ' +
                                'not nnodes long!')  
                self.q[:] = np.power(self.drainage_area, self.m_sp)
            elif self.discharge_method == 'discharge_field':
                if self.discharge_field is not None:                    
                    if type(self.discharge_field) is str:
                        self.q[:] = self._grid.at_node[self.discharge_field]
                    elif len(self.discharge_field) == self.grid.number_of_nodes:
                        self.q[:] = np.array(self.discharge_field)
                    else:
                        raise TypeError('Supplied type of discharge_field ' +
                                'was not recognised, or array was ' +
                                'not nnodes long!')  
            else:
                raise ValueError('Specify discharge method for stoch hydro!')
        self.Es = self.K_sed * self.q * np.power(self.slope, self.n_sp) * \
            (1.0 - np.exp(-self.soil__depth / self.H_star))
        self.Er = self.K_br * self.q * np.power(self.slope, self.n_sp) * \
            np.exp(-self.soil__depth / self.H_star)
        self.sed_erosion_term = self.K_sed * self.q * \
            np.power(self.slope, self.n_sp)
        self.br_erosion_term = self.K_br * self.q * \
            np.power(self.slope, self.n_sp)
            
    def run_one_step(self, dt=1.0, flooded_nodes=None, dynamic_dt = False, 
                     flow_director=None, slope_thresh=1.0, **kwds):
        """Calculate change in rock and alluvium thickness for
           a time period 'dt'.
        
        Parameters
        ----------
        dt : float (optional, default 1.0)
            Model timestep [T]
        flooded_nodes : array (optional, default None)
            Indices of flooded nodes, passed from flow router
        dynamic_dt : boolean (optional, default False)
            Value to determine if the run_one_step method should use a dynamic
            dt in order to ensure no negative soil thickness values
        flow_director : instantiated FlowDirector instance. 
            Flow director to use for recalculating slopes within the dynamic
            time stepping routine. 
        """        
        
        # Check that if dynamic time-stepping is specified that a FlowDirector
        # instance has been provided.
        
        if dynamic_dt:
            PERMITTED_DIRECTORS = ['FlowDirectorSteepest',
                               'FlowDirectorD8',
                               'FlowDirectorMFD',
                               'FlowDirectorDINF']
            if isinstance(flow_director, Component) & ( flow_director._name in PERMITTED_DIRECTORS):
                pass
            else:
                raise ValueError('When dynamic_dt is set to True, an '
                                 'instantiated FlowDirector must be passed '
                                 'to the run_one_step method in order for '
                                 'recalculation of link slopes. This error '
                                 'indicates that an appropriate FlowDirector '
                                 'was not passed.')
        
        # calculate erosion terms 
        self._calculate_erosion_terms_from_hydrology()

        # Calculate qs_in and deposition per time
        self._calculate_qs_in_and_deposition_rate()
        
        # calculate new soil depth. 
        soil__depth = self.soil__depth.copy()
        soil__depth = self.calculate_soil_depth(dt, 
                                                soil__depth, 
                                                self.deposition_pertime, 
                                                flooded_nodes)
        
        bedrock__elevation = self.bedrock__elevation.copy()
        bedrock__elevation = self._update_bedrock_elevation(dt, bedrock__elevation, soil__depth)
        
        # Default option is to use the supplied dt
        if dynamic_dt == False:
            
            # set soil depth to the calcuated value
            self.soil__depth[:] = soil__depth.copy()
            
            # update bedrock elevation
            self.bedrock__elevation[:] = bedrock__elevation.copy()
            
            # update topographic elevation by summing bedrock and soil
            self.topographic__elevation[self.is_not_closed] = self.bedrock__elevation[self.is_not_closed] + \
                                                              self.soil__depth[self.is_not_closed] 
                                                              
            # If there is negative soil, raise a warning. This is an indication 
            # of timesteps that are too long. 
            if np.any(self.soil__depth<0):
                raise Warning('Soil depth is negative, this probably means the'
                              ' timestep is too big in the hybrid alluvium'
                              ' method')
                
            # if slopes are too steep, raise a warning. Technically the slopes
            # will be steeper by the time the warning is raised since the flow
            # director has not been re-run since the elevations have changed in
            # the time step. 
            if np.any(self._grid['node']['topographic__steepest_slope'] > slope_thresh):
                raise Warning('Topographic slopes are exceeding the slope '
                              'threshold in the hybrid alluvium component'
                              'it is recommended that you use a shorter or '
                              'dynamic timestep')
   
        # an alternative option uses a dynamic dt to ensure that soil thickness
        # never gets below zero.
        else: 
            # get original bedrock elevation and soil depth
            bedrock__elevation_orig = self.bedrock__elevation.copy()
            soil__depth_orig = self.soil__depth.copy()
            
            # get original values of fields that the flow director will overwrite. 
            topographic__elevation_orig = self._grid['node']['topographic__elevation'].copy()
            flow__receiver_node_orig = self._grid['node']['flow__receiver_node'].copy()
            topographic__steepest_slope_orig = self._grid['node']['topographic__steepest_slope'].copy()
            flow__link_to_receiver_node_orig = self._grid['node']['flow__link_to_receiver_node'].copy()
            flow__sink_flag_orig = self._grid['node']['flow__sink_flag'].copy()
            
            # set soil depth to the calcuated value
            self.soil__depth[:] = soil__depth.copy()
            
            # update bedrock elevation
            self.bedrock__elevation[:] = bedrock__elevation.copy()
            
            # update topographic elevation by summing bedrock and soil
            self.topographic__elevation[self.is_not_closed] = self.bedrock__elevation[self.is_not_closed] + \
                                                              self.soil__depth[self.is_not_closed] 
            # run the flow  this will update the slopes, etc.
            flow_director.run_one_step()
            
            # identify where slopes are too steep. 
            too_steep = self.slope > slope_thresh
            
            # if soil depth anywhere is negative or NAN, or if slopes are too
            # steep, the  enter the dynamic time option
            if np.any(soil__depth<0) or np.any(np.isnan(soil__depth)) or np.any(too_steep):
                
                print('dynamicTime!')
                
                all_positive = False
                number_of_sub_timesteps = 2
                
                while all_positive == False:
                    
                    # calculate number of sub_timesteps
                    sub_dt = float(dt)/number_of_sub_timesteps
                    self.sub_dt = sub_dt
                    print('trying timesteps of', sub_dt)
                    for nst in range(number_of_sub_timesteps):
                        
                        if nst == 1 and number_of_sub_timesteps>2:
                            flow_director.run_one_step()
                            
                        # update hydrology and erosion terms
                        self._calculate_erosion_terms_from_hydrology()
                        
                        # Calculate qs_in and deposition per time
                        self._calculate_qs_in_and_deposition_rate()
                        
                        # calculate soil depth
                        self.soil__depth[:] = self.calculate_soil_depth(sub_dt, 
                                                                soil__depth,
                                                                self.deposition_pertime, 
                                                                flooded_nodes)
                        # calculate bedrock elevation
                        self.bedrock__elevation[:] = self._update_bedrock_elevation(sub_dt, bedrock__elevation, soil__depth)
                        
                        # update topographic elevation by summing bedrock and soil
                        self.topographic__elevation[self.is_not_closed] = self.bedrock__elevation[self.is_not_closed] + \
                                                                          self.soil__depth[self.is_not_closed] 
                        
                        # run flow director, this will update the slopes, etc. 
                        flow_director.run_one_step()
                        
                        # identify where slopes are too steep. 
                        too_steep = self.slope > slope_thresh
                        
                        if np.any(soil__depth<0) or np.any(np.isnan(soil__depth)) or np.any(too_steep):
                            # after each sub-itteration check the soil depths
                            # if at this sub timestep size soil depth has gone 
                            # negative, break, don't finish this attempted itteration
                            # and move on to an attempt with a smaller dt size. 
                            break
                        
                        # if all soil depths are still above zero and not nan,
                        # and slopes are not too steep. continue
                        else:
                            pass
                            
                    # after re-running the subtimestep, re-check about positive
                    # soil values
                    if np.all(soil__depth>=0) and np.any(np.isnan(soil__depth)) and np.sum(too_steep)==0:
                        # if all soil depths are positive, exit the loop. 
                        all_positive = True
                        print('suceeded with timesteps of ', sub_dt)
                    else:
                        print('failed before reaching the end of the itterations with timesteps of ', sub_dt)
                        print('got through ', nst, ' subtimesteps before failing')
                        # otherwise, double the number of sub-timesteps. 
                        number_of_sub_timesteps = number_of_sub_timesteps*2
                        
                        # and put back topography and slopes from before dynamic
                        # timestepping was attempted
                        self.bedrock__elevation[:] = bedrock__elevation_orig.copy()
                        self.soil__depth[:] = soil__depth_orig.copy()
                        self._grid['node']['topographic__elevation'][:] = topographic__elevation_orig.copy()
                        self._grid['node']['flow__receiver_node'][:] = flow__receiver_node_orig.copy()
                        self._grid['node']['topographic__steepest_slope'][:] = topographic__steepest_slope_orig.copy()
                        self._grid['node']['flow__link_to_receiver_node'][:] = flow__link_to_receiver_node_orig.copy()
                        self._grid['node']['flow__sink_flag'][:] = flow__sink_flag_orig.copy()
                    
                    if number_of_sub_timesteps>4000:
                        raise ValueError('dt provided for Hybrid Alluvium '
                                         'Model is so big that in order to be '
                                         'stable the dt has been dynamically '
                                         'subdivided '+str(number_of_sub_timesteps)+
                                         '. times. Revise paramters with a '
                                         'more appropriate dt.')
            
            # if soil depth as calculated by original dt is always positive,
            # and the slopes are stable, continue. 
            else:
                print('still stable')
                pass
            
        


    def calculate_soil_depth(self, dt, soil__depth, deposition_pertime, flooded_nodes):
        """Calculate and return soil depth."""
        
        #now, the analytical solution to soil thickness in time:
        #need to distinguish D=kqS from all other cases to save from blowup!
        
        # Identify flooded nodes
        flooded = np.full(self._grid.number_of_nodes, False, dtype=bool)
        flooded[flooded_nodes] = True        
        
        #distinguish cases:
        blowup = deposition_pertime == self.K_sed * self.q * self.slope

        ##first, potential blowup case:
        #positive slopes, not flooded
        selected_nodes = (self.q > 0) & (blowup==True) & (self.slope > 0) & \
            (flooded==False) & (self.is_not_closed)
        soil__depth[selected_nodes] = self.H_star * \
            np.log((self.sed_erosion_term[selected_nodes] / self.H_star) * dt + \
            np.exp(self.soil__depth[selected_nodes] / self.H_star))
            
        #positive slopes, flooded
        selected_nodes = (self.q > 0) & (blowup==True) & (self.slope > 0) & \
            (flooded==True) & (self.is_not_closed)
        soil__depth[selected_nodes] = (deposition_pertime[selected_nodes] / (1 - self.phi)) * dt   
                        
        #non-positive slopes, not flooded
        selected_nodes = (self.q > 0) & (blowup==True) & (self.slope <= 0) & \
            (flooded==False) & (self.is_not_closed)
        soil__depth[selected_nodes] += (deposition_pertime[selected_nodes] / \
            (1 - self.phi)) * dt    
        
        ##more general case:
        #positive slopes, not flooded
        selected_nodes = (self.q > 0) & (blowup==False) & (self.slope > 0) & \
            (flooded==False) & (self.is_not_closed)
        soil__depth[selected_nodes] = self.H_star * \
            np.log((1 / ((deposition_pertime[selected_nodes] * (1 - self.phi)) / \
            (self.sed_erosion_term[selected_nodes]) - 1)) * \
            (np.exp((deposition_pertime[selected_nodes] * (1 - self.phi) - \
            (self.sed_erosion_term[selected_nodes]))*(dt / self.H_star)) * \
            (((deposition_pertime[selected_nodes] * (1 - self.phi) / \
            (self.sed_erosion_term[selected_nodes])) - 1) * \
            np.exp(self.soil__depth[selected_nodes] / self.H_star)  + 1) - 1))
        
        #places where slope <= 0 but not flooded:
        selected_nodes = (self.q > 0) & (blowup==False) & (self.slope <= 0) & \
            (flooded==False) & (self.is_not_closed)
        soil__depth[selected_nodes] += (deposition_pertime[selected_nodes] / \
            (1 - self.phi)) * dt     
                        
        #flooded nodes:
        selected_nodes = (self.q > 0) & (blowup==False) & (flooded==True) & (self.is_not_closed)
        soil__depth[selected_nodes] += \
            (deposition_pertime[selected_nodes] / (1 - self.phi)) * dt     
        
        return soil__depth
    
    def _calculate_erosion_terms_from_hydrology(self):
        """Calculate erosion terms depending on hydrology method used."""
        #Choose a method for calculating erosion:
        if self.method == 'stochastic_hydrology':        
            self.stochastic_hydrology()        
        elif self.method == 'simple_stream_power':
            self.simple_stream_power()
        elif self.method == 'threshold_stream_power':
            self.threshold_stream_power()
        else:
            raise ValueError('Specify an erosion method!')
    
    def _update_bedrock_elevation(self, dt, bedrock__elevation, soil__depth):
        """Update bedrock elevation."""
        bedrock__elevation[self.q > 0 & self.is_not_closed] += dt * \
                               (-self.br_erosion_term[self.q > 0 & self.is_not_closed] * \
                                (np.exp(-soil__depth[self.q > 0 & self.is_not_closed] / self.H_star)))
        return bedrock__elevation
                               
    def _calculate_qs_in_and_deposition_rate(self):
        """Calculate qs_in and deposition through time."""
         # instantiate a grid for qs_in
        self.qs_in = np.zeros(self.grid.number_of_nodes)            
            
        #iterate top to bottom through the stack, calculate qs
        for j in np.flipud(self.stack):
            if self.q[j] == 0:
                self.qs[j] = 0
            else:
                self.qs[j] = (((self.Es[j]) + (1-self.F_f) * self.Er[j]) / \
                    (self.v_s / self.q[j])) * (1.0 - \
                    np.exp(-self.link_lengths[j] * self.v_s / self.q[j])) + \
                    (self.qs_in[j] * np.exp(-self.link_lengths[j] * \
                    self.v_s / self.q[j]))
            self.qs_in[self.flow_receivers[j]] += self.qs[j]
        
        # create a variable for deposition per time
        self.deposition_pertime = np.zeros(self.grid.number_of_nodes)
        
        # set value for deposition per time. 
        self.deposition_pertime[self.q > 0] = (self.qs[self.q > 0] * \
            (self.v_s / self.q[self.q > 0]))