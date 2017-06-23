#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:13:25 2017

@author: barnhark
"""
import numpy as np

def hybrid_H_etab_Qs_solver(v, 
                            Ht, 
                            eta_bt, 
                            flow_recievers, 
                            q, 
                            Q, 
                            K_sed, 
                            K_br, 
                            omega_sed, 
                            omega_br, 
                            H_star, 
                            F_f, 
                            phi, 
                            v_s, 
                            dt, 
                            dx, 
                            n, 
                            flooded,
                            H_boundary_condition_inds,
                            eta_boundary_condition_inds, 
                            qs_boundary_condition_inds,
                            H_bc,
                            eta_bbc,
                            qs_bc):
    
    """Calculation of residuals for H, eta_b, and Qs for global solution.
    
    More text here!
    """
    # extract the number of nodes. 
    
    num_nodes = int((v.size + len(H_bc) + len(eta_bbc) + len(qs_bc))/3)
    
    node_id = np.arange(num_nodes)
    
    # extract H, eta_b, and Qs
    num_H = num_nodes - len(H_bc)
    num_eta = num_nodes - len(eta_bbc)
    num_Q = num_nodes - len(qs_bc)

    # chunk v into correct parts for H, eta_b, and Qs    
    H = v[0:num_H]
    eta_b = v[num_H:num_H+num_eta]
    Qs = v[num_H+num_eta:num_H+num_eta+num_Q]
    
    # put the boundary condition values in the right place. 
    for i in range(len(H_boundary_condition_inds)):
        ind = H_boundary_condition_inds[i]
        H = np.insert(H, ind, H_bc[i])
        
    for i in range(len(eta_boundary_condition_inds)):
        ind = eta_boundary_condition_inds[i]
        eta_b = np.insert(eta_b, ind, eta_bbc[i])
        
    for i in range(len(qs_boundary_condition_inds)):
        ind = qs_boundary_condition_inds[i]
        Qs = np.insert(Qs, ind, qs_bc[i])
    
    # calculate slope and topographic elevation for ease
    eta = H + eta_b
    
    S = (eta - eta[flow_recievers]) / dx[node_id]

    S[S<0.0] = 0.0 # make slopes of less than zero, effectively flat. 
    S[flooded] = 0.0 # make slopes when node flooded zero, so no erosion 
    # occurs, but depostion can continue. 
    
    dQsdx = (Qs - Qs[flow_recievers]) / dx[node_id]
    
    # calculate E_r and E_s
    E_r = (K_br * q * np.power(S, n))
    E_s = (K_sed * q * np.power(S, n))
    
    # Calculate E_r and E_s terms including the thresholds, omega_br and 
    # omega_sed. If thresholds are zero, fix. 
    if type(omega_br) is float:
        if omega_br>0:
            Er_term = (E_r-omega_br*(1.0-np.exp(-E_r/omega_br)))
        else:
            Er_term = E_r
    else:
        if np.all(omega_br>0):
            Er_term = (E_r-omega_br*(1.0-np.exp(-E_r/omega_br)))
        else:
            Er_term = (E_r-omega_br*(1.0-np.exp(-E_r/omega_br)))
            Er_term[omega_br==0] = E_r
            
    if type(omega_sed) is float:
        if omega_sed>0:
            Es_term = (E_s-omega_sed*(1.0-np.exp(-E_s/omega_sed)))
        else:
            Es_term = E_s
    else:
        if np.all(omega_sed>0):
            Es_term = (E_s-omega_sed*(1.0-np.exp(-E_s/omega_sed)))
        else:
            Es_term = (E_s-omega_sed*(1.0-np.exp(-E_s/omega_sed)))
            Es_term[omega_sed==0] = E_s
    
    # behaviour when Q=0 and/or flooded nodes, slope . 

    # calculate settling, make sure this is OK (and zero) when Q is zero. 
    settling_term = np.zeros(H.shape)
    settling_term[Q>0] = (v_s * Qs[Q>0])/Q[Q>0]
    
    # residual function for eta_b
    f_eta_b = -((eta_b - eta_bt)/dt) - Er_term * (np.exp(-H/H_star)) 
    
    # resiual function for H
    f_H = -((H - Ht)/dt) + (settling_term * (1.0)/(1.0-phi)) - Es_term*(1.0-np.exp(-H/H_star)) 
    
    # residual function for Q
    f_Qs = -dQsdx + ((Es_term * (1.0 - np.exp(-H/H_star))) + (1.0 - F_f) * Er_term * (np.exp(-H / H_star))) - settling_term
    
    # delete the correct portions of f_H, f_eta_b, and f_Qs related to the bcs.
    f_H = np.delete(f_H, H_boundary_condition_inds)
    f_eta_b = np.delete(f_eta_b, eta_boundary_condition_inds)
    f_Qs = np.delete(f_Qs,qs_boundary_condition_inds)
    
    f = np.concatenate((f_H, f_eta_b, f_Qs), axis=0)
    
    return f