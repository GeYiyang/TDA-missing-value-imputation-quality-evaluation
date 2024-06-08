# -*- coding: utf-8 -*-
"""
Created on Sat Feb 8 16:47:46 2024

@author: Ge Yiyang
"""
import numpy as np
import pandas as pd 
from sklearn.cluster import DBSCAN
import csv
import gower
import statmapper as stm
from sklearn_tda import MapperComplex
from statmapper import compute_topological_features
import gudhi as gd
from sklearn.manifold import MDS
from sklearn_tda import *
import random

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

np.random.seed(100)

def M(data, res, gain):
    """
    Constructs a Mapper complex from data using specified resolution and gain.
    
    Parameters:
    - data: DataFrame containing the dataset.
    - res: the resolution parameter of the Mapper.
    - gain: the gain parameter of the Mapper.
    
    Returns:
    - A tuple (MapperComplex, fil), where 'MapperComplex' is the constructed Mapper complex
      and 'fil' is the filter function used.
    """
    # Determine which features are categorical
    is_categ = categ['categorical'].values == 1
    # Calculate the Gower distance matrix considering categorical features
    gmat = gower.gower_matrix(data, cat_features=is_categ)
    # Apply Multidimensional Scaling to the Gower matrix
    fil = MDS(n_components=2, dissimilarity='precomputed').fit_transform(gmat)
    # Ensure filter values have the correct dimensions
    if fil.ndim == 1:
        fil = fil.reshape(-1, 1)
    n_filters = np.shape(fil)[1]
    # Define Mapper parameters
    params = {"filters": fil, "filter_bnds": np.array([[np.nan] * n_filters]), "colors": fil, 
              "resolutions": np.array([res] * n_filters), "gains": np.array([gain] * n_filters),
              "inp": "distance matrix", "clustering": DBSCAN(metric='precomputed', eps=0.4)}
    # Create and fit the Mapper complex
    M = MapperComplex(**params).fit(gmat)
    return M, fil

def B(C1, C2, topo_type='connected_components'):
    """
    Computes the bottleneck distance between two Mapper complexes.
    
    Parameters:
    - C1, C2: Tuples containing Mapper complexes and their filter functions.
    - topo_type: String specifying the type of topological feature to consider.
    
    Returns:
    - Float representing the bottleneck distance between topological features of C1 and C2.
    """
    assert topo_type in ['connected_components', 'downbranch', 'upbranch', 'loop']
    M1, fil1 = C1
    M2, fil2 = C2
    # Compute topological features of both complexes
    dgm_1, _ = compute_topological_features(M1, func=fil1, func_type="data", topo_type=topo_type)
    dgm_2, _ = compute_topological_features(M2, func=fil2, func_type="data", topo_type=topo_type)
    # Prepare data for bottleneck distance calculation
    npts, npts_boot = len(dgm_1), len(dgm_2)
    D1 = np.array([[dgm_1[pt][1][0], dgm_1[pt][1][1]] for pt in range(npts) if dgm_1[pt][0] <= 1]) 
    D2 = np.array([[dgm_2[pt][1][0], dgm_2[pt][1][1]] for pt in range(npts_boot) if dgm_2[pt][0] <= 1])
    # Calculate the bottleneck distance
    B_value = gd.bottleneck_distance(D1, D2)
    return B_value

if __name__ == '__main__':
    # Load data, data1 is the complete case, data2 is the imputed data
    data1 = pd.read_csv('complete_case.csv')
    data2 = pd.read_csv('imputed.csv')
    categ = pd.read_csv('categorical_items.csv')[['index', 'categorical']]
    # Output the shape of the datasets
    print(data1.shape)
    print(data2.shape)
    # Define resolution and gain values, select the range of "resolution" and "gain" carefully.
    res_list = [5, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25]
    gain_list = [0.1, 0.2, 0.3, 0.4]
    method = ['connected_components', 'downbranch', 'upbranch', 'loop']
    output_A = {}
    # Run experiments for each combination of resolution and gain
    for res in res_list:
        for gain in gain_list:
            C1 = M(data1, res, gain)
            C2 = M(data2, res, gain)
            Bs = []
            for i in method:
                b_value = B(C1, C2, i)
                Bs.append(b_value)
            B_value = max(Bs)
            output_A[(res, gain)] = B_value
            print(f"res = {res}, gain = {gain}: B_value = {B_value}")

            
# Load data
data1 = pd.read_csv('complete_case.csv')
data2 = pd.read_csv('imputed.csv')
# Load categorical feature indicators
categ = pd.read_csv('categorical_items.csv')[['index', 'categorical']]

# Define the number of bootstrap samples
num_sample = 100
# Define the types of topological features to evaluate
method = ['connected_components', 'downbranch', 'upbranch', 'loop']
output_B = {}

# Iterate over each combination of resolution (res) and gain settings defined previously
for res in res_list:
    for gain in gain_list:
        # Lists to store Mapper complexes from bootstrapped samples
        samp1 = []
        samp2 = []
        # Create bootstrap samples and compute their Mapper complexes
        for i in range(num_sample):
            # Randomly sample indices for data1 and data2, select the sample carefully
            r1 = random.sample(range(0, len(data1)), 550)
            r2 = random.sample(range(0, len(data2)), 550)
            # Generate Mapper complexes for randomly selected data
            samp1.append(M(data1.iloc[r1], res, gain))
            samp2.append(M(data2.iloc[r2], res, gain))
        # Combine and shuffle the samples to randomize pairs
        samp = samp1 + samp2
        random.shuffle(samp)
        # Split the shuffled list back into two groups
        samp_1 = samp[:num_sample]
        samp_2 = samp[num_sample:]
        # Store the maximum bottleneck distance from different topological feature comparisons
        max_Bs = []
        for t in range(num_sample):
            Bs = []
            # Compute bottleneck distances for each topology method
            for j in method:
                Bs.append(B(samp_1[t], samp_2[t], j))
            max_Bs.append(max(Bs))
        # Record the results for the current resolution and gain settings
        output_B[(res, gain)] = max_Bs
        # Output the results
        print("res = {}, gain = {}: result = {}".format(res, gain, max_Bs))

# Iterate over each combination of resolution and gain in Output A
for comb in output_A:
    res, gain = comb
    B_value = output_A[comb]
    # Count how many distances in Output B are greater than or equal to the distance in Output A
    count = sum(1 for distance in output_B[comb] if distance >= B_value)
    # Calculate the proportion of distances greater than or equal to Output A, interpreted as a p-value
    p_value = count / 100
    print(f"p_value for res = {res}, gain = {gain}: {p_value}")
    




