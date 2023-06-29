#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# define functions to calculate PS, following py21cmmc
import numpy as np
from powerbox.tools import get_power
from scipy.interpolate import interp1d

tmp1 = 333
def compute_power(
   box,
   length,
   k_bins_edges,
   log_bins=True,
   ignore_kperp_zero=True,
   ignore_kpar_zero=False,
   ignore_k_zero=False,
):
    # Determine the weighting function required from ignoring k's.
    k_weights = np.ones(box.shape, dtype=np.float32)
    n0 = k_weights.shape[0]
    n1 = k_weights.shape[-1]

    if ignore_kperp_zero:
        k_weights[n0 // 2, n0 // 2, :] = 0
    if ignore_kpar_zero:
        k_weights[:, :, n1 // 2] = 0
    if ignore_k_zero:
        k_weights[n0 // 2, n0 // 2, n1 // 2] = 0

    res = get_power(
        box,
        boxlength=length,
        bins=k_bins_edges,
        bin_ave=False,
        get_variance=False,
        log_bins=log_bins,
        k_weights=k_weights,
    )

    res = list(res)
    k = res[1]
    if log_bins:
        k = np.exp((np.log(k[1:]) + np.log(k[:-1])) / 2)
    else:
        k = (k[1:] + k[:-1]) / 2

    res[1] = k
    return res

def powerspectra(lightcone_box, BOX_LEN, HII_DIM,lightcone_redshifts, nchunks=15, logk=True):
    data = []
    # Split the lightcone into chunks and find the redshift values that correspond to the middle of
    # these chunks
    chunk_indices_boundaries_array = np.round(np.linspace(0,lightcone_box.shape[2]-1,nchunks+1))
    chunk_indices_boundaries = chunk_indices_boundaries_array.astype(int).tolist()
    chunk_indices = ((chunk_indices_boundaries_array[1:]+chunk_indices_boundaries_array[:-1])/2).astype(int).tolist()
    z_boundaries = lightcone_redshifts[chunk_indices_boundaries]
    redshift_grid = interp1d(chunk_indices_boundaries,z_boundaries, kind='cubic')
    z = redshift_grid(chunk_indices)
    # Compute the power spectrum for each chunk
    for i in range(nchunks):
        # Get the chunk's boundaries
        start = chunk_indices_boundaries[i]
        end = chunk_indices_boundaries[i + 1]
        chunklen = (end - start) * BOX_LEN/lightcone_box.shape[0]
        # Get the edges of the k-bins (based on the bins of 21cmFast version 2)
        Delta_k = 2*np.pi/BOX_LEN # 1/Mpc
        k_max = Delta_k*HII_DIM # 1/Mpc
        k_factor = 1.5
        k_bins_edges = []
        k_ceil = Delta_k
        while (k_ceil < k_max):
            k_bins_edges.append(k_ceil)
            k_ceil *= k_factor
        k_bins_edges = np.array(k_bins_edges)
        # Compute the power spectrum
        power, k = compute_power(lightcone_box[:, :, start:end],
                                (BOX_LEN, BOX_LEN, chunklen),
                                k_bins_edges,
                                log_bins=logk)
        data.append({"k": k, "delta": power * k ** 3 / (2 * np.pi ** 2)})
    # Return output
    return data, z

