#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from hd2d_src.HopTrack.core.Local_Density_Calculator import Local_Density_Calculator
from hd2d_src.HopTrack.utils import *


_GLOBAL_A = None
_GLOBAL_MU = None


def init_worker(a, mu):
    """
    Initialize global variables in each worker process.
    On Linux, this works efficiently because multiprocessing uses fork by default.
    """
    global _GLOBAL_A, _GLOBAL_MU
    _GLOBAL_A = a
    _GLOBAL_MU = mu


def compute_mu(a, dt):
    """
    Compute squared displacement at a given lag time dt:
        mu[i] = |r_i(dt) - r_i(0)|^2
    """
    frames = np.asarray(a.frames)

    pos0 = frames[0, :, 2:4]
    pos_dt = frames[dt, :, 2:4]

    disp = pos_dt - pos0
    mu = np.sum(disp**2, axis=1)

    return mu


def pearson_corr(x, y):
    """
    Pearson correlation coefficient between two 1D arrays.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)

    return np.corrcoef(x[mask], y[mask])[0, 1]


def calc_corr_for_R(R):
    """
    Worker function for one probing radius R.
    Uses global _GLOBAL_A and _GLOBAL_MU.
    """
    global _GLOBAL_A, _GLOBAL_MU

    ld = Local_Density_Calculator(_GLOBAL_A, R, [0])
    ld = np.asarray(ld)
    LUV = np.array(1 - ld) * (R * 2) ** 2

    corr = pearson_corr(LUV, _GLOBAL_MU)

    return R, corr


if __name__ == "__main__":
    for tag in [500, 2000, 10000, 25000]:
        obj_a_path = f'/home/xiaochu/Public/LUV_in_HD/data/input/0805/a_cut_0_{tag}.obj'
        out_save = f'/home/xiaochu/Public/LUV_in_HD/apps/correlation_factor_vs_R/correlation_factor_vs_R_dt_{tag}.txt'

        with open(obj_a_path, "rb") as f:
            custom_unpickler = CustomUnpickler(f)
            a = custom_unpickler.load()

        with open(obj_a_path, "rb") as f:
            custom_unpickler = CustomUnpickler(f)
            a_unwrap = custom_unpickler.load()

        a_unwrap.unwrap_vec()

        Rc = [2, 4, 6, 8, 10, 12, 14, 16, 18]
        dt = 1

        mu = compute_mu(a_unwrap, dt)

        max_workers = min(len(Rc), os.cpu_count() or 1)
        # max_workers = 2

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker,
            initargs=(a, mu)
        ) as executor:
            results = list(executor.map(calc_corr_for_R, Rc))

        R_values = np.array([r for r, c in results])
        corr_values = np.array([c for r, c in results])

        data = np.column_stack([
            R_values,
            corr_values
        ])

        np.savetxt(
            out_save,
            data,
            header="R corr",
            fmt=["%g", "%.10f"]
        )


        print("dt =", dt)
        print("R =", R_values)
        print("corr =", corr_values)
        print("Saved to:", out_save)