import hpvsim as hpv
import sciris as sc
import scipy
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import multiprocess
import multiprocess.pool

import optuna as op

import time


seed = 3
name = f"C:\\Users\\fabia\\Documents\\GitHub\\hpvsim\\RawTemp\\13Oct24_2_{seed}"

n_workers = 1   #number of workers per calibration
n_agents = 10e3    
n_trials = 2000
leakiness = 1

verbose = True #set to True for output at the end of every completed trial
datafiles = [   'Set 6\\cancer_incidence_2000.csv',
                'Set 6\\cancers_2020.csv',
                'Set 6\\hpv_prevalence_2020.csv',
                ]
prunings = [(-1, op.pruners.MedianPruner(),'nop'),
            (1, op.pruners.HyperbandPruner(min_resource=1, max_resource=n_trials, reduction_factor=3),'h'),
            (1, op.pruners.MedianPruner(),'m'),
            (1,op.pruners.SuccessiveHalvingPruner(reduction_factor=2),'sh2'),
            (1,op.pruners.SuccessiveHalvingPruner(reduction_factor=3),'sh3'),
            (1,op.pruners.SuccessiveHalvingPruner(reduction_factor=4),'sh4'),
            ]#-1:no pruning ; 1:basic pruning ; 2:leaky pruning ; 3:adaptive pruning


if __name__== "__main__":
    # Sim to be calibrated
    pars = dict(n_agents=n_agents,
                start=1980,
                end=2023, 
                dt=0.25, 
                location='nigeria', #united kingdom
                verbose = 0, #1 means verbose, 0 means not verbose
                )
    sim = hpv.Sim(pars)

    #Parameters for calibration
    genotype_pars = dict(
        hpv16=dict(
            cin_fn=dict(k=[0.5, 0.2, 1.0]),
            dur_cin=dict(par1=[6, 4, 12])
        ),
        hpv18=dict(
            cin_fn=dict(k=[0.5, 0.2, 1.0]),
            dur_cin=dict(par1=[6, 4, 12])
        )
    )
    calib_pars = dict(
            beta=[0.05, 0.010, 0.20],
            f_cross_layer= [0.05, 0.01, 0.2],
            m_cross_layer= [0.05, 0.01, 0.2],
          #  condoms = [0.27,0.1,0.38],
        )

    processtimes = []



    for pruning, pruner, nameext in prunings: #Do one run for each seed
        #Set up calibration object    
        results_to_plot = []#'cancer_incidence', 'asr_cancer_incidence']

        calib = hpv.Calibration(
            sc.dcp(sim),
            calib_pars=calib_pars,
            genotype_pars=genotype_pars,
            extra_sim_result_keys=results_to_plot,
            verbose = False,
            datafiles=datafiles,
            keep_db=True,
            rand_seed=seed,
            name=f"{name}_{nameext}",
            total_trials=n_trials, n_workers=n_workers,
            #leakiness = 0,
            pruner=pruner,
            pruning = pruning #-1:no pruning ; 1:basic pruning ; 2:leaky pruning ; 3:adaptive pruning
        )

        #Calibrate and time calibration
        start_time = time.process_time()
        calib.calibrate(die=False,plots=["learning_curve", "timeline"], detailed_contplot=None,save_to_csv=None)
        end_time = time.process_time()
        processtimes.append(f"{nameext}: {end_time - start_time}")
        print(f"Duration (process time, unitless) = {processtimes[-1]}")
        #calib.plot_learning_curve()
        #calib.plot_timeline()
    print()
    print(f"Processtimes={processtimes}")