# Import HPVsim

import hpvsim as hpv

from calibration import *

rand_seed = 1

# Configure a simulation with some parameters


import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import scipy

import sciris
import optuna



def normal_confidence_interval(data, alpha):
        '''
        Returns (l,mu,u) where l and u are the lower and upper bounds respectively of the 100(1-alpha)%-confidence interval for the mean of the provided list of data, and mu is the mean.
        We assume the data is normally distributed.

        We use the result that for X sampled iid from N(mu, v), (Xbar-mu)/(S/sqrt(n)) ~ t_{n-1}, where Xbar is the sample mean and S^2 is the sample variance [Niel Laws, 2023, A9 Statistics Lecture Notes Oxford University] 

        Parameters:
            data    = a list of numeric data
            alpha   = a real number in (0,1)
        
        Pre:
            alpha<-(0,1)
            len(data) >= 2 (else its sample variance is ill-defined)
        '''
        n=len(data)
        #Calculate sample mean and sample variance
        Xbar=0
        for x in data:
            Xbar += x
        Xbar /= n

        S2 = 0
        for x in data:
            S2 += (x-Xbar) ** 2
        S2 /= (n-1)
        S = S2 ** 0.5

        #Calculate CI
        offset = scipy.stats.t.cdf(1-alpha/2, n-1) * S/(n ** 0.5)
        return (Xbar-offset,Xbar, Xbar+offset)



calibration_start_time = sciris.tic()
study_cutoff_time = 60*0.5 #10mins, in seconds
def time_cutoff_callback(study, trial):
     '''
     Stops a study {study_cutoff_time} seconds after {calibration_start_time}, to stop a study after a certain amount of time, even if the desired number of trials has not yet been reached
     '''
     time_diff = sciris.toc(calibration_start_time, output=True) #output=whether to return the time difference
     #print(f"time_diff = {time_diff}")
     if time_diff >= study_cutoff_time:
          study.stop()


if __name__ == "__main__":
    pars = dict(n_agents=10e3, 
                start=1980, end=2025, dt=0.25,
                location='nigeria', #united kingdom
                verbose = 0,
                #rand_seed = rand_seed,
                debut= dict(f=dict(dist='normal', par1=16.0, par2=3.1), #changing the distribution for sexual debut ages, as they are signficantly different in the uk to nigeria
                            m=dict(dist='normal', par1=16.0, par2=4.1))) #TODO: find were this data comes from

    

    

    # Specify some parameters to adjust during calibration.
    # The parameters in the calib_pars dictionary don't vary by genotype,
    # whereas those in the genotype_pars dictionary do. Both kinds are
    # given in the order [best, lower_bound, upper_bound].

    calib_pars = dict(
            beta=[0.05, 0.010, 0.20],
            f_cross_layer= [0.05, 0.01, 0.2],
            m_cross_layer= [0.05, 0.01, 0.2],
          #  condoms = [0.27,0.1,0.38],
        )

    

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
    

    # List the datafiles that contain data that we wish to compare the model to:
    datafiles=[#'fabiandata\\calib14jan23\\new_cervical_cancer_cases.csv',
               #'fabiandata\\calib14jan23\\genotype_distrib_cancer.csv',
               'docs\\tutorials\\nigeria_cancer_cases_super_truncated.csv',
                'fabiandata\\calib14jan23\\2010data2.csv',
               'fabiandata\\calib14jan23\\2010data3.csv'
                ]

    

    # List extra results that we don't have data on, but wish to include in the
    # calibration object so we can plot them.
    results_to_plot = []#'cancer_incidence', 'asr_cancer_incidence']

    
    #calib.plot(res_to_plot=5);
    #calib.plot(res_to_plot=1) #single best result is plotted alongside the data


    ### Generate results ###
    no_trials = 1000000
    no_workers = 20
    trials_managed = {
         1:[],
      #   0.5:[],
         0:[]
    }

    no_runs = 2
    results = {1    :[],
        #       0.5  :[],
               0    :[]
    }

    pruners = [
         optuna.pruners.HyperbandPruner(),
         optuna.pruners.MedianPruner(),
         optuna.pruners.SuccessiveHalvingPruner()
    ]
    i = 0
    for pruner in pruners:
        i+=1
        bigname = f"CalibrationRawResults\\hpvsim_calubration_ukdata_14thJuly2024_13_prune{i}"
        for leakiness  in results.keys():
            for run in range(1,no_runs+1):
                print(f"---------- Leakiness {leakiness}, Run {run}/{no_runs} ----------")
                name = f"{bigname}_leakiness{leakiness}_run{run}"
                calib = hpv.Calibration(
                    hpv.Sim(pars),
                    calib_pars=calib_pars,
                    genotype_pars=genotype_pars,
                    extra_sim_result_keys=results_to_plot,
                    verbose = False,
                    datafiles=datafiles,
                    keep_db=True,
                    name=name,
                    #rand_seed=rand_seed,
                    total_trials=no_trials, n_workers=no_workers,
                    leakiness = leakiness,
                    pruning = 2 #-1:no pruning ; 1:basic pruning ; 2:leaky pruning ; 3:adaptive pruning
                )
                calibration_start_time = sciris.tic()
                calib.calibrate(die=False,plots=["learning_curve", "timeline"], detailed_contplot=[],save_to_csv=None, optuna_callback=time_cutoff_callback)
                #calib.plot_learning_curve()
                #calib.plot_timeline()

                results[leakiness].append(calib.learning_curve.data[1].y[-1])

                trials_managed[leakiness].append(calib.learning_curve.data[1].x[-1])


        ### Present Results ###
        print("-----------------RESULTS-----------------")
        print(results)
        print("------------------------------------------")
        print("-----------------TRIALS MANAGED-(ensure these are about the same for each leakiness, to demonstrate computer is giving a 'fair chance' to each)----------------")
        print(trials_managed)
        print("------------------------------------------")
        #Also, save these results to a file
        with open(f"{bigname}_results.txt", "a") as f:
            print("-----------------RESULTS-----------------", file=f)
            print(results,  file=f)
            print("------------------------------------------", file=f)
            print("-----------------TRIALS MANAGED-(ensure these are about the same for each leakiness, to demonstrate computer is giving a 'fair chance' to each)----------------",  file=f)
            print(trials_managed, file=f)
            print("------------------------------------------", file=f)

        
        LBs = []
        means = []
        UBs = []

        for leakiness in results.keys():
            lb, mean, ub = normal_confidence_interval(results[leakiness], 0.05)
            LBs.append(lb)
            means.append(mean)
            UBs.append(ub)


        fig = go.Figure()

        #Add a trace for the 95% CI
        fig.add_trace(go.Scatter(x=list(results.keys()), y=UBs, fill=None, line_color='green',mode='lines',line_width=0,showlegend=False))
        fig.add_trace(go.Scatter(x=list(results.keys()), y=LBs, fill='tonexty',line_color='green', mode='lines', line_width=0,showlegend=False))
        fig.add_trace(go.Scatter(x=list(results.keys()), y=means, name='Average' ,fill=None,line=dict(color='black', width=2)))
        fig.update_layout(xaxis_title="Leakiness", yaxis_title='Best Loss', title=f'Loss after {study_cutoff_time/60} minutes')#{no_trials} trials')
        fig.update_xaxes(type='category')


        fig.show()
