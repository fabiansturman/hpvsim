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





if __name__ == "__main__":
    pars = dict(n_agents=10,#10e3, 
                start=1975, end=2025, dt=0.25,
                location='nigeria', #united kingdom
           #     verbose = 0,  #comment this to make the sim verbose
              #  rand_seed = rand_seed,
             #   debut= dict(f=dict(dist='normal', par1=16.0, par2=3.1), #changing the distribution for sexual debut ages, as they are signficantly different in the uk to nigeria
              #              m=dict(dist='normal', par1=16.0, par2=4.1)) #TODO: refernece source of data
    )
    

    

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
               'Set 3\\cancer_incidence_2000.csv',
               'Set 3\\cancers_2010.csv',
               'Set 3\\hpv_prevalence_2020.csv',
                ]

    

    # List extra results that we don't have data on, but wish to include in the
    # calibration object so we can plot them.
    results_to_plot = []#'cancer_incidence', 'asr_cancer_incidence']


    ### Generate results ###
    no_trials = 500
    no_workers = 20
    no_runs = 1#10
    

    pruners = [ #If adding anything to here, make sure I update prunernames (below) appropriately
     #    optuna.pruners.HyperbandPruner(min_resource=1, max_resource=no_trials, reduction_factor=3),
         optuna.pruners.MedianPruner(),
      #   optuna.pruners.SuccessiveHalvingPruner(),
    ]

    prunernames = [
      #  'Hyperband',
        'Median',
     #   'Succ. Halving',
    ]

    colours = [     #I need at least as many colours listed here as I have pruners
                'blue', 
                'green',
                'beige',
                'hotpink'
    ]

    fig = go.Figure()
    dur_fig = go.Figure()

    i = 0
    for pruner in pruners:
        i+=1

        #Storing data from the calibrations - keys in these dictionaries are the leakinesses
        trials_managed = {      
             #   1:[],
            #   0.5:[],
                0:[]
            }
        results = {         
            #    1    :[],   
            # 0.5  :[],
                0    :[]
        }
        durations = {
         #   1    :[],   
        # 0.5  :[],
            0    :[]
        }


        #We need the absolute path - I have been being cheeky using relative paths so far
        bigname = f"C:\\Users\\fabia\\Documents\\GitHub\\hpvsim\\RawTemp\\t42p{i}" #Update the number before 'p' here to keep files stored in the temp files folder unique for a study
        for leakiness  in results.keys():
            for run in range(1,no_runs+1):
                print(f"---------- Leakiness {leakiness}, Run {run}/{no_runs} ----------")

                name = f"{bigname}_{leakiness}_{run}"


                 #If this JournalStorage syntax works, stick with that and implement that - make it the default storage option
                #Also I should confirm that it is not (much, perhaps?) slower than using SQLlite - by redoing some previous experiemnt

                journal_file_path = f"{name}.log"
                lock_obj = optuna.storages.JournalFileOpenLock(journal_file_path)
       #         try:
        #            lock_obj.release() 
         #       except RuntimeError:
          #          print("Tried to release lock on database file, but there was none. Either file does not exist, or was released properly at end of last program which had it.")
#Perhaps this is creating the error -I will just do new file names then ig hmmm

                calib = hpv.Calibration(
                    hpv.Sim(pars),
                    calib_pars=calib_pars,
                    genotype_pars=genotype_pars,
                    extra_sim_result_keys=results_to_plot,
                    verbose = True,
                    datafiles=datafiles,
                    keep_db = True,     #When using SQLlite storage, need to keep this to True to work on my setup (else there is an error at the start) - appears that the  calibration.remove_db function creates a database file when trying to delete the study, and then this file can't be deleted because it remains open somehow? 
                    storage= optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(journal_file_path,lock_obj=lock_obj)), #for multithreading, optuna reccomends this storage mode instead of the default SQLlite.
                    name=name,
                 #   rand_seed=rand_seed,
                    total_trials=no_trials, n_workers=no_workers,
                    leakiness = leakiness,
                    pruner = pruner,
                    pruning = -1 #-1:no pruning ; 1:basic pruning ; 2:leaky pruning ; 3:adaptive pruning
                            #HAVE SET PRUNING TO NO PRUNING!!!!!!
                )
                calibration_start_time = sciris.tic()
                calib.calibrate(die=False,plots=["learning_curve", "timeline"], detailed_contplot=[],save_to_csv=None)
                calibration_duration = sc.toc(calibration_start_time, output=True)
                calib.plot_learning_curve()
                calib.plot_timeline()
                


                results[leakiness].append(calib.learning_curve.data[1].y[-1])
                trials_managed[leakiness].append(calib.learning_curve.data[1].x[-1])
                durations[leakiness].append(calibration_duration)


        ### Present Results ###
        print(f"-----------------RESULTS ({pruners[i-1]})-----------------")
        print(results)
        print("------------------------------------------")
        print("-----------------TRIALS MANAGED-(ensure these are about the same for each leakiness, to demonstrate computer is giving a 'fair chance' to each)----------------")
        print(trials_managed)
        print("------------------------------------------")
        print("-----------------DURATIONS----------------")
        print(durations)
        print("------------------------------------------")
        #Also, save these results to a file
        with open(f"{bigname}_results.txt", "w") as f:
            print("-", file=f)
        with open(f"{bigname}_results.txt", "a") as f:
            print(f"-----------------RESULTS ({pruners[i-1]})-----------------", file=f)
            print(results,  file=f)
            print("------------------------------------------", file=f)
            print("-----------------TRIALS MANAGED-(ensure these are about the same for each leakiness, to demonstrate computer is giving a 'fair chance' to each)----------------",  file=f)
            print(trials_managed, file=f)
            print("------------------------------------------", file=f)
            print("-----------------DURATIONS----------------",  file=f)
            print(durations, file=f)
            print("------------------------------------------", file=f)




        mins = []
        LBs = []
        means = []
        UBs = []
        maxs = []

        dur_mins = []
        dur_LBs = []
        dur_means = []
        dur_UBs = []
        dur_maxs = []

        for leakiness in results.keys():
            lb, mean, ub = normal_confidence_interval(results[leakiness], 0.05)
            LBs.append(lb)
            means.append(mean)
            UBs.append(ub)

            mins.append(min(results[leakiness]))
            maxs.append(max(results[leakiness]))

            dur_lb, dur_mean, dur_ub = normal_confidence_interval(durations[leakiness], 0.05)
            dur_LBs.append(dur_lb)
            dur_means.append(dur_mean)
            dur_UBs.append(dur_ub)

            dur_mins.append(min(durations[leakiness]))
            dur_maxs.append(max(durations[leakiness]))



        colour = colours[i-1]

        #Add a trace for the 95% CI of durations
        dur_fig.add_trace(go.Scatter(x=list(results.keys()), y=dur_UBs, fill=None, line_color=colour,mode='lines',line_width=0,showlegend=False))
        dur_fig.add_trace(go.Scatter(x=list(results.keys()), y=dur_LBs, fill='tonexty',line_color=colour, mode='lines', line_width=0,showlegend=False))
        dur_fig.add_trace(go.Scatter(x=list(results.keys()), y=dur_means, name=f'Average, pruner {prunernames[i-1]}' , mode='lines',fill=None,line=dict(color=colour, width=2)))

        #uncomment this to have min/max points added to the plot, but not when comparing different pruners
        dur_fig.add_trace(go.Scatter(x=list(results.keys()), y=dur_mins, fill=None, mode='markers', 
                                            name=f'Min, pruner {prunernames[i-1]}',
                                            marker=dict(
                                                color=colour,
                                                size=10,
                                            )))
        dur_fig.add_trace(go.Scatter(x=list(results.keys()), y=dur_maxs, fill=None,mode='markers',
                                            name=f'Max, pruner {prunernames[i-1]}',
                                             marker=dict(
                                                color=colour,
                                                size=10,
                                            )))

        dur_fig.update_layout(xaxis_title="Leakiness", yaxis_title='Time (s)', title=f'Time to run {no_trials} trials (counting pruned trials)')
        dur_fig.update_xaxes(type='category')


        #Add a trace for the 95% CI of losses
        fig.add_trace(go.Scatter(x=list(results.keys()), y=UBs, fill=None, line_color=colour,mode='lines',line_width=0,showlegend=False))
        fig.add_trace(go.Scatter(x=list(results.keys()), y=LBs, fill='tonexty',line_color=colour, mode='lines', line_width=0,showlegend=False))
        fig.add_trace(go.Scatter(x=list(results.keys()), y=means, name=f'Average, pruner {prunernames[i-1]}' , mode='lines',fill=None,line=dict(color=colour, width=2)))

        #uncomment this to have min/max points added to the plot, but not when comparing different pruners
        fig.add_trace(go.Scatter(x=list(results.keys()), y=mins, fill=None, mode='markers', 
                                                name=f'Min, pruner {prunernames[i-1]}',
                                                marker=dict(
                                                color=colour,
                                                size=10,
                                                )))
        fig.add_trace(go.Scatter(x=list(results.keys()), y=maxs, fill=None,mode='markers', 
                                            name=f'Max, pruner {prunernames[i-1]}',
                                            marker=dict(
                                                color=colour,
                                                size=10,
                                            )))

        fig.update_layout(xaxis_title="Leakiness", yaxis_title='Best Loss', title=f'Loss after {no_trials} trials')
        fig.update_xaxes(type='category')


    fig.show()
    dur_fig.show()
