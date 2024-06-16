# Import HPVsim
import hpvsim as hpv

import optuna as op
import sciris as sc
import numpy as np

rand_seed = 5 #random seed both for our simulation and for optuna; set to None to assign variable seeds to both

if __name__ == "__main__":#to allow for using several workers in parellell

    # Create the age analyzers.
    az = hpv.age_results(
        result_args=sc.objdict(
            # The keys of this dictionary are any results you want by age, and can be any key of sim.results. We are analyzing these 4 values by ages in 2000, 2010, 2015,
            hpv_prevalence=sc.objdict( 
                years=[2000,2010,2020],
                edges=np.array([0., 15., 20., 25., 30., 35. ,40., 45., 50., 55., 65.,70.,75.,80.,85., 100.]), #define the extremities of the age buckets for each analyser
            ),
            cancer_incidence=sc.objdict(
                years=[2000,2010,2020],
                edges=np.array([0., 15., 20., 25., 30., 35. ,40., 45., 50., 55., 65.,70.,75.,80.,85., 100.]), 
            ),
        )
    )

    #Define the interventions that will be presetn in the simulation, mimicking that found in the uk (kinda)
    #prob = 0.6 # prob = 60% means we vaccinate 60% of girls (or girls and boys if we add sex = [0,1])

    #vx = hpv.routine_vx(prob=prob, start_year=2015, age_range=[9,10], product='bivalent')
        #TODO: I CANT SEEM TO GET INTERVENTIONS ADDED TO MY SIM WHEN CALIBRATING, WITHOUT GETTING A VALUEERROR 'PROVIDE EITHER A LIST OF YEARS OR A START YEAR, OR BOTH'

    # Configure a simulation with some parameters
    pars = dict(n_agents=10e3, 
                start=1970,
                end=2023, 
                dt=0.25, 
                location='nigeria', #united kingdom
                genotypes=[16, 18, 'hi5'],
                verbose = 0, #1 means verbose, 0 means not verbose
                rand_seed = rand_seed,
                analyzers=[az]
                )
    sim = hpv.Sim(pars)
    #print(f"Simulation's parameter keys: <{sim.pars.keys()}>")

    # Specify some parameters to adjust during calibration.
    # The parameters in the calib_pars dictionary don't vary by genotype,
    # whereas those in the genotype_pars dictionary do. Both kinds are
    # given in the order [best, lower_bound, upper_bound].
    calib_pars = dict(
            beta=[0.25, 0.010, 0.7], #consider changing the upper bound to 0.20, not 0.7
            hpv_control_prob=[.2, 0, 1]
        )

    genotype_pars = dict(
        hpv16=dict(
            cin_fn=dict(k=[0.5, 0.2, 1.0]),
            dur_cin=dict(par1=[6, 4, 12])
        ),
        hpv18=dict(
            cin_fn=dict(k=[0.5, 0.2, 1.0]),
            dur_cin=dict(par1=[6, 4, 12])
        ),
        hi5=dict(
            cin_fn=dict(k=[0.5, 0.2, 1.0]),
            dur_cin=dict(par1=[6, 4, 12])
        )
    )

    # List the datafiles that contain data that we wish to compare the model to: #TODO: confirm with RObyn that so far the model deoes not work if we wish to plot teh same kind of data from a datafile over more than 1 year. right?
    datafiles=[ 'docs\\tutorials\\nigeria_cancer_cases.csv',
    #            'docs\\tutorials\\nigeria_cancer_cases_super_truncated.csv',
                'docs\\tutorials\\nigeria_cancer_types.csv',
             #   'fabiandata\\calib14jan23\\2000data1.csv',
     #           'fabiandata\\calib14jan23\\2000data2.csv',
      #          'fabiandata\\calib14jan23\\2000data3.csv',
     #           'fabiandata\\calib14jan23\\2010data1.csv',
     #           'fabiandata\\calib14jan23\\2010data2.csv',
               # 'fabiandata\\calib14jan23\\2010data3.csv',
       # 'fabiandata\\calib14jan23\\cancer_deaths.csv',
        #       'fabiandata\\calib14jan23\\new_cervical_cancer_cases.csv',
         #      'fabiandata\\calib14jan23\\genotype_distrib_cancer.csv'
               ]

    # List extra results that we don't have data on, but wish to include in the
    # calibration object so we can plot them.
    results_to_plot = ['cancer_incidence', 'asr_cancer_incidence']

    # Create the calibration object, run it, and plot the results
    calib = hpv.Calibration(
        sim,
        calib_pars=calib_pars,
        genotype_pars=genotype_pars,
        extra_sim_result_keys=results_to_plot,
        datafiles=datafiles,
        total_trials=10, n_workers=1, 
        keep_db=True, #for some reason there is a bug where if i set keep_db to its default value of false, i get a WinError 32 (the process cannot access the file because it is being used by another process) relating to line 431 of calibration.py
        name="CalibrationRawResults\\hpvsim_calubration_13June24_5_",
        rand_seed = rand_seed, #rand_seed, #Keeping random seed constant for reproducibility (random seed is used for optuna runs)
        sampler_type = "tpe",       #Accepted values are  ["random", "grid", "tpe", "cmaes", "nsgaii", "qmc", "bruteforce"]
        sampler_args = None, # dict(constant_lia r=True), # dict(multivariate=True, group=True, constant_liar=True)       # Refer to optuna documentation for relevant arguments for a given sampler type; https://optuna.readthedocs.io/en/stable/reference/samplers/index.html 
        prune = False, #when pruning, we can suppress warnings with the flag python -W ingore foo.py, when running file foo.py in command prompt.
                        #TODO: get warnings due to division by zero to go away, first by suppressing them in-code and then by not doing unnecessary calculations in the first place!
        
    )

    #Make a list which will contain all parmaeter combos, for which we wil then generate a contour plot
    ps = ['beta', 'hi5_cin_fn_k', 'hi5_dur_cin_par1','hpv16_cin_fn_k', 'hpv16_dur_cin_par1', 'hpv18_cin_fn_k', 'hpv18_dur_cin_par1', 'hpv_control_prob']
    all_parameter_pairs = []
    for i in range(len(ps)):
        for j in range(i+1, len(ps)):
            all_parameter_pairs.append((ps[i],ps[j]))
#[('beta', 'hi5_cin_fn_k'), ('hi5_dur_cin_par1','hpv16_cin_fn_k'), ('hpv16_dur_cin_par1', 'hpv18_cin_fn_k'),( 'hpv18_dur_cin_par1', 'hpv_control_prob')]

    calib.calibrate(die=False,plots=["learning_curve", "timeline"], detailed_contplot=[],save_to_csv=None) #bad idea to do a countour plot with many trials - takes a lot of memory
    calib.plot_learning_curve()
    #calib.plot_contour()
    calib.plot_timeline()
    calib.plot(res_to_plot=10) #plots the res_to_plot best results
    print(calib.df)

    #Load the best found parameters into the sim and record results from it
    best_pars = calib.trial_pars_to_sim_pars() # Returns best parameters from calibration in a format ready for sim running
    sim.update_pars(best_pars)

    sim.run()
  #  a = sim.get_analyzer()
   # a.plot()
    sim.plot()