# Import HPVsim
import hpvsim as hpv
import sciris as sc
import numpy as np

import json


if __name__ == "__main__":
    # Create the age analyzers, from which to read the synthetic data
    az1 = hpv.age_results(
        result_args=sc.objdict(
            # The keys of this dictionary are any results you want by age, and can be any key of sim.results. We are analyzing these 4 values by ages in 2000, 2010, 2015,
            hpv_prevalence=sc.objdict( 
                years=[2000,2010,2020,2023],
                edges=np.array([0., 15., 20., 25., 30., 35. ,40., 45., 50., 55., 65.,70.,75.,80.,85., 100.]), #define the extremities of the age buckets for each analyser
            )
        )
    )
    az2 = hpv.age_results(
        result_args=sc.objdict(
            # The keys of this dictionary are any results you want by age, and can be any key of sim.results. We are analyzing these 4 values by ages in 2000, 2010, 2015,
            cancer_incidence=sc.objdict(
                years=[2000,2010,2020,2023],
                edges=np.array([0., 15., 20., 25., 30., 35. ,40., 45., 50., 55., 65.,70.,75.,80.,85., 100.]), 
            )
        )
    )
    az3 = hpv.age_results(
        result_args=sc.objdict(
            # The keys of this dictionary are any results you want by age, and can be any key of sim.results. We are analyzing these 4 values by ages in 2000, 2010, 2015,
            cancers=sc.objdict(
                years=[2000,2010,2020,2023],
                edges=np.array([0., 15., 20., 25., 30., 35. ,40., 45., 50., 55., 65.,70.,75.,80.,85., 100.]), 
            )
        )
    )


    # Configure simulation
    pars = dict(n_agents=1e6,#6*10e3,   
                start=1970,
                end=2023, 
                dt=0.25, 
                location='nigeria', #united kingdom
                genotypes=[16, 18, 'hi5'],
                analyzers=[az1,az2,az3],
                verbose = False
                )
    sim = hpv.Sim(pars)

    #Pick parameters to be calibrated
    calib_pars = dict(
            beta=[0.25, 0.010, 0.7],
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

    # List the datafiles that contain data that we wish to compare the model to:
    datafiles=[ 'docs\\tutorials\\nigeria_cancer_cases.csv', #this is data from 2020
                'docs\\tutorials\\nigeria_cancer_types.csv', #this is data from 2000
                ] 

    # List extra results that we don't have data on, but wish to include in the
    # calibration object so we can plot them. 
    results_to_plot = []#'cancer_incidence', 'asr_cancer_incidence']

    # Create the calibration object, run it, and plot the results
    calib = hpv.Calibration(
        sim,
        calib_pars=calib_pars, 
        genotype_pars=genotype_pars,
        extra_sim_result_keys=results_to_plot,
        verbose = True,   #set to False to get no output in the calibration
        datafiles=datafiles,
        keep_db=True,
        #rand_seed=rand_seed,
        name="CalibrationRawResults\\hpvsim_calibration_ukdata_5thAug2024_36",
        #total_trials=5, n_workers=1,
        total_trials=1500, n_workers=8,
        #leakiness = 0,
        pruning = -1 #-1:no pruning ; 1:basic pruning ; 2:leaky pruning ; 3:adaptive pruning
    )

    calib.calibrate(die=False,plots=["learning_curve", "timeline"], detailed_contplot=None,save_to_csv=None) #bad idea to do a countour plot with many trials - takes a lot of memory
    calib.plot_learning_curve()
    calib.plot_timeline()
    calib.plot(res_to_plot=5) #plots the res_to_plot best results
    print(calib.df)


    
    #Load the best found parameters into the sim and record results from it
    best_pars = calib.trial_pars_to_sim_pars() # Returns best parameters from calibration in a format ready for sim running
    sim.update_pars(best_pars)

    sim.run() #note that this sim will run until the end of 2023, unlike the sims in our calibration which we change to terminate early at the end of 2020 as we have no data beyond that point (we only change a deep copy of the sim, though, so it - rightly - does not affect the original choice of end year)
    sim.plot()
    sim.to_excel('nigeriasimdata_cal.xlsx')



    i = 0
    for a in sim.analyzers:
        i+=1
        a.plot()
        a_json = a.to_json()
        a_json_formatted = json.dumps(a_json, indent="\t")
        with open(f"analyzer{i}Data.json", "w") as outfile:
            outfile.write(a_json_formatted)