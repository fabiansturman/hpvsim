# Import HPVsim
import hpvsim as hpv
import optuna as op
import sciris as sc
import numpy as np


if __name__ == "__main__":
    # Create the age analyzers.
    az = hpv.age_results(
        result_args=sc.objdict(
            # The keys of this dictionary are any results you want by age, and can be any key of sim.results. We are analyzing these 4 values by ages in 2000, 2010, 2015,
            hpv_prevalence=sc.objdict( 
                years=[2000,2020],
                edges=np.array([0., 15., 20., 25., 30., 35. ,40., 45., 50., 55., 65.,70.,75.,80.,85., 100.]), #define the extremities of the age buckets for each analyser
            ),
            cancer_incidence=sc.objdict(
                years=[2000,2020],
                edges=np.array([0., 15., 20., 25., 30., 35. ,40., 45., 50., 55., 65.,70.,75.,80.,85., 100.]), 
            ),
        )
    )


    # Configure simulation
    pars = dict(n_agents=10,#6*10e3, 
                start=1970,
                end=2023, 
                dt=0.25, 
                location='nigeria', #united kingdom
                genotypes=[16, 18, 'hi5'],
                analyzers=[az],
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
    datafiles=[ 'docs\\tutorials\\nigeria_cancer_cases.csv',
                'docs\\tutorials\\nigeria_cancer_types.csv',]

    # List extra results that we don't have data on, but wish to include in the
    # calibration object so we can plot them.
    results_to_plot = []#'cancer_incidence', 'asr_cancer_incidence']

    # Create the calibration object, run it, and plot the results
    calib = hpv.Calibration(
        sim,
        calib_pars=calib_pars,
        genotype_pars=genotype_pars,
        extra_sim_result_keys=results_to_plot,
        verbose = False,
        datafiles=datafiles,
        keep_db=True,
        #rand_seed=rand_seed,
        name="CalibrationRawResults\\hpvsim_calibration_ukdata_26thJuly2024_5",
       # total_trials=4, n_workers=1,
        total_trials=4000, n_workers=15,
        #leakiness = 0,
        pruning = -1 #-1:no pruning ; 1:basic pruning ; 2:leaky pruning ; 3:adaptive pruning
    )

    calib.calibrate(die=False,plots=["learning_curve", "timeline"], detailed_contplot=None,save_to_csv=None) #bad idea to do a countour plot with many trials - takes a lot of memory
    calib.plot_learning_curve()
    calib.plot_timeline()
    calib.plot(res_to_plot=1) #plots the res_to_plot best results
    print(calib.df)

    #Load the best found parameters into the sim and record results from it
    best_pars = calib.trial_pars_to_sim_pars() # Returns best parameters from calibration in a format ready for sim running
    sim.update_pars(best_pars)

    sim.run()
    a = sim.get_analyzer()
    a.plot()
    sim.plot()
    sim.to_excel('nigeriasimdata_cal.xlsx')