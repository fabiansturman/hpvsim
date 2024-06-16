# Import HPVsim

import hpvsim as hpv

rand_seed = 1

# Configure a simulation with some parameters

if __name__ == "__main__":
    pars = dict(n_agents=10e4, 
                start=1980, end=2025, dt=0.25,
                location='united kingdom', 
                verbose = 0,
                #rand_seed = rand_seed,
                debut= dict(f=dict(dist='normal', par1=16.0, par2=3.1), #changing the distribution for sexual debut ages, as they are signficantly different in the uk to nigeria
                            m=dict(dist='normal', par1=16.0, par2=4.1))) #TODO: find were this data comes from

    sim = hpv.Sim(pars)

    

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
    datafiles=['fabiandata\\calib14jan23\\new_cervical_cancer_cases.csv',
               'fabiandata\\calib14jan23\\genotype_distrib_cancer.csv',
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
        keep_db=True,
        name="CalibrationRawResults\\hpvsim_calubration_ukdata_13thJune24_8",
        total_trials=1, n_workers=1,
        prune = True
    )

    
    calib.calibrate(die=False,plots=["learning_curve", "timeline"], detailed_contplot=[],save_to_csv=None)

    calib.plot_learning_curve()
    calib.plot_timeline()

    #calib.plot(res_to_plot=5);
    #calib.plot(res_to_plot=1) #single best result is plotted alongside the data