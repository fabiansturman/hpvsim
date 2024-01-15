# Import HPVsim
import hpvsim as hpv

rand_seed = 10

if __name__ == "__main__":#to allow for using several workers in parellell

    #Define the interventions that will be presetn in the simulation, mimicking that found in the uk (kinda)
    #prob = 0.6 # prob = 60% means we vaccinate 60% of girls (or girls and boys if we add sex = [0,1])

    #vx = hpv.routine_vx(prob=prob, start_year=2015, age_range=[9,10], product='bivalent')
        #TODO: I CANT SEEM TO GET INTERVENTIONS ADDED TO MY SIM WHEN CALIBRATING, WITHOUT GETTING A VALUEERROR 'PROVIDE EITHER A LIST OF YEARS OR A START YEAR, OR BOTH'

    # Configure a simulation with some parameters
    pars = dict(n_agents=10e3, 
                start=1980,
                end=2023, 
                dt=0.25, 
                location='nigeria', #united kingdom
                genotypes=[16, 18, 'hi5'],
                verbose = 0,
                rand_seed = rand_seed)
    sim = hpv.Sim(pars)
    #print(f"Simulation's parameter keys: <{sim.pars.keys()}>")

    # Specify some parameters to adjust during calibration.
    # The parameters in the calib_pars dictionary don't vary by genotype,
    # whereas those in the genotype_pars dictionary do. Both kinds are
    # given in the order [best, lower_bound, upper_bound].
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
                'docs\\tutorials\\nigeria_cancer_types.csv'
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
        total_trials=1, n_workers=1, 
        keep_db=True, #for some reason there is a bug where if i set keep_db to its default value of false, i get a WinError 32 (the process cannot access the file because it is being used by another process) relating to line 431 of calibration.py
        name="CalibrationRawResults\\hpvsim_calubration_15jan24_29",
        rand_seed = rand_seed #Keeping random seed constant for reproducibility (random seed is used for optuna runs)
    )
    calib.calibrate(die=True)
    calib.plot_learning_curve()
    calib.plot(res_to_plot=10) #plots the 10 (best??) results