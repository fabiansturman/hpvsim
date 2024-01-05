# Import HPVsim
import hpvsim as hpv

if __name__ == "__main__":#to allow for using several workers in parellell

    # Configure a simulation with some parameters
    pars = dict(n_agents=10e3, start=1970, end=2020, dt=0.25, location='nigeria')
    sim = hpv.Sim(pars)

    # Specify some parameters to adjust during calibration.
    # The parameters in the calib_pars dictionary don't vary by genotype,
    # whereas those in the genotype_pars dictionary do. Both kinds are
    # given in the order [best, lower_bound, upper_bound].
    calib_pars = dict(
            beta=[0.05, 0.010, 0.20],hpv_control_prob=[.9, 0.5, 1]
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
    datafiles=['docs\\tutorials\\nigeria_cancer_cases.csv',
            'docs\\tutorials\\nigeria_cancer_types.csv']

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
        total_trials=7000, n_workers=12, 
        keep_db=True, #for some reason there is a bug where if i set keep_db to its default value of false, i get a WinError 32 (the process cannot access the file because it is being used by another process) relating to line 431 of calibration.py
        name="hpvsim_calubration_7000iters" #to keep track of my calibrations
    )
    calib.calibrate(die=True)
    calib.plot(res_to_plot=4);