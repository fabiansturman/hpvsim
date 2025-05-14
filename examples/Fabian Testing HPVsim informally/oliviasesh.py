# Import HPVsim
import hpvsim as hpv


import sciris
print(sciris.__version__)


# Configure a simulation with some parameters
pars = dict(n_agents=10e3, start=1980, end=2020, dt=0.25, location='nigeria')
sim = hpv.Sim(pars)

# Specify some parameters to adjust during calibration.
# The parameters in the calib_pars dictionary don't vary by genotype,
# whereas those in the genotype_pars dictionary do. Both kinds are
# given in the order [best, lower_bound, upper_bound].
calib_pars = dict(
        beta=[0.05, 0.010, 0.20],
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
    keep_db=True,
    verbose = True,
    name="CalibrationRawResults\\hpvsim_calubration_13June24_14",
    rand_seed=0,
    total_trials=2, n_workers=1,
    #  leakiness = 1,
    pruning = -1 , #-1:no pruning ; 1:basic pruning ; 2:leaky pruning ; 3:adaptive pruning,
    pruner = None# optuna.pruners.HyperbandPruner(min_resource=1, max_resource=n_trials, reduction_factor=3)
)

calib.calibrate(die=True,plots=["learning_curve", "timeline"], detailed_contplot=[],save_to_csv=None)
calib.plot(res_to_plot=4);



