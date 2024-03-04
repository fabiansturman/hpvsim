'''
Define the calibration class
'''
"""
import os
import multiprocessing as mp
import numpy as np
import pylab as pl
import pandas as pd
import sciris as sc
from . import misc as hpm
from . import plotting as hppl
from . import analysis as hpa
from . import parameters as hppar
from .settings import options as hpo # For setting global options


__all__ = ['Calibration']



def import_optuna():
    ''' A helper function to import Optuna, which is an optional dependency '''
    try:
        import optuna as op # Import here since it's slow
    except ModuleNotFoundError as E: # pragma: no cover
        errormsg = f'Optuna import failed ({str(E)}), please install first (pip install optuna)'
        raise ModuleNotFoundError(errormsg)
    return op

class UncalibratedException(Exception):
    '''
    A type of exception which can be thrown when we try to do things with an uncalibrated calibration object which can only be done with a calibrated one 
    '''
    pass

class Calibration(sc.prettyobj):
    '''
    A class to handle calibration of HPVsim simulations. Uses the Optuna hyperparameter
    optimization library (optuna.org), which must be installed separately (via
    pip install optuna).

    Note: running a calibration does not guarantee a good fit! You must ensure that
    you run for a sufficient number of iterations, have enough free parameters, and
    that the parameters have wide enough bounds. Please see the tutorial on calibration
    for more information.

    Args:
        sim          (Sim)      : the simulation to calibrate
        datafiles    (list)     : list of datafile strings to calibrate to
        calib_pars   (dict)     : a dictionary of the parameters to calibrate of the format dict(key1=[best, low, high])
        genotype_pars(dict)     : a dictionary of the genotype-specific parameters to calibrate of the format dict(genotype=dict(key1=[best, low, high]))
        hiv_pars     (dict)     : a dictionary of the hiv-specific parameters to calibrate of the format dict(key1=[best, low, high])
        extra_sim_results (list): list of result strings to store
        fit_args     (dict)     : a dictionary of options that are passed to sim.compute_fit() to calculate the goodness-of-fit
        par_samplers (dict)     : an optional mapping from parameters to the Optuna sampler to use for choosing new points for each; by default, suggest_float
        n_trials     (int)      : the number of trials per worker
        n_workers    (int)      : the number of parallel workers (default: maximum
        total_trials (int)      : if n_trials is not supplied, calculate by dividing this number by n_workers)
        name         (str)      : the name of the database (default: 'hpvsim_calibration')
        db_name      (str)      : the name of the database file (default: 'hpvsim_calibration.db')
        keep_db      (bool)     : whether to keep the database after calibration (default: false)
        storage      (str)      : the location of the database (default: sqlite)
        rand_seed    (int)      : if provided, use this random seed to initialize Optuna runs (for reproducibility)
        sampler_type (str)      : choice of Optuna sampler type. Default value is "tpe". Options: ["random", "grid", "tpe", "cmaes", "nsgaii", "qmc", "bruteforce"]
        sampler_args (dict)     : argument-value pairs passed to the sampler's constructor. Refer to optuna documentation for relevant arguments for a given sampler type; https://optuna.readthedocs.io/en/stable/reference/samplers/index.html 
        prune        (bool)     : whether or not to do pruning
        label        (str)      : a label for this calibration object
        die          (bool)     : whether to stop if an exception is encountered (default: false)
        verbose      (bool)     : whether to print details of the calibration
        kwargs       (dict)     : passed to hpv.Calibration()

    Returns:
        A Calibration object

    **Example**::

        sim = hpv.Sim(pars, genotypes=[16, 18])
        calib_pars = dict(beta=[0.05, 0.010, 0.20],hpv_control_prob=[.9, 0.5, 1])
        calib = hpv.Calibration(sim, calib_pars=calib_pars,
                                datafiles=['test_data/south_africa_hpv_data.xlsx',
                                           'test_data/south_africa_cancer_data.xlsx'],
                                total_trials=10, n_workers=4)
        calib.calibrate()
        calib.plot()

    '''

    def __init__(self, sim, datafiles, calib_pars=None, genotype_pars=None, hiv_pars=None, fit_args=None, extra_sim_result_keys=None, 
                 par_samplers=None, n_trials=None, n_workers=None, total_trials=None, name=None, db_name=None, estimator=None,
                 keep_db=None, storage=None, 
                 rand_seed=None, sampler_type=None, sampler_args=None, prune=True,
                 label=None, die=False, verbose=True):

        import multiprocessing as mp # Import here since it's also slow

        # Handle run arguments
        if n_trials  is None: n_trials  = 20
        if n_workers is None: n_workers = mp.cpu_count()
        if name      is None: name      = 'hpvsim_calibration'
        if db_name   is None: db_name   = f'{name}.db'
        if keep_db   is None: keep_db   = False
        if storage   is None: storage   = f'sqlite:///{db_name}'
        if total_trials is not None: n_trials = int(np.ceil(total_trials/n_workers))


        if sampler_type is not None:
            if sampler_type not in ["random", "grid", "tpe", "cmaes", "nsgaii", "qmc", "bruteforce"]:
                raise Exception('Sampler type is not an accepted value. Accepted values are ["random", "grid", "tpe", "cmaes", "nsgaii", "qmc", "bruteforce"].')
        else:
            sampler_type = "tpe" #this is consistent with the default sampler type for Optuna, which is TPE
        if sampler_args is None:
            sampler_args = dict()
        
        self.run_args   = sc.objdict(n_trials=int(n_trials), n_workers=int(n_workers), name=name, db_name=db_name,
                                     keep_db=keep_db, storage=storage, 
                                     rand_seed=rand_seed, sampler_type=sampler_type, sampler_args=sampler_args, prune=prune)

        # Handle other inputs
        self.label          = label
        self.sim            = sim
        self.calib_pars     = calib_pars
        self.genotype_pars  = genotype_pars
        self.hiv_pars       = hiv_pars
        self.extra_sim_result_keys = extra_sim_result_keys
        self.fit_args       = sc.mergedicts(fit_args)
        self.par_samplers   = sc.mergedicts(par_samplers)
        self.die            = die
        self.verbose        = verbose
        self.calibrated     = False
        self.learning_curve = None                     #Assigned a value once self.calibrated = True

        # Create age_results intervention
        self.target_data = []
        for datafile in datafiles:
            self.target_data.append(hpm.load_data(datafile))

        sim_results = sc.objdict()      #instantiates an empty instance of sciris.objdict, which is a module in Python that provides a dictionary-like object with attribute-style access (it is a subclass of the build-in dict class).
        age_result_args = sc.objdict()  #[is age_result_args data then combined into sim_results somehow???]  sim_results contains data to fit to which is not split by age, while age_result_args contains data to fit to which is split by age 

        # *Go through each of the target keys and determine how we are going to get the results from sim*
        for targ in self.target_data: #iterates through each DataFrame in the self.targetdata list, i.e. targ is effectively one of the parameters we want to fit
            targ_keys = targ.name.unique() #targ.name gets each row's value in the *name* column of our dataframe, indexed by row number. Then targ.name.unique() gets us a list of all the distinct values in the *name* column; this needs to only have 1 value so that each dataframe fits to only 1 kind of targetdata
            if len(targ_keys) > 1:
                errormsg = f'Only support one set of targets per datafile, {len(targ_keys)} provided'
                raise ValueError(errormsg)
            if 'age' in targ.columns: #this checks whether the results presented in our datafile are also seperated by age
                age_result_args[targ_keys[0]] = sc.objdict(  #At this point in code execution, we know targ_keys is a singleton list. e.g. if targ_keys = ['cancers'], then we are setting age_result_args['cancers'] to be a dictionary containing...
                    datafile=sc.dcp(targ), #...a deep copy of our targetdata dataframe
                    compute_fit=True, #we are computing a fit for these age results (is this because age_result_args is a dictionary with all possible target keys as keys, but of course we are only fitting those for which we have data??)
                )
            else: #in this case, the data in targ is not split by age, so we add it to sim_results
                sim_results[targ_keys[0]] = sc.objdict(
                    data=sc.dcp(targ)
                )

        ar = hpa.age_results(result_args=age_result_args)
        self.sim['analyzers'] += [ar]
        if hiv_pars is not None:
            self.sim['model_hiv'] = True # if calibrating HIV parameters, make sure model is running HIV
        self.sim.initialize()
        for rkey in sim_results.keys():
            sim_results[rkey].timepoints = sim.get_t(sim_results[rkey].data.year.unique()[0], return_date_format='str')[0]//sim.resfreq 
            #to change to be a list of ALL the unique timepoints relevant for that datafile, i need to turn unique()[0] to just unique()
            if 'weights' not in sim_results[rkey].data.columns:
                sim_results[rkey].weights = np.ones(len(sim_results[rkey].data))
        self.age_results_keys = age_result_args.keys()
        self.sim_results = sim_results
        self.sim_results_keys = sim_results.keys()

        self.result_args = sc.objdict()
        for rkey in self.age_results_keys + self.sim_results_keys:
            self.result_args[rkey] = sc.objdict()
            if 'hiv' in rkey:
                self.result_args[rkey].name = self.sim.hivsim.results[rkey].name
                self.result_args[rkey].color = self.sim.hivsim.results[rkey].color
            else:
                self.result_args[rkey].name = self.sim.results[rkey].name
                self.result_args[rkey].color = self.sim.results[rkey].color

        if self.extra_sim_result_keys:
            for rkey in self.extra_sim_result_keys:
                self.result_args[rkey] = sc.objdict()
                self.result_args[rkey].name = self.sim.results[rkey].name
                self.result_args[rkey].color = self.sim.results[rkey].color
        # Temporarily store a filename
        self.tmp_filename = 'tmp_calibration_%05i.obj'

        
        #Populate a list of all the years for which we have data that is split by age, and then all the years for which we have data that is not split by age
        self.age_result_years = []  #all the years for which we have data which is split up by age
        self.sim_result_years = []  #all the years for which we have data which is not split up by age
        self.years            = []  #all the years for which we have data, either split by age or not
        for targ in self.target_data:
            targ_years = list(targ.year.unique())
            if 'age' in targ.columns:                                       #this checks whether the results presented in our datafile are also seperated by age
                self.age_result_years += targ_years
                self.age_result_years = list(set(self.age_result_years))
            else: 
                self.sim_result_years += targ_years
                self.sim_result_years = list(set(self.sim_result_years))


        self.years = list(set(self.age_result_years + self.sim_result_years))

        #Set the self.sim_end_year instance variable to stop wasteful tail-running of our sim
        if sim['end'] > max(self.years):                #If our simulation is going on for too long, set self.sim_end_year s.t. when we run our sim, we stop earlier...
            self.sim_end_year = max(self.years)
        else:
            self.sim_end_year = sim['end']
            #... and the alternate case (sim['end'] < max(self.years) is not possible, as index out of bounds error will be thrown when validating our variables; see analysis.py's validate_variables function)

        return


    def run_sim(self, trial, calib_pars=None, genotype_pars=None, hiv_pars=None, label=None, return_sim=False, prune=None):
        ''' Create and run a simulation '''
        #If prune=None, we use the default prune value (whatever was set before)
        if prune is None:
            prune = self.run_args.prune

        op = import_optuna()

        sim = sc.dcp(self.sim) #Creates a deep copy of the provided simulation object, this is what we will be dealing with
        
        sim['end'] = self.sim_end_year
        if label: sim.label = label

        new_pars = self.get_full_pars(sim=sim, calib_pars=calib_pars, genotype_pars=genotype_pars, hiv_pars=hiv_pars) #Get values for our adjustable parameters from optuna ...
        sim.update_pars(new_pars)        #... and then update our sim with our new values for the adjustable parameters
        sim.initialize(reset=True, init_analyzers=False) # Necessary to reinitialize the sim here so that the initial infections get the right parameters

        
                

        
        # Run the sim
        try:
            if prune:
                #define the callback function which will be called each year by our sim, to determine whether it should be pruned
                def callback(current_year):
                                #          if current_year in self.years: #For testing; printing out the itnermediate goodness-of-fit values that have been computed
                                #             print()
                                    #            print(f"Results for year {current_year}")
                                    #           print(f"    Age results gof: {self.get_cumulative_age_results_gof(sim,current_year)}")
                                    #          print(f"    Sim results gof: {self.get_cumulative_sim_results_gof(sim,current_year)}")
                                    #         print(f"    Total gof: {self.get_cumulative_gof(sim,current_year)}")

                    if current_year in self.years:
                        #We only are able to consider pruning if we can compare our model's output to at least part of our dataset 
                        intermediate_gof = self.get_cumulative_gof(sim,current_year)
                        trial.report(intermediate_gof, current_year)

                        if trial.should_prune():
                            raise op.TrialPruned()
                        
                sim.run(callback=callback, callback_annual = True)
            else:
                sim.run()
            if return_sim:
                return sim
            else:
                return sim.fit

        except Exception as E:
            if isinstance(E, op.TrialPruned): #catch pruning and throw it on
                raise op.TrialPruned()
            if self.die:
                raise E
            else:
                warnmsg = f'Encountered error running sim!\nParameters:\n{new_pars}\nTraceback:\n{sc.traceback()}'
                hpm.warn(warnmsg)
                output = None if return_sim else np.inf
                return output
            
    @staticmethod
    def update_dict_pars(name_pars, value_pars):
        ''' Function to update parameters from nested dict to nested dict's value '''
        new_pars = sc.dcp(name_pars)
        target_pars_flatten = sc.flattendict(value_pars)
        for key, val in target_pars_flatten.items():
            try: 
                sc.setnested(new_pars, list(key), val)
            except Exception as e:
                errormsg = f"Parameter {'_'.join(key)} is not part of the sim, nor is a custom function specified to use them"
                raise ValueError(errormsg)
        return new_pars
    
    def update_dict_pars_from_trial(self, name_pars, value_pars):
        ''' Function to update parameters from nested dict to trial parameter's value '''
        # new_pars = sc.dcp(name_pars)
        new_pars = {}
        name_pars_keys = sc.flattendict(name_pars).keys()
        for key in name_pars_keys:
            name = '_'.join(key)
            sc.setnested(new_pars, list(key), value_pars[name])
        return new_pars
    
    def update_dict_pars_init_and_bounds(self, initial_pars, par_bounds, target_pars):
        ''' Function to update initial parameters and parameter bounds from a trial pars dict'''
        target_pars_keys = sc.flattendict(target_pars)
        for key, val in target_pars_keys.items():
            name = '_'.join(key)
            initial_pars[name] = val[0]
            par_bounds[name] = np.array([val[1], val[2]])
        return initial_pars, par_bounds
       
    
    def get_full_pars(self, sim=None, calib_pars=None, genotype_pars=None, hiv_pars=None):
        ''' Make a full pardict from the subset of regular sim parameters, genotype parameters, and hiv parameters used in calibration'''
        # Prepare the parameters
        new_pars = {}     

        if genotype_pars is not None:
            new_pars['genotype_pars'] = self.update_dict_pars(sim['genotype_pars'], genotype_pars)

        if hiv_pars is not None:
            new_pars['hiv_pars'] = self.update_dict_pars(sim.hivsim.pars['hiv_pars'], hiv_pars)

        if calib_pars is not None:
            calib_pars_flatten = sc.flattendict(calib_pars)
            for key, val in calib_pars_flatten.items():
                if key[0] in sim.pars and key[0] not in new_pars:
                    new_pars[key[0]] = sc.dcp(sim.pars[key[0]])
                try:
                    sc.setnested(new_pars, list(key), val) # only update on keys that have values in sim.pars. If this line makes error, raise error errormsg 
                except Exception as e:
                    errormsg = f"Parameter {'_'.join(key)} is not part of the sim, nor is a custom function specified to use them"
                    raise ValueError(errormsg)       
        return new_pars

    def trial_pars_to_sim_pars(self, trial_pars=None, which_pars=None, return_full=True):
        '''
        Create genotype_pars and pars dicts from the trial parameters.
        Note: not used during self.calibrate.
        Args:
            trial_pars (dict): dictionary of parameters from a single trial. If not provided, best parameters will be used
            return_full (bool): whether to return a unified par dict ready for use in a sim, or the sim pars and genotype pars separately

        **Example**::
        
            sim = hpv.Sim(genotypes=[16, 18])
            calib_pars = dict(beta=[0.05, 0.010, 0.20],hpv_control_prob=[.9, 0.5, 1])
            genotype_pars = dict(hpv16=dict(prog_time=[3, 3, 10]))
            calib = hpv.Calibration(sim, calib_pars=calib_pars, genotype_pars=genotype_pars
                                datafiles=['test_data/south_africa_hpv_data.xlsx',
                                           'test_data/south_africa_cancer_data.xlsx'],
                                total_trials=10, n_workers=4)
            calib.calibrate()
            new_pars = calib.trial_pars_to_sim_pars() # Returns best parameters from calibration in a format ready for sim running
            sim.update_pars(new_pars)
            sim.run()
        '''

        # Initialize
        calib_pars = sc.objdict()
        genotype_pars = sc.objdict()
        hiv_pars = sc.objdict()

        # Deal with trial parameters
        if trial_pars is None:
            try:
                if which_pars is None or which_pars==0:
                    trial_pars = self.best_pars
                else:
                    ddict = self.df.to_dict(orient='records')[which_pars]
                    trial_pars = {k:v for k,v in ddict.items() if k not in ['index','mismatch']}
            except:
                errormsg = 'No trial parameters provided.'
                raise ValueError(errormsg)

        # Handle genotype parameters
        if self.genotype_pars is not None:
            genotype_pars = self.update_dict_pars_from_trial(self.genotype_pars, trial_pars)

        # Handle hiv sim parameters
        if self.hiv_pars is not None:
            hiv_pars = self.update_dict_pars_from_trial(self.hiv_pars, trial_pars)

        # Handle regular sim parameters
        if self.calib_pars is not None:
            calib_pars = self.update_dict_pars_from_trial(self.calib_pars, trial_pars)

        # Return
        if return_full:
            all_pars = self.get_full_pars(sim=self.sim, calib_pars=calib_pars, genotype_pars=genotype_pars, hiv_pars=hiv_pars)
            return all_pars
        else:
            return calib_pars, genotype_pars, hiv_pars

    def sim_to_sample_pars(self):
        ''' Convert sim pars to sample pars '''

        initial_pars = sc.objdict()
        par_bounds = sc.objdict()

        # Convert regular sim pars
        if self.calib_pars is not None:
            initial_pars, par_bounds = self.update_dict_pars_init_and_bounds(initial_pars, par_bounds, self.calib_pars)

        # Convert genotype pars
        if self.genotype_pars is not None:
            initial_pars, par_bounds = self.update_dict_pars_init_and_bounds(initial_pars, par_bounds, self.genotype_pars)

       # Convert hiv pars
        if self.hiv_pars is not None:
            initial_pars, par_bounds = self.update_dict_pars_init_and_bounds(initial_pars, par_bounds, self.hiv_pars)

        return initial_pars, par_bounds

    def trial_to_sim_pars(self, pardict=None, trial=None):
        '''
        Take in an optuna trial and sample from pars, after extracting them from the structure they're provided in
        '''
        pars = sc.dcp(pardict)
        pars_flatten = sc.flattendict(pardict)
        for key, val in pars_flatten.items():
            sampler_key = '_'.join(key)
            low, high = val[1], val[2]
            step = val[3] if len(val) > 3 else None

            if key in self.par_samplers:  # If a custom sampler is used, get it now (Not working properly for now)
                try:
                    sampler_fn = getattr(trial, self.par_samplers[key])
                except Exception as E:
                    errormsg = 'The requested sampler function is not found: ensure it is a valid attribute of an Optuna Trial object'
                    raise AttributeError(errormsg) from E
            else:
                sampler_fn = trial.suggest_float
            value = sampler_fn(sampler_key, low, high, step=step)
            #print(f"Optuna has sampled us the value {value} for the key {sampler_key}")
            sc.setnested(pars, list(key), value)
        return pars

    def get_cumulative_sim_results_gof(self, sim, year_up_to):
        '''
        Computes and returns the goodness of fit of our sim against any sim_results for which we have data (i.e. any data which is not split by age), up to a given year
        '''
        #TODO: as in the existing non-cumulative version of this process, this is currently for a single timepoint per rkey and should be generalised
        fit = 0

        results = sim.compute_intermediate_states()

        for rkey in self.sim_results:
            timepoint = self.sim_results[rkey].timepoints[0]
            year = timepoint * (sim.resfreq * sim["dt"]) + sim["start"] #sim.resfreq * sim["dt"] = time in years between each timepoint

            if year < year_up_to + 1: #e.g. if the timepoint is such that our year is 1998.8,then we want to consider this data if year_up_to==1998, as we are stopping beyond 1998 
                if sim.results[rkey][:].ndim==1:
                    model_output = results[rkey][timepoint] 
                else:
                    model_output = results[rkey][:,timepoint] #e.g. if we have several genotypes, this gets us the rkey result at index timepoint[0] for each genotype in a list 

                gofs = hpm.compute_gof(self.sim_results[rkey].data.value, model_output) #computes the goodness of fit of the model, using the highly customizable compute_gof function in misc.py 
                losses = gofs * self.sim_results[rkey].weights   #each row in an Dataframe containing data that we want to fit to is by default weighted 1, but if we want to weight more heavily for certain quantities (which would mean we care more about fitting those quantities than other, lesser-weighted ones), we can add a 'weight' column to our csv files and just weight our rows as we desire
                mismatch = losses.sum()
                fit += mismatch   #this first declares, and then assigns values to, an instance variable sim.fit, which accumulates the objective function value for this trial
        
        return fit
    

    def get_cumulative_age_results_gof(self, sim, year_up_to):
        '''
        Computes and returns the goodness of fit of our sim against any age results for which we have data (i.e. any data which is split by age), up to a given year
        '''
        fit = 0
        for analyzer in sim.analyzers:
            if isinstance(analyzer, hpa.Analyzer):
                fit += analyzer.cumulative_fit(self,year_up_to)

        return fit

    def get_cumulative_gof(self, sim, year_up_to):
        '''
        Computes and returns the goodness of fit of the provided sim against all data we have avaliable, up to the given year
        '''
        fit = 0  #TODO: maybe use np.inf instead? optuna documentation says it supports all float-like types for intermediate value reporting and specifies numpy.flaot32 as an example, but does it support this particular constnat of the type??

        if year_up_to >= min(self.age_result_years):            #If we have at least some age result data at or before the given end year, then compute its gof
            fit += self.get_cumulative_age_results_gof(sim, year_up_to)
        if year_up_to >= min(self.sim_result_years):            ##If we have at least some sim result data at or before the given end year, then compute its gof
            fit += self.get_cumulative_sim_results_gof(sim, year_up_to)

        return fit


    def run_trial(self, trial, save=True):
        ''' Define the objective for Optuna ''' #TODO: (fabian) make a function that checks the fit of a model at a given timepoint, and then create a function which computes the fit *up to* a given timepoint instead, and then use this to implement reporting of intermediate values, and early stopping, within our sim. This would require a way to do 'callback functions' within the running of a sim, called back at every nth time jump (or maybe every time jump? but probably sticking to every nth timejumpo helps speed it up as calcualtion of the cumulative goodness of fit won't be the easiest thing in the world!)
        #(Using Optuna), sample values for every parameter which we are letting vary in this calibration.
        if self.genotype_pars is not None:
            genotype_pars = self.trial_to_sim_pars(self.genotype_pars, trial)
        else:
            genotype_pars = None
        if self.hiv_pars is not None:
            hiv_pars = self.trial_to_sim_pars(self.hiv_pars, trial)
        else:
            hiv_pars = None
        if self.calib_pars is not None:
            calib_pars = self.trial_to_sim_pars(self.calib_pars, trial)
        else:
            calib_pars = None

        #genotype_pars (resp. hiv_pars, calib_pars) are dictionaries in the same structure as they were when passed into the calibration object's constructor, ...
        #... but instead of defining ranges in the form [best, lower, upper] for each parameter, the dictionary 'value' of each parameter 'key' is the single ...
        #...parameter value we will be using in this trial


        sim = self.run_sim(trial, calib_pars, genotype_pars, hiv_pars, return_sim=True) 

        #Check whether the simulation has been pruned (terminated early as it is unpromising); if it has done then run_sim will have returned None instead of a sim object
        if sim is None:
            op = import_optuna()
            raise op.TrialPruned()


        # Compute fit for sim results and save sim results (TODO: THIS IS FOR A SINGLE TIMEPOINT. GENERALIZE THIS)
        sim_results = sc.objdict()
#        print(f"self.sim_results: {self.sim_results}") ## right now, this is pretty much just printing nigeria_cancer_types.csv

        
    #    print(f"fit = {sim.fit}") #at this point, there is alreay a non-zero value in sim.fit; i presume from nigeria_cancer_cases.csv?

        for rkey in self.sim_results:
#            print(f"rkey: {rkey}")


 #           print(f"{rkey} : {sim.results[rkey]}")
        #    print(f"self.sim_results[rkey].timepoints : {self.sim_results[rkey].timepoints}") 
            #timepoints appears to be the nummber of years since the start of the sim, when our data refers. 
            #for example for a sim starting at 1980 and all data fro a given rkey in 2015, timepoints = [35]. 
            #But if some data in the rkey is from 2017 too, timepoints=[35] still, not [35,37]. though if i start from 1982 instead of 1980, timepoints=[33]
            #so... is it that timepoints is the *first* time data for a given rkey is provided
            #THEN IF DATA IS PROVIDED AT TWO DIFFERENT TIMEPOINTS FOR THE SAME RKEY, IS THAT THE SITUATI9ON IN WHICH WE ONLY COMPUTE FOR THE FIRST TIMEPOINT?
            
            #So: it appears that (at least for a sim which is set up to record a full collection of its results every year), for a given rkey (e.g.'cancerous_genotypes_dist') self.sim_results[rkey].timepoints gives us a list of the timepoints in our sim's results dictionary that matter for us for our rkey, i.e. the points in our rkey's time-indexed result list which are relevent for us when computing gof.
            #however, it appears that our timepoints list only includes the year of the first line in our csv file (e..g if the lines go 2015, 2012, 2012 then it is 35, but if they go 2012, 2015, 2015, they are 32, both with simulations starting at 1980) 

            #As it stands, timepoints[0] gives us the index in the reveant rkeys result array of the real-valued result we want to compare with a corresponding data-file result 


            #Timepoints are in years, as opposed to dt, because when they are created in the constructor, their values are divided by resfreq

            #rkey is something like 'cancerous_genotypes_dist' ; it is the result type that we want to extract (for a given year) from our model, with which to compute the gof

            if sim.results[rkey][:].ndim==1:
                model_output = sim.results[rkey][self.sim_results[rkey].timepoints[0]] 
            else:
                model_output = sim.results[rkey][:,self.sim_results[rkey].timepoints[0]] #e.g. if we have several genotypes, this gets us the rkey result at index timepoint[0] for each genotype in a list 

   #         print(f"model_output: {model_output}")

    #        print(f"self.sim_results[rkey].data: {self.sim_results[rkey].data}")
            
            #self.sim_results[rkey].data is a Dataframe with columns named 'year','name', 'value', etc... we care about getting the 'value' column, to then compare with the values we have got in our model results
            #The function hpm.compute_gof acts as a metric, giving us a 'distance' between the two parameter vectors
            #By default, hpm.compute_gof(y,y_pred) gives us "normalised absolute error", i.e. |y - y_pred|/max(y) ;i.e. we normalise by the maxmimum value of this rkey at this timepoint in our data, so that we don't end up beign biased towards larger absolute values in our dataset (e.g. total HPV infections will be typically signfiicantly larger than total HPV deaths, so instead what this will do is ensure we try and get the 'percentage error' minimised on both, as a percentage of the data we are fitting to, istead of the value itself)
           ###### diffs = self.sim_results[rkey].data.value - model_output - we dont need this line of code, it is covered by hpm.compute_gof
            gofs = hpm.compute_gof(self.sim_results[rkey].data.value, model_output) #computes the goodness of fit of the model, using the highly customizable compute_gof function in misc.py 
            losses = gofs * self.sim_results[rkey].weights   #each row in an Dataframe containing data that we want to fit to is by default weighted 1, but if we want to weight more heavily for certain quantities (which would mean we care more about fitting those quantities than other, lesser-weighted ones), we can add a 'weight' column to our csv files and just weight our rows as we desire
            mismatch = losses.sum()
     #####       sim.fit += mismatch   #this first declares, and then assigns values to, an instance variable sim.fit, which accumulates the objective function value for this trial
            sim_results[rkey] = model_output


        #I have verified that if we run a trial with both nigeria csv files as our dataset, vs with just one of them, we DO get very different gofs, even though (because we have used the same random seed) the parameter values in our model are the same. 
            #so SURELY both datafiles have an effect on our GOF; badly fitting to one of them will change our GOF and ultimatley the result of our calibration
            #SO WHERE IS nigeria_cancer_cases.csv being used!? where is it contributing to our gof???
        '''
        for rkey in self.sim_results:
            for i in len(self.sim_results[rkey].timepoints):
                if sim.results[rkey][:].ndim==1:
                    model_output = sim.results[rkey][self.sim_results[rkey].timepoints[timepoint]]
                else:
                    model_output = sim.results[rkey][:,self.sim_results[rkey].timepoints[timepoint]] #e.g. if we have several genotypes, this gets us the rkey result at index timepoint[0] for each genotype in a list 
            
                #We will be comparing our model output to just the data for the relevant year
                relevant_data = sc.dcp(self.sim_results[rkey].data)
                relevant_data = relevant_data[relevant_data.year == timepoint * sim.resfreq + sim["start"]]

                weights = sc.dcp(self.sim_results[rkey].weights)
                rows_to_delete = self.sim_results[rkey].data.index[self.sim_results[rkey].data['year'] > timepoint*sim.resfreq + sim["start"]].tolist()
                weights = np.delete(weights, rows_to_delete)

                gofs = hpm.compute_gof(self.sim_results[rkey].data.value, model_output) #computes the goodness of fit of the model, using the highly customizable compute_gof function in misc.py 
                losses = gofs * weights   #each row in an Dataframe containing data that we want to fit to is by default weighted 1, but if we want to weight more heavily for certain quantities (which would mean we care more about fitting those quantities than other, lesser-weighted ones), we can add a 'weight' column to our csv files and just weight our rows as we desire
                mismatch = losses.sum()
                sim.fit += mismatch   #this first declares, and then assigns values to, an instance variable sim.fit, which accumulates the objective function value for this trial
                sim_results[rkey] = model_output #TODO: this overrides model output until the final timepoint, not great'''

        extra_sim_results = sc.objdict()


        if self.extra_sim_result_keys:
            for rkey in self.extra_sim_result_keys:
                model_output = sim.results[rkey]
                extra_sim_results[rkey] = model_output

  #      print(f"sim_results: {sim_results}")

        # Store results in temporary files (TODO: consider alternatives)
        if save:
            results = dict(sim=sim_results, analyzer=sim.get_analyzer('age_results').results,
                           extra_sim_results=extra_sim_results)
            filename = self.tmp_filename % trial.number
            sc.save(filename, results)

        #Report values back to optuna 
            
        return self.get_cumulative_gof(sim, np.inf)


    def make_sampler(self, seed_offset:int = 0):
        '''Makes and returns a sampler according to the information in self.run_args. 
           If seed_offset is not 0, adds it to the seed when creating the sampler'''
        op = import_optuna()

        #To use a given random seed for Optuna's parameter suggestion, add it to the constructor's parameter list
        if self.run_args.rand_seed is not None:
            self.run_args.sampler_args['seed'] = self.run_args.rand_seed + seed_offset


        #Instantiate the desired sampler, unpacking the dictionary self.sampler_args to use as constructor arguments
        match self.run_args.sampler_type:
            case "random":      sampler = op.samplers.RandomSampler(**self.run_args.sampler_args)        
            case "grid":        sampler = op.samplers.GridSampler(**self.run_args.sampler_args)
            case "tpe":         sampler = op.samplers.TPESampler(**self.run_args.sampler_args)
            case "cmaes":       sampler = op.samplers.CmaEsSampler(**self.run_args.sampler_args)
            case "nsgaii":      sampler = op.samplers.NSGAIISampler(**self.run_args.sampler_args)
            case "qmc":         sampler = op.samplers.QMCSampler(**self.run_args.sampler_args)
            case "bruteforce":  sampler = op.samplers.BruteForceSampler(**self.run_args.sampler_args)
        
        return sampler
    

    def worker(self, id:int):
        ''' Run a single worker.
            pre: 1<=id<=self.run_args.n_workers && each worker has a distinct id
        '''
        op = import_optuna()
        if self.verbose:
            op.logging.set_verbosity(op.logging.DEBUG)
        else:
            op.logging.set_verbosity(op.logging.ERROR)

        sampler = self.make_sampler(id) #Construct a sampler according to the data in self.run_args; we offset the seed by 'id' iff we are using a user-defined seed
        study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.name, sampler=sampler)

        output = study.optimize(self.run_trial, n_trials=self.run_args.n_trials, callbacks=None) #self.run_trial is our objective function (i.e. will take our study and use optuna's suggested value to get us a goodness-of-fit value which is returned). 
        
        return output


    def run_workers(self):
        ''' Run multiple workers in parallel '''
        if self.run_args.n_workers > 1: # Normal use case: run in parallel
            worker_ids = list(range(1, self.run_args.n_workers+1))
            output = sc.parallelize(self.worker, iterarg=worker_ids)
        else: # Special case: just run one
            output = [self.worker(0)]
        return output

    def run_workers_RAM(self, n_workers:int):
        '''Runs multiple workers in parellel, using only primary storage for storing optuna results '''
        op = import_optuna()
        if self.verbose:
            op.logging.set_verbosity(op.logging.DEBUG)
        else:
            op.logging.set_verbosity(op.logging.ERROR)



        def RAM_worker(task_source):
            '''
            This takes a source of task tuples (genotype_pars, hiv_pars, calib_pars, trial_number, transmitter), and runs a simulation on them.
            They catch any pruning exceptions and return -1 is pruning happens, else they return the cost of the trial. THOUGH RIGHT NOW PRUNING DOES NOT HAPPEN FOR RAM WORKERS
            If the queue is closed, or if we get None from it, that means the optimisation has completed, so kill the worker.
            '''
            save = True
            
            print("starting ram worker")
            try:
                print("about to get stuff from task queue")
                x = task_source.get(block = True, timeout = None)
                print("gotten something from task   queue")
                while x is not None:
                    genotype_pars, hiv_pars, calib_pars, trial_number, transmitter = x

                    sim = self.run_sim(None, calib_pars, genotype_pars, hiv_pars, return_sim=True, prune=False) #TODO: my architecture relies on the worker being totally independant from the optuna trial, so I am not too sure how I would get pruning to work without some way to interact with the trial in a protected environment

                    sim_results = sc.objdict()

                    for rkey in self.sim_results:
                        if sim.results[rkey][:].ndim==1:
                            model_output = sim.results[rkey][self.sim_results[rkey].timepoints[0]] 
                        else:
                            model_output = sim.results[rkey][:,self.sim_results[rkey].timepoints[0]]
                    
                    gofs = hpm.compute_gof(self.sim_results[rkey].data.value, model_output) 
                    losses = gofs * self.sim_results[rkey].weights  
                    mismatch = losses.sum()
                    sim_results[rkey] = model_output

                    extra_sim_results = sc.objdict()


                    if self.extra_sim_result_keys:
                        for rkey in self.extra_sim_result_keys:
                            model_output = sim.results[rkey]
                            extra_sim_results[rkey] = model_output

                    if save:
                        results = dict(sim=sim_results, analyzer=sim.get_analyzer('age_results').results,
                                    extra_sim_results=extra_sim_results)
                        filename = self.tmp_filename % trial_number
                        sc.save(filename, results)

                    #Report values back to relevant optuna thread-worker
                    fit= self.get_cumulative_gof(sim, np.inf)
                    transmitter.send(fit)

                    #wait for next task and repeat                    
                    x = task_source.get(block = True, timeout = None)

                return () #if our task_source is giving us None, that means we are out of tasks and can kill our worker
            except ValueError:
                if self.verbose:
                    print("Closing worker")
                return () #in this case, the queue has been closed, so we can quit our worker
        
        global task_queue
        task_queue = mp.Queue() #will hold tuples of the form (genotype_pars, hiv_pars, calib_pars, transmitter) where transmitter is the sending-end of a pipe down which to return the result
        

        def objective(trial):
            '''
            Called in its own thread by optuna.
            This function samples a set of suggested parameter values, and passes this job to the task queue, for a worker to pick up and send back to it.
            '''

            #(Using Optuna), sample values for every parameter which we are letting vary in this calibration.
            if self.genotype_pars is not None:
                genotype_pars = self.trial_to_sim_pars(self.genotype_pars, trial)
            else:
                genotype_pars = None
            if self.hiv_pars is not None:
                hiv_pars = self.trial_to_sim_pars(self.hiv_pars, trial)
            else:
                hiv_pars = None
            if self.calib_pars is not None:
                calib_pars = self.trial_to_sim_pars(self.calib_pars, trial)
            else:
                calib_pars = None
            
            reciever, transmitter = mp.Pipe(duplex=False)

            print("adding to task queue")
            task_queue.put((1))#genotype_pars, hiv_pars, calib_pars, trial.number, transmitter))

            res = reciever.recv() #blocks until there is something to recieve; either a goodness of fit or -1 to indicate a pruned trial

            if res == -1:
                raise op.TrialPruned()
            else:
                return res

        #Start off our workers
        for _ in range(n_workers):
            p = mp.Process(target=RAM_worker, args = (task_queue,))
            p.start()


        sampler = self.make_sampler() #when all data is in the RAM, we only have one optuna trial object at any one time, so only one sampler; no need to pass a uniqueness parameter in therefore
        study = op.create_study(sampler = sampler)
        study.optimize(objective,n_trials = self.run_args.n_trials, n_jobs = n_workers) #n_workers threads will be sending tasks down the task_queue but are hindered by Python's GIL. 
                                                                #...however for each of the threads here, we have a process which does work concurrently, evaluating the tasks.
        #The above code waits until optimisation is done, which itself depends on the workers completing the tasks. So it is blocking.

        #Now, we kill each RAM_worker instance by sending n_workers x None down the queue, and closing it.
        for _ in range(n_workers):
            task_queue.put(None)
        task_queue.close()

        return study



    def remove_db(self):
        '''
        Remove the database file if keep_db is false and the path exists.
        '''
        try:
            op = import_optuna()
            op.delete_study(study_name=self.run_args.name, storage=self.run_args.storage)
            if self.verbose:
                print(f'Deleted study {self.run_args.name} in {self.run_args.storage}')
        except Exception as E:
            print('Could not delete study, skipping...')
            print(str(E))
        if os.path.exists(self.run_args.db_name):
            os.remove(self.run_args.db_name)
            if self.verbose:
                print(f'Removed existing calibration {self.run_args.db_name}')
        return
    

    def make_study(self):
        ''' Make a study, deleting one if it already exists '''
        op = import_optuna()
        if not self.run_args.keep_db:
            self.remove_db()
        
        sampler = self.make_sampler()

        output = op.create_study(storage=self.run_args.storage, study_name=self.run_args.name, sampler=sampler) 
        return output


    def calibrate(self, calib_pars=None, genotype_pars=None, hiv_pars=None, verbose=True, load=True, tidyup=True, useRAM=False, **kwargs):
        '''
        Actually perform calibration.

        Args:
            calib_pars (dict): if supplied, overwrite stored calib_pars
            verbose (bool): whether to print output from each trial
            plot_intermediate: whether to output a plot of all the intermediate trial values as the calibration progressed
            kwargs (dict): if supplied, overwrite stored run_args (n_trials, n_workers, etc.)
        '''
        op = import_optuna()

        # Load and validate calibration parameters
        if calib_pars is not None:
            self.calib_pars = calib_pars
        if genotype_pars is not None:
            self.genotype_pars = genotype_pars
        if hiv_pars is not None:
            self.hiv_pars = hiv_pars
        if (self.calib_pars is None) and (self.genotype_pars is None) and (self.hiv_pars is None):
            errormsg = 'You must supply calibration parameters (calib_pars or genotype_pars) either when creating the calibration object or when calling calibrate().'
            raise ValueError(errormsg)
        self.run_args.update(kwargs) # Update optuna settings (by updating the dictionary that holds the data we will pass)

        # Run the optimization
        t0 = sc.tic() #together with sc.toc() this will calculate how long the optimisation took
        if not useRAM:
            self.make_study() #we don't need to bother getting the returned value from this function; the created study is stored, and accessible from, our saved database file. Our workers will acces it through this.
            self.run_workers()
            study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.name, sampler = self.make_sampler())
            self.best_pars = sc.objdict(study.best_params)
            self.elapsed = sc.toc(t0, output=True)

            #Update the plot of learning curve for this calibration object
            self.learning_curve = op.visualization.plot_optimization_history(study)
        else:
            study = self.run_workers_RAM(1)


        # Collect analyzer results
        # Load a single sim
        sim = self.sim # TODO: make sure this is OK #sc.jsonpickle(self.study.trials[0].user_attrs['jsonpickle_sim'])
        self.ng = sim['n_genotypes']
        self.glabels = [g.upper() for g in sim['genotype_map'].values()]

        # Replace with something else, this is fragile
        self.analyzer_results = []
        self.sim_results = []
        self.extra_sim_results = []
        self.unloadable_study_indices= [] #indices of studies which failed to load; these cannot be queried for data to plot
        if load:
            print('Loading saved results...')
            for trial in study.trials:
                n = trial.number
                try:
                    filename = self.tmp_filename % trial.number
                    results = sc.load(filename)
                    self.sim_results.append(results['sim'])
                    self.analyzer_results.append(results['analyzer'])
                    self.extra_sim_results.append(results['extra_sim_results'])
                    if tidyup:
                        try:
                            os.remove(filename)
                            print(f'    Removed temporary file {filename}')
                        except Exception as E:
                            errormsg = f'Could not remove {filename}: {str(E)}'
                            print(errormsg)
                    print(f'  Loaded trial {n}')
                except Exception as E:
                    errormsg = f'Warning, could not load trial {n}: {str(E)}'
                    self.unloadable_study_indices.append(n)
                    print(errormsg)

        # Compare the results
        self.initial_pars, self.par_bounds = self.sim_to_sample_pars()
        self.parse_study(study) #parses the study into a dataframe

        # Tidy up
        self.calibrated = True #i.e. this calibration object has now been used for a calibration
        if not self.run_args.keep_db:
            self.remove_db()

        print(f"Calibration took {self.elapsed} seconds")

        return self


    def parse_study(self, study):
        '''Parse the study into a data frame -- called automatically '''
        best = study.best_params
        self.best_pars = best

        print('Making results structure...')
        results = []
        n_trials = len(study.trials)
        failed_trials = []
        pruned_trials = []
        for trial in study.trials:
            data = {'index':trial.number, 'mismatch': trial.value}
            for key,val in trial.params.items():
                data[key] = val
            if data['mismatch'] is None:
                failed_trials.append(data['index'])
            elif trial.number in self.unloadable_study_indices: 
                pruned_trials.append(data['index'])
            else:
                results.append(data)
        print(f'Processed {n_trials} trials; {len(failed_trials)} failed, {len(pruned_trials)} pruned')

        keys = ['index', 'mismatch'] + list(best.keys())
        data = sc.objdict().make(keys=keys, vals=[])
        for i,r in enumerate(results):
            for key in keys:
                if key not in r:
                    warnmsg = f'Key {key} is missing from trial {i}, replacing with default'
                    hpm.warn(warnmsg)
                    r[key] = best[key]
                data[key].append(r[key])
        self.data = data
        self.df = pd.DataFrame.from_dict(data)
        self.df = self.df.sort_values(by=['mismatch']) # Sort

        return


    def to_json(self, filename=None, indent=2, **kwargs):
        '''
        Convert the data to JSON.
        '''
        order = np.argsort(self.df['mismatch'])
        json = []
        for o in order:
            row = self.df.iloc[o,:].to_dict()
            rowdict = dict(index=row.pop('index'), mismatch=row.pop('mismatch'), pars={})
            for key,val in row.items():
                rowdict['pars'][key] = val
            json.append(rowdict)
        self.json = json
        if filename:
            return sc.savejson(filename, json, indent=indent, **kwargs)
        else:
            return json


    def plot(self, res_to_plot=None, fig_args=None, axis_args=None, data_args=None, show_args=None, do_save=None,
             fig_path=None, do_show=True, plot_type='sns.boxplot', **kwargs):
        '''
        Plot the calibration results

        Args:
            res_to_plot (int): number of results to plot. if None, plot them all
            fig_args (dict): passed to pl.figure()
            axis_args (dict): passed to pl.subplots_adjust()
            data_args (dict): 'width', 'color', and 'offset' arguments for the data
            do_save (bool): whether to save
            fig_path (str or filepath): filepath to save to
            do_show (bool): whether to show the figure
            kwargs (dict): passed to ``hpv.options.with_style()``; see that function for choices
        '''

        # Import Seaborn here since slow
        if sc.isstring(plot_type) and plot_type.startswith('sns'):
            import seaborn as sns 
            if plot_type.split('.')[1]=='boxplot':
                extra_args=dict(boxprops=dict(alpha=.3), showfliers=False)
            else: extra_args = dict()
            plot_func = getattr(sns, plot_type.split('.')[1])
        else:
            plot_func = plot_type
            extra_args = dict()

        # Handle inputs
        fig_args = sc.mergedicts(dict(figsize=(12,8)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.08, right=0.92, bottom=0.08, top=0.92), axis_args)
        d_args = sc.objdict(sc.mergedicts(dict(width=0.3, color='#000000', offset=0), data_args))
        show_args = sc.objdict(sc.mergedicts(dict(show=dict(tight=True, maximize=False)), show_args))
        all_args = sc.objdict(sc.mergedicts(fig_args, axis_args, d_args, show_args))

        # Pull out results to use
        analyzer_results = sc.dcp(self.analyzer_results)
        sim_results = sc.dcp(self.sim_results)

        # Get rows and columns
        if not len(analyzer_results) and not len(sim_results):
            errormsg = 'Cannot plot since no results were recorded)'
            raise ValueError(errormsg)
        else:
            all_dates = [[date for date in r.keys() if date != 'bins'] for r in analyzer_results[0].values()]
            dates_per_result = [len(date_list) for date_list in all_dates]
            other_results = len(sim_results[0].keys())
            n_plots = sum(dates_per_result) + other_results
            n_rows, n_cols = sc.get_rows_cols(n_plots)

        # Initialize
        fig, axes = pl.subplots(n_rows, n_cols, **fig_args)
        if n_plots>1:
            for ax in axes.flat[n_plots:]:
                ax.set_visible(False)
            axes = axes.flatten()
        pl.subplots_adjust(**axis_args)

        # Pull out attributes that don't vary by run
        age_labels = sc.objdict()
        for resname,resdict in zip(self.age_results_keys, analyzer_results[0].values()):
            age_labels[resname] = [str(int(resdict['bins'][i])) + '-' + str(int(resdict['bins'][i + 1])) for i in range(len(resdict['bins']) - 1)]
            age_labels[resname].append(str(int(resdict['bins'][-1])) + '+')

        # determine how many results to plot 
        if res_to_plot is not None:
            index_to_plot = self.df.iloc[0:res_to_plot, 0].values

            #We need to pad analyzer_results and sim_results in the missing positions (from the pruned trials), so that indices align
            #Note that, as pruned trials are not amde part of the results structure in self.parse_study, they will not form part of the results to be outputted
            self.unloadable_study_indices.sort() #increasing order sort
            for i in self.unloadable_study_indices:
                analyzer_results.insert(i,None)
                sim_results.insert(i,None)
            
            analyzer_results = [analyzer_results[i] for i in index_to_plot]
            sim_results = [sim_results[i] for i in index_to_plot]


        # Make the figure
        with hpo.with_style(**kwargs):

            plot_count = 0
            for rn, resname in enumerate(self.age_results_keys):
                x = np.arange(len(age_labels[resname]))  # the label locations

                for date in all_dates[rn]:

                    # Initialize axis and data storage structures
                    if n_plots>1:
                        ax = axes[plot_count]
                    else:
                        ax = axes
                    bins = []
                    genotypes = []
                    values = []

                    # Pull out data
                    thisdatadf = self.target_data[rn][(self.target_data[rn].year == float(date)) & (self.target_data[rn].name == resname)]
                    unique_genotypes = thisdatadf.genotype.unique()

                    # Start making plot
                    if 'genotype' in resname:
                        for g in range(self.ng):
                            glabel = self.glabels[g].upper()
                            # Plot data
                            if glabel in unique_genotypes:
                                ydata = np.array(thisdatadf[thisdatadf.genotype == glabel].value)
                                ax.scatter(x, ydata, color=self.result_args[resname].color[g], marker='s', label=f'Data - {glabel}')

                            # Construct a dataframe with things in the most logical order for plotting
                            for run_num, run in enumerate(analyzer_results):
                                genotypes += [glabel]*len(x)
                                bins += x.tolist()
                                values += list(run[resname][date][g])

                        # Plot model
                        modeldf = pd.DataFrame({'bins':bins, 'values':values, 'genotypes':genotypes})
                        ax = plot_func(ax=ax, x='bins', y='values', hue="genotypes", data=modeldf, **extra_args)

                    else:
                        # Plot data
                        ydata = np.array(thisdatadf.value)
                        ax.scatter(x, ydata, color=self.result_args[resname].color, marker='s', label='Data')
                        # Construct a dataframe with things in the most logical order for plotting
                        for run_num, run in enumerate(analyzer_results):
                            bins += x.tolist()
                            values += list(run[resname][date])

                        # Plot model
                        modeldf = pd.DataFrame({'bins':bins, 'values':values})
                        ax = plot_func(ax=ax, x='bins', y='values', data=modeldf, color=self.result_args[resname].color, **extra_args)

                    # Set title and labels
                    ax.set_xlabel('Age group')
                    ax.set_title(f'{self.result_args[resname].name}, {date}')
                    ax.legend()
                    ax.set_xticks(x, age_labels[resname], rotation=45)
                    plot_count += 1

            for rn, resname in enumerate(self.sim_results_keys):
                if n_plots > 1:
                    ax = axes[plot_count]
                else:
                    ax = axes
                bins = sc.autolist()
                values = sc.autolist()
                thisdatadf = self.target_data[rn+sum(dates_per_result)][self.target_data[rn + sum(dates_per_result)].name == resname]
                ydata = np.array(thisdatadf.value)
                x = np.arange(len(ydata))
                ax.scatter(x, ydata, color=pl.cm.Reds(0.95), marker='s', label='Data')

                # Construct a dataframe with things in the most logical order for plotting
                for run_num, run in enumerate(sim_results):
                    bins += x.tolist()
                    if sc.isnumber(run[resname]):
                        values += sc.promotetolist(run[resname])
                    else:
                        values += run[resname].tolist()

                # Plot model
                modeldf = pd.DataFrame({'bins': bins, 'values': values})
                ax = plot_func(ax=ax, x='bins', y='values', data=modeldf, **extra_args)

                # Set title and labels
                date = thisdatadf.year[0]
                ax.set_title(self.result_args[resname].name + ', ' + str(date))
                ax.legend()
                if 'genotype_dist' in resname:
                    ax.set_xticks(x, self.glabels)
                    ax.set_xlabel('Genotype')
                plot_count += 1

        return hppl.tidy_up(fig, do_save=do_save, fig_path=fig_path, do_show=do_show, args=all_args)
    
    def plot_learning_curve(self, filename: str = None):
        '''
        Plot the learning curve for our most recent calibration.

        Pre: calibated

        Args:
            filename (str): filename to which to save the learning curve (if left as None, curve is only shown to the user and not saved)
        '''
        if not self.calibrated:
            raise UncalibratedException("Cannot plot learning curve without first calibrating a simulation. First call the calibrate function on this object.")

        self.learning_curve.show() #this plot is defined in the calibrate function

        if filename is not None:
            self.learning_curve.write_image(filename)

"""