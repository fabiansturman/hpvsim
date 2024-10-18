import hpvsim as hpv
import sciris as sc
import scipy
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import multiprocess
import multiprocess.pool

import optuna as op

'''
This file defines a version of the Calibration class which can perform several calibrations independantly and concurrently.
This allows for calibrations to be run with several seeds at the same time - particularly useful when there are enough resources to run...
... a calibration beyond the point where it converges, at which point it may be beneficial to run a new calibration with a different seed.
'''


def import_optuna():
    ''' A helper function to import Optuna, which is an optional dependency '''
    try:
        import optuna as op # Import here since it's slow
    except ModuleNotFoundError as E: # pragma: no cover
        errormsg = f'Optuna import failed ({str(E)}), please install first (pip install optuna)'
        raise ModuleNotFoundError(errormsg)
    return op

#Define a means for us to use multiprocessing which both allows processes to spawn off child processes (so that we can run calibrations with several workers concurrently - i.e. processes are not daemon), and allows complex arguments to be passed from the main program to the process (here, dill 'pickling' is used instead of standard python pickling)
# See https://pypi.org/project/multiprocess/ for more, and https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic for the inspiration of the following 12 lines of code.
class NoDaemonProcess(multiprocess.Process):
    @property
    def daemon(self):
        return False
    @daemon.setter
    def daemon(self, val):
        pass
class NoDaemonProcessPool(multiprocess.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NoDaemonProcessPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc


class UncalibratedException(Exception):
    '''
    A type of exception which can be thrown when we try to do things with an uncalibrated calibration object which can only be done with a calibrated one 
    '''
    pass

class MultiCalibration:
    '''
    A class which handles running several instances of a calibration in parelell, each with a different seed, and then formatting/outputting results in useful ways. 

    Parameters of the class constructor:
        sim             = the simulation we are calibrating
        seeds: [int]    = the list of random seeds with which we will intialise each calibration instance (and the corresponding deep copy of sim). Default value [0].
        workers:int     = the number of independant workers we will use in calibration. 
        name: str       = the filename database for this calibration
        calibrate_args  = a dictionary of arguments to pass to the Calibration.calibrate function when it is called on each Calibration instance
        **kwargs        = all other argumements which are passed directly into the constructors of our calibration instances

        
            If workers<=len(seeds), our outcome is deterministic - as each seed's calibration is performed independantly and its hpvsim instance is seeded according to the seed (+len(seeds))

    Instance variables of the class:
        calibrations    = a list of all the calibrations objects which we will calibrate in parelell

    '''

    def __init__(   self, 
                    sim,
                    seeds=[0],
                    workers=None, 
                    name="",
                    calibrate_args={},
                    **kwargs):
        if workers is None or workers<len(seeds): 
            workers=len(seeds)
        
        #Populate the list of calibration instances we will be running in parelell
        self.calibrations = []
        for seed in seeds:
            sim_dcp = sc.dcp(sim) #set up the simulation for this calibration instance
            sim_dcp["rand_seed"] = len(seeds) + seed #so that random values from our simulation are unrelated to those picked by optuna (by offsetting our seed in a way which also gives a seed unique across the whole program)

            calib_name = name + "_calibInstance" + str(seed)
            kwargs["name"] = calib_name #add this name to the dictionary of arguments we will pass to our calibration instance
            kwargs["rand_seed"] = seed
            
            #give this calibration instance the appropriate number of workers
            if workers>=len(seeds):
                # At least as many workers as we have seeds
                # Distribute our workers amongst our seeds. Each seed gets at either {workers//len(seeds)} workers or {1 + workers//len(seeds)} 
                if workers % len(seeds) > 0:
                    kwargs["n_workers"] = workers//len(seeds) + 1 #we have spare workers when dividing by the number of seeds we have, so offer an extra worker to this seed. 
                    workers -=1 #to keep track of one of our 'excess' workers being allocated to this seed
                else:
                    kwargs["n_workers"] = workers//len(seeds) #we have no (more) 'excess' workers to allocate; the number of workers we have is a multiple of the number of seeds we have
            else:
                #If we have more seeds than we have workers, we have every calibration isntance have a sigle worker, and then our scheduler here (the NoDaemonProcessPool) splits the seeds up between our workers, with each worker processing seeds sequentially
                kwargs["n_workers"] = 1

            calib = hpv.Calibration(sim_dcp, **kwargs)
            self.calibrations.append(calib)
        
        self.name=name
        self.seeds = seeds
        self.calibrate_args = calibrate_args
        self.calibrated = False 
        self.df         = None          #If self.calibrated, this stores a dataframe of all the trials that have been run across all calibrations
                #self.df is in the same format as the Calibration.df instance variable in the HPVsim calibration class, created by the parse_study function (https://docs.idmod.org/projects/hpvsim/en/latest/api/_autosummary/hpvsim.calibration.Calibration.html#hpvsim.calibration.Calibration)
                #... the only difference is that in the 'index' column, each value is of the format n:k where n is the random seed number of the calibration from which the trial comes and k is the trial number within this calibration
                #DTI: self.df is sorted increasinly by mismatch (so the best parameter set is at the top)

    def worker(self, calibration):
        '''
        calibration: HPVsim.Calibration instance

        Runs calibration.calibrate() for the provided Calibration instance, returns this result
        '''
        cal = calibration.calibrate(**self.calibrate_args) #cal is the calibrated Calibration object

        #Save information from the calibration
        final_loss = cal.learning_curve.data[1].y[-1]
        pruning_mode = cal.pruning
        pruner = cal.pruner
        with open(f"{self.name}_results.txt", "a") as f:
            print(f"(pmode={pruning_mode}, pruner={pruner}, seed={cal.sim["rand_seed"]}, name={cal.name}, data_distib={cal.data_distribution}): (final loss={final_loss}, duration={cal.elapsed}, trials processed={cal.processed_trials}, failed={cal.failed_trials}, pruned={cal.pruned_trials})", file=f)

        return cal

    def calibrate(self):
        '''
        Returns a list of all our calibrated Calibration instances 
        '''
        #We run the calibrations
        if len(self.calibrations) > 1:
            with NoDaemonProcessPool(len(self.calibrations)) as pool:
                output = sc.parallelize(self.worker, iterarg=self.calibrations, parallelizer=pool.map)    #Iterates through each calibration, running each cal on a new thread. Each cal them instantiates a new thread for each worker in it - so C calibrations each with W workers will require C + CW threads if W>1, and C threads if W=1.
        else: #If we just need to run one worker, no need to use multiprocessing
            output = [self.worker(self.calibrations[0])]

        self.calibrations = output
        self.calibrated = True

        for prc in multiprocess.active_children():
            prc.terminate()

        #Populate self.df
        dfs = []
        for calibration in self.calibrations: #add the calibration's dataframe to our list of dataframes, with its index updated to differentiate them between calibrations
            df = sc.dcp(calibration.df) #make a deep copy so any changes to indices of the calibration dataframe do not tamper with the calibration object itself
            seed = calibration.run_args["rand_seed"]
            height=df.shape[0]
            df = df.sort_values(by=['index']) # Sort increasingly by index s.t. we can replace them accurately
            newindices = []
            for i in range(height): newindices.append(f"{seed}:{i}")
            newindices = pd.Series(newindices, name='index')
            df.update(newindices)
            dfs.append(df)
        self.df = pd.concat(dfs)

        self.df = self.df.sort_values(by=['mismatch']) # Sort increasingly by mismatch s.t. the best are at the top


        return self.calibrations
    
    def to_json(self, filename=None, indent=2, **kwargs):
        '''
        Convert the data to JSON. As in the hpv.Calibration class
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

    
    def get_best_params(self, n:int):
        '''
        Returns the n best parameter configurations across all the calibrations that make up this instance as a dataframe
        '''
        if not self.calibrated:
            raise UncalibratedException("this instance has not yet been calibrated")
        
        
        return self.df.head(n) 

    
    def normal_confidence_interval(self, data, alpha):
        '''
        Returns (l,u) where l and u are the lower and upper bounds respectively of the 100(1-alpha)%-confidence interval for the mean of the provided list of data.
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
        return (Xbar-offset, Xbar+offset)


    
    def get_learning_curves(self, title=""):
        '''
        Returns learning curves for each calibration, supimposed into 1 chart, as a plotly.graph_objects.Figure instance
        '''

        if not self.calibrated:
            raise UncalibratedException("this instance has not yet been calibrated")

        plot = self.calibrations[0].learning_curve
        seed = self.calibrations[0].run_args["rand_seed"]
        newnames= {'Objective Value':f'Seed {seed} Samples', 'Best Value': f'Seed {seed} learning curve', 'Infeasible Trial':f'Seed {seed}infeasible trials'}
        plot.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                               legendgroup = newnames[t.name])
                                               )
        plot=go.Figure(data = plot.data[1])
        
        for calib in self.calibrations[1:]:
            plotpart = calib.learning_curve
            seed = calib.run_args["rand_seed"]
            newnames= {'Objective Value':f'Seed {seed} Samples', 'Best Value': f'Seed {seed} learning curve', 'Infeasible Trial':f'Seed {seed}infeasible trials'}
            plotpart.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                                legendgroup = newnames[t.name])
                                                )
            plot = go.Figure(data = plot.data + (plotpart.data[1],))

        plot.layout.title = title
        
        return plot
    

    def get_learning_curves_CI(self, title="", name='trace 1', colour='green'):
        '''
        Returns a as a plotly.graph_objects.Figure instance, showing a 95% CI for the learning curve at each point of calibration, calcualted from the calibrations performed by this MultiCalibration instance
        '''
        if not self.calibrated:
            raise UncalibratedException("this instance has not yet been calibrated")

        #Calcualte, for every trial number, the average of all the learning curve polylines at that point
        calib_data = []
        for calib in self.calibrations:
            #FOr each calibration, we will add a list bestobj_line=[x_0,x_1,x_2,...] to calib_data, where x_0,x_1,... is the best obj value at trial 0, trial 1, etc
            x = calib.learning_curve.data[1]["x"]
            y = calib.learning_curve.data[1]["y"]

            bestobj_line = []
            bestobj_sofar = x[0] #while iterating through the trials of this calibration, this stores the best objective value found so far
            for i in range(x[-1]+1):
                #Update the best objective value found so far if we need to
                if i in x:
                    k= x.index(i) #k is the position in x where we have trial i
                    bestobj_sofar=y[k]
                #Add the best objective found so far to our line
                bestobj_line.append(bestobj_sofar)
            calib_data.append(bestobj_line)



        means = [] #For each trial number (i.e. each x-value) we store the mean 'best cost' over all our trials...
        lowers = []; uppers = []#...as well as lower and upper bounds for our 95% confidence interval
        xs=[]
        n_lines = len(calib_data) #the number of lines over which we take the average at each point
        for i in range(len(calib_data[0])): #len(calib_data[0]) is just how many elements are in one of our polylines
            ys = []
            for line in calib_data:
                ys.append(line[i])
            (l,u) = self.normal_confidence_interval(ys, 0.05)

            xs.append(i)
            means.append(sum(ys)/n_lines)
            lowers.append(l)
            uppers.append(u)
            
        print(xs)
        print(uppers)
        print(lowers)
        print(means)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=uppers, fill=None, line_color=colour,mode='lines',line_width=0,showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=lowers, fill='tonexty',line_color=colour, mode='lines', line_width=0,showlegend=False))
        fig.add_trace(go.Scatter(name=name,x=xs, y=means, fill=None,line=dict(color=colour, width=2)))
        fig.update_layout(xaxis_title="trial #", yaxis_title='loss', title=title)
        
        return fig
            

    def get_calibration_times(self):
        '''
        Returns a list of the times it took to calibrate each of the Calibration instances in this multicalibration 
        '''
        if not self.calibrated:
            raise UncalibratedException("this instance has not yet been calibrated")

        durations = []
        for calib in self.calibrations:
            durations.append(calib.elapsed)

        return durations
    


    


##############

def make_multical(name,
                     seeds=[1,2],
                     n_workers = 20,
                     n_agents = 5*10e3,
                     n_trials = 5000,
                     leakiness = 1,
                     pruning = -1, #-1:no pruning ; 1:basic pruning ; 2:leaky pruning ; 3:adaptive pruning
                     pruner = None,
                     datafiles = [],
                     verbose = True
                     ):
    '''
    Sets up and returns a multicalibration instance with the desired number seeds, and the specified pruning information
    '''
    rss = seeds

    pars = dict(n_agents=n_agents,
                start=1975,
                end=2023, 
                dt=0.25, 
                location='nigeria', #united kingdom
                verbose = 0, #1 means verbose, 0 means not verbose
                )
    sim = hpv.Sim(pars)

    #Parameters for calibration
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
    calib_pars = dict(
            beta=[0.05, 0.010, 0.20],
            f_cross_layer= [0.05, 0.01, 0.2],
            m_cross_layer= [0.05, 0.01, 0.2],
          #  condoms = [0.27,0.1,0.38],
        )
   
    results_to_plot = []#'cancer_incidence', 'asr_cancer_incidence']
    

    mc = MultiCalibration(sim=sim, seeds=rss, workers=n_workers, 
                                name=name,
                                calibrate_args={'plots':["learning_curve", "timeline"]},
                                calib_pars=calib_pars,
                                genotype_pars=genotype_pars,
                                extra_sim_result_keys=results_to_plot,
                                datafiles=datafiles,
                                total_trials=n_trials,
                                keep_db=True, 
                                sampler_type = "tpe",
                                sampler_args = None, 
                                leakiness = leakiness,
                                pruning = pruning,
                                pruner = pruner,
                                verbose = verbose)
    
    return mc


if __name__ == "__main__":
    #parameters - we are doing 4 pruning setups (3 with pruners and 1 with no pruning) and have 16 threads available, so can run 4 calibrations concurrently per setup (this computer has 20 theads, but 19 usable due to this program also taking a thread)
    name = "C:\\Users\\fabia\\Documents\\GitHub\\hpvsim\\RawTemp\\Sep2724_2"
    seeds = [6,7,8,9,10] #Trying 5 seeds to see if it still takes between 9-10 times as long to run any given one of those cals compared to leaving all 20 workers to that cal, and that these times are pretty consistnet (as this main calling thread should be able to sleep)
    n_workers = 1   #number of workers per calibration
    n_agents = 10e3
    n_trials = 5000
    leakiness = 0.45
    pruning = 2 #-1:no pruning ; 1:basic pruning ; 2:leaky pruning ; 3:adaptive pruning
    verbose = False #set to True for output at the end of every completed trial
    datafiles =  [ 
                  'Set 3\\cancer_incidence_2000.csv',
                  'Set 3\\cancers_2010.csv',
                  'Set 3\\hpv_prevalence_2020.csv',
                ]
    

    pruners = [op.pruners.HyperbandPruner(min_resource=1, max_resource=n_trials, reduction_factor=3),
               op.pruners.SuccessiveHalvingPruner(reduction_factor=3)
                ]

    

    #set up multicalibrations
    mc = make_multical(name = f"{name}_hyperband_leaky",
                          seeds=seeds,
                          n_workers=n_workers,
                          n_agents = n_agents,
                          n_trials = n_trials,
                          leakiness=leakiness,
                          pruning=2,
                          pruner = pruners[0],
                          datafiles=datafiles,
                          verbose = verbose)
    #mc_p0 = make_multical(name = f"{name}_hyperband_full",
     #                     seeds=seeds,
      #                    n_workers=n_workers,
       #                   n_agents = n_agents,
        #                  n_trials = n_trials,
         #                 leakiness=leakiness,
          #                pruning=1,
           #               pruner = pruners[0],
            #              datafiles=datafiles,
             #             verbose = verbose)
    mc_p1 = make_multical(name = f"{name}_sh3_leaky",
                          seeds=seeds,
                          n_workers=n_workers,
                          n_agents = n_agents,
                          n_trials = n_trials,
                          leakiness=leakiness,
                          pruning=2,
                          pruner = pruners[1],
                          datafiles=datafiles,
                          verbose = verbose)
    #mc_p2 = make_multical(name = f"{name}_sh3_full",
     #                     seeds=seeds,
      #                    n_workers=n_workers,
       #                   n_agents = n_agents,
        #                  n_trials = n_trials,
         #                 leakiness=leakiness,
          #                pruning=1,
           #               pruner = pruners[1],
            #              datafiles=datafiles,
             #             verbose = verbose)
    
    #Add the calibrations of the pruning multicalibrations to our main multicalibration, to make a big multicalibration to do 4 types at the same time
        #This is not explicitly supported by the Multicalibration interface; but we are just adding extra calibrations to it to do. 
   # mc.calibrations += mc_p0.calibrations 
    mc.calibrations += mc_p1.calibrations 
   # mc.calibrations += mc_p2.calibrations 
    mc.seeds = mc.seeds*2#4 #this is just done so that there is a 1-2-1 correspondance between seeds in mc.seeds and (the seeds of) each calibration in mc.calibrations, for the plots made by this test rig

    #Calibrate all
    mc.calibrate()

    #Plot superimposed learning curve
    plt = mc.get_learning_curves(title = f"Learning curves for this multicalibration")
    plt.show()

    

    
    #Plot best fits of the cal
    i=0
    for c in mc.calibrations:
        c.plot(res_to_plot=4)
        i+=1


 