#Testing rig to generate data for testing parelell calibration.
#We are generating data for a contour/3D plot with 'best loss at end of calibration' on z (vertical) axis, time to perform calibration on the x axis, and #seeds on y axis
#Such a graph is generated using 4 workers, 8 workers, adn 12 workers
from multirunner import * 
import datetime
import sciris as sc

log_filename="logs\\8workerssims.txt"


def single_multicalibration_nigeria(name, seeds:[int], workers:int, trials:int):
    '''
    Returns a multirunner instance configured with the specified number of seeds and workers, calibrating over {trials} trials to the data in the given datafiles
    '''
    #Set up a sim using some parameters which can get an acceptable fit to the Nigeria dataset
    pars = dict(n_agents=10e4, #increasing resolution to 10e4 from 10e3 to get better fits 
                start=1980,
                end=2023, 
                dt=0.25, 
                location='nigeria', 
                genotypes=[16, 18, 'hi5'],
                verbose = 0, 
                )
    sim = hpv.Sim(pars)

    #Define the parameters we will be tuning
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

    #Set up our multirunner instance
    datafiles=[ 'docs\\tutorials\\nigeria_cancer_cases.csv',
                'docs\\tutorials\\nigeria_cancer_types.csv',]
    multirunner = Multirunner(sim=sim, seeds=seeds, workers=workers, 
                                name=name,
                                calibrate_args={'plots':["timeline"]},
                                calib_pars=calib_pars,
                                genotype_pars=genotype_pars,
                                extra_sim_result_keys=[],
                                datafiles=datafiles,
                                total_trials=trials,
                                keep_db=True, 
                                sampler_type = "tpe",
                                sampler_args = None, 
                                prune = False )
    
    return multirunner


def log(line:str):
    '''
    Writes data relating to the generation of this data to a log, so we can keep track of they key milestones
    '''
    with open(log_filename, "a") as file:
        file.write(f"{datetime.datetime.now()}> {line} \n")
    


if __name__ == "__main__":
    log("Starting data generation")

    points={} #points will be stored in the format (#seeds, time):(obj). This allows us to run teh code several times to get several values for each key and take the average, for more robust results
    i = 0
    n_workers=8
    for seeds in [1,2,4,6,8]:#have up to as many seeds as we have workers
        multirunner = single_multicalibration_nigeria(name = f"CalibrationRawResults\\tpsc_2Mar24_9_seeds{seeds}",
                                                          seeds = list(range(seeds)), #the set of seeds is not random, so that we have determinism. WE can change this and take the average of results to get a more reliable plot if needed
                                                          workers = n_workers,
                                                          trials = 7000)
        multirunner.calibrate()
        
        log(f"Finished multicalibration with {seeds} seeds. ")
        
        #Calcualte the objective value at each timepoint
        obj_times = {}
        times = []
        for calib in multirunner.calibrations: #combine all objectivevalue-time dictionaries of the calibrations into one big one
            obj_times.update(calib.get_objective_time_dictionary())
            times += calib.get_objective_time_dictionary().keys()
        times.sort()
        best_obj_so_far = obj_times[times[0]] 
        for time in times: #iterate through time, making our 'best objective value seen so far' be the best one seen across all the calibrations at this point in time
            if obj_times[time] < best_obj_so_far:
                best_obj_so_far = obj_times[time]
            points[(seeds, time)] = best_obj_so_far

            #note that for creating averages over multiple trials, linearly inerpolate to get cost against trial# nd then to get time/trial, then avergae out time/trial and use those vaues. this is valid as the timelines are all basically linear, but im not sure if they are allthe same gradient. i assume they would be tho in which case this is not needed ofc,, bascially as long as all of them last about the same time for a fixed setup, a basic pointwise average is good

        log(f"Calculated dictionary entries for this multicalibration with {seeds} seeds: {obj_times}")

        best_obj = multirunner.df['mismatch'].min()
        best_param = multirunner.get_best_params(1)
        log(f"Best mismatch={best_obj} with paramater config={best_param}.")


    log(f"Final data dictionary: {points}")
    print(points)

    import pickle
    with open("resultsdict"+log_filename, "wb") as f:
        pickle.dump(points, f)
        
        
        
    '''
        multirunner = single_multicalibration_nigeria(name = f"CalibrationRawResults\\tpsc_1Mar24_3_seeds{seeds}_trials{trials}_",
                                                          seeds = list(range(seeds)), #the set of seeds is not random, so that we have determinism. WE can change this and take the average of results to get a more reliable plot if needed
                                                          workers = n_workers,
                                                          trials = trials)
        multirunner.calibrate()

        
            ##surely for a value for seeds, we can make a single multirunner and just run it for , say, 6561 trials.
            #then to get the value for (seeds, 3), for instance, we look at the best objective across the first 3 trials of all the seeds, and then the latest seed to get to 3.
            #as, if one process finishes early, it is not like resources will move on to the next process and speed it up, this should be a fair calculation, and save a lot of time


        df = multirunner.df

        trial_lengths=[]#trial_lengths[i] will contain a list [l1,l2,...] of lengths of the 1st, 2nd, etc trials of multirunner.calibrations[i]
        for calib in multirunner.calibrations:
            calib_trial_lengths = []
'''
    '''

        for trials in [3,9,27,81,243,500,729,1000,1500,2187,3000,4000,5000,6561]: # number of trials follows a c. logarithmic scale (some extra points added for ) as we expect signficiantly dimisnishing returns as trial # increases 
            i+=1
            log(f"Starting multicalibration {i}: seeds={seeds}, trials={trials}.")

            multirunner = single_multicalibration_nigeria(name = f"CalibrationRawResults\\tpsc_1Mar24_4_seeds{seeds}_trials{trials}_",
                                                          seeds = list(range(seeds)), #the set of seeds is not random, so that we have determinism. WE can change this and take the average of results to get a more reliable plot if needed
                                                          workers = n_workers,
                                                          trials = trials)
            t0 = sc.tic()
            multirunner.calibrate()
            elapsed_time = sc.toc(t0, output=True)

            best_obj = multirunner.df['mismatch'].min()
            best_param = multirunner.get_best_params(1)
            points[(seeds,trials)] = (best_obj, elapsed_time)

            #Log the elapsed time of this calibration, the best objective value that it achieved, and the parameter configuration that achieved this mismatch
            log(f"Finished multicalibration {i} in {elapsed_time} seconds. Best mismatch={best_obj} with paramater config={best_param}.")

            #Plot timelines of each calibration in this multirunner, to verify that nothing has messed up the timing of the calibrations, and that the time is reliable
            for c in multirunner.calibrations:
                c.plot_timeline()
    
    print(points)
    log(str(points))
    '''

