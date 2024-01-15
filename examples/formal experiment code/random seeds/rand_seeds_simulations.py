import hpvsim as hpv
import sciris as sc

def make_intervention_sim():
    '''
    Function to make a sim with interventions, inspired by the tutorial on interventions on the HPVsim org website; https://docs.idmod.org/projects/hpvsim/en/latest/tutorials/tut_interventions.html accessed 15th Jan 2024
    '''
    prob = 0.6
    screen      = hpv.routine_screening(start_year=2015, prob=prob, product='via', label='screen') # Routine screening

    to_triage   = lambda sim: sim.get_intervention('screen').outcomes['positive'] # Define who's eligible for triage
    triage      = hpv.routine_triage(eligibility=to_triage, prob=prob, product='hpv', label='triage') # Triage people
    
    to_treat    = lambda sim: sim.get_intervention('triage').outcomes['positive'] # Define who's eligible to be assigned treatment
    assign_tx   = hpv.routine_triage(eligibility=to_treat, prob=prob, product='tx_assigner', label='assign_tx') # Assign treatment
    
    to_ablate   = lambda sim: sim.get_intervention('assign_tx').outcomes['ablation'] # Define who's eligible for ablation treatment
    ablation    = hpv.treat_num(eligibility=to_ablate, prob=prob, product='ablation') # Administer ablation
    
    to_excise   = lambda sim: sim.get_intervention('assign_tx').outcomes['excision'] # Define who's eligible for excision
    excision    = hpv.treat_delay(eligibility=to_excise, prob=prob, product='excision') # Administer excision


    # Define the parameters
    pars = dict(
        n_agents      = 20e3,       # Population size
        n_years       = 35,         # Number of years to simulate
        rand_seed     = 2,          # Set a non-default seed
        genotypes     = [16, 18],   # Include the two genotypes of greatest general interest
    )

    sim = hpv.Sim(pars, interventions = [screen, triage, assign_tx, ablation, excision]) #we add interventions as a list of interventions
    return sim


def verify_seed_usage(sim, n_copies: int, rand_seeds:[int] = None, plot: bool = False):
    '''
    Informally verifies that simulations with a fixed rand_seed value run deterministically.

    Takes a simulation and gives it the provided random seed. Makes n_copies deepcopies of it, and checks outputs are all equal.

    Args:
        sim (sim.Sim)       : a simulation object
        n_copies (int)      : the number of concurrent deep copies of the simulation to run
        rand_seeds ([int])  : random seeds to use with each of the deep copies respectively. If none, we keep the seed of the original sim.
                                pre= if rand_seed is not None, then len(rand_seed) = n_copies
        plot (bool)         : plots the graph of each simulation iff plot
    
    Returns:
        different_sims (  [(sim.Sim, sim.Sim)]  ): a list of all the pairs of sims that were found to be different to each other
    '''
    #Create sims
    sims = []
    for i in range(n_copies):
        sims.append(sc.dcp(sim, die=False, verbose=True))      #verbose=True, so that if a deep copy fails, we are notified
        print(f"Sim copy {i} made")

    #Update seeds. Note that if rand_seeds is None, all the deep copies will remain with the same random seed as that of the original sim.
    if rand_seeds is not None:
        for i in range(n_copies):
            sims[i]['rand_seed'] = rand_seeds[i]

    #Run the sims in parelell
    if n_copies > 1:                                        # Normal use case: run in parallel
        run_sim = lambda sim : sim.run()                    # Takes a simulation and runs it
        sims = sc.parallelize(run_sim, iterarg=sims)               # Map the sim-running function over all our sims, in parallel
    else:                                                   # If just running one sim, no need to do anything special
        sims = [sims[0].run()]

    #It is easier to check if all the sims are the same and raise an error if not, than comparing eevry pair of sims to find ones that are different
    different_sims = [] #Stores any sims with differences that have been found, in pairs (2-tuples)
    for i in range(n_copies - 1):
        try:
            hpv.diff_sims(sims[i], sims[i+1], output=True, die=True)
        except:
            print(f"Found differences between sims {i} and {i+1}")
            different_sims.append((sims[i],sims[i+1]))

    if len(different_sims) == 0:
        print(f"All {len(sims)} sims identical")

    if plot:
        for s in sims:
            s.plot()

    return different_sims



if __name__ == "__main__":
    sim = make_intervention_sim()
    verify_seed_usage(sim, n_copies=20, plot=False) #, rand_seed=[1,2]