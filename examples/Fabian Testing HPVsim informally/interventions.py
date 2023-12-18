import hpvsim as hpv

if __name__=="__main__":
    # Define a series of interventions to screen, triage, assign treatment, and administer treatment

    #NOTE: There are two key categories of interventions: routine ones, which take place yearly between two years (e.g. yearly between 2020 and 2030), and campaigns, which are passed in a list of years in which they occur, e.g: hpv.campaign_screening(years=[2020,2030], prob=0.2) means the intervention is delivered twice (once in 2020, then again in 2030)

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
        verbose       = 0,          # Don't print details of the run
        rand_seed     = 2,          # Set a non-default seed
        genotypes     = [16, 18],   # Include the two genotypes of greatest general interest
    )

    # Create the sim with and without interventions
    orig_sim = hpv.Sim(pars, label='Baseline')
    sim = hpv.Sim(pars, interventions = [screen, triage, assign_tx, ablation, excision], label='With screen & treat') #we add interventions as a list of interventions

    # Run and plot
    msim = hpv.parallel(orig_sim, sim)
    msim.plot();