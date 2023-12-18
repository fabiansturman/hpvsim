import hpvsim as hpv


if __name__ =="__main__":
    #Define the parameters for the model
    pars = dict(
    n_agents      = 20e3,       # Population size
    n_years       = 70,         # Number of years to simulate
    verbose       = 0,          # Don't print details of the run
    rand_seed     = 2,          # Set a non-default seed
    genotypes     = [16, 18],   # Include the two genotypes of greatest general interest
    )

    prob = 0.6 # prob = 60% means we vaccinate 60% of girls (or girls and boys if we add sex = [0,1])

    vx = hpv.routine_vx(prob=prob, start_year=2015, age_range=[9,10], product='bivalent')

    # Create the sim with and without interventions
    orig_sim = hpv.Sim(pars, label='Baseline')
    sim = hpv.Sim(pars, interventions = vx, label='With vaccination')

    # Run and plot
    msim = hpv.parallel(orig_sim, sim)
    msim.plot();