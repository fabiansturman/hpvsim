#we can also run different sims in parelell using the shorthand hpv.parelell, which is an alias for hpv.MultiSim().run()

import hpvsim as hpv

def custom_vx(sim):
    if sim.yearvec[sim.t] == 2000:
        target_group = (sim.people.age>9) * (sim.people.age<14)
        sim.people.peak_imm[0, target_group] = 1

if __name__ == "__main__":
    pars = dict(
        location = 'tanzania', # Use population characteristics for Japan
        n_agents = 10e3, # Have 50,000 people total in the population
        start = 1980, # Start the simulation in 1980
        n_years = 50, # Run the simulation for 50 years
        burnin = 10, # Discard the first 20 years as burnin period
        verbose = 0, # Do not print any output
    )

    # Running with multisims -- see Tutorial 3
    s1 = hpv.Sim(pars, label='Default')
    s2 = hpv.Sim(pars, interventions=custom_vx, label='Custom vaccination')
    hpv.parallel(s1, s2).plot(['hpv_incidence', 'cancer_incidence']);


"""NOTE:Because multiprocess pickles the sims when running them, sims[0] (before being run by the multisim) 
and msim.sims[0] are not the same object. After calling msim.run(), always use sims from the multisim object, 
not from before. In contrast, if you don’t run the multisim (e.g. if you make a multisim from already-run sims),
 then sims[0] and msim.sims[0] are indeed exactly the same object.
"""
