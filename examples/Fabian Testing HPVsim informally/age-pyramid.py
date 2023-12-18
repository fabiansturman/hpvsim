#Age pyramids, like snapshots, take a picture of the people at a given point in time, and then bin them into age groups by sex


import numpy as np
import hpvsim as hpv


if __name__ == "__main__":
    # Create some parameters
    pars = dict(n_agents=50e3, start=2000, n_years=30, dt=0.5) #dt is measured in decades

    # Make the age pyramid analyzer
    age_pyr = hpv.age_pyramid(
        timepoints=['2010', '2020'],
        #datafile='south_africa_age_pyramid.csv', - if we have this as a datafile, we can load this in and plot alongside!
        edges=np.linspace(0, 100, 21))

    # Make the sim, run, get the analyzer, and plot
    sim = hpv.Sim(pars, location='south africa', analyzers=age_pyr)
    sim.run()
    a = sim.get_analyzer()
    fig = a.plot(percentages=True);