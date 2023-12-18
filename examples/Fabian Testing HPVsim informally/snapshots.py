#Snapshots take "pictures" of the sim.people object at different points in time.
#This is useful because, although most of the information from sim.people is retrievable at the end of the simulation from the stored events, it is much easier to see what is going on at the time

#The following example uses a snapshot to create a figure to demonstrate age mixing patterns among sexual contacts

import numpy as np
import hpvsim as hpv

if __name__ == "__main__":
    pars = dict(beta=0.5, n_agents=50e3, start=1970, n_years=60, dt=1., location='japan')

    #Make the snapshot object and add it to our sim before running it
    snap = hpv.snapshot(timepoints=['2020', '2030'])
    sim = hpv.Sim(pars, analyzers=snap)
    sim.run()

    a = sim.get_analyzer()
    people = a.snapshots[0]

    # Plot age mixing
    import pylab as pl
    import matplotlib as mpl
    fig, ax = pl.subplots(nrows=1, ncols=1, figsize=(5, 4))

    fc = people.contacts['m']['age_f'] # Get the age of female contacts in marital partnership
    mc = people.contacts['m']['age_m'] # Get the age of male contacts in marital partnership
    h = ax.hist2d(fc, mc, bins=np.linspace(0, 75, 16), density=True, norm=mpl.colors.LogNorm())
    ax.set_xlabel('Age of female partner')
    ax.set_ylabel('Age of male partner')
    fig.colorbar(h[3], ax=ax)
    ax.set_title('Marital age mixing 2020')
    pl.show();



    #Do the same again for 2030
    people = a.snapshots[1]

    # Plot age mixing
    import pylab as pl
    import matplotlib as mpl
    fig, ax = pl.subplots(nrows=1, ncols=1, figsize=(5, 4))

    fc = people.contacts['m']['age_f'] # Get the age of female contacts in marital partnership
    mc = people.contacts['m']['age_m'] # Get the age of male contacts in marital partnership
    h = ax.hist2d(fc, mc, bins=np.linspace(0, 75, 16), density=True, norm=mpl.colors.LogNorm())
    ax.set_xlabel('Age of female partner')
    ax.set_ylabel('Age of male partner')
    fig.colorbar(h[3], ax=ax)
    ax.set_title('Marital age mixing 2030')
    pl.show();