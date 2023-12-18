#Analysers are objects which do not change the behaviour of a simulation, but just report on its internal state
#They almost have something to do with sim.people (an object which contains all our agents and methods from changing them from one state to another (e.g. from susceptible to infected))

#The results in sim.results already include results disaggregated by age (e.g. sim.results['cancers_by_age']) ...
#... but these results use standardised age bins which may not match the avaliable data (and we want to be compatible with the avaiable data we are dealing with always!)

#This exmaple shows how we can set up results by age

import numpy as np
import sciris as sc
import hpvsim as hpv

if __name__=="__main__":
    # Create some parameters, setting beta (per-contact transmission probability) higher
    # to create more cancers for illutration
    pars = dict(beta=0.5, n_agents=50e3, start=1970, n_years=50, dt=1., location='tanzania')

    # Also set initial HPV prevalence to be high, again to generate more cancers
    pars['init_hpv_prev'] = {
        'age_brackets'  : np.array([  12,   17,   24,   34,  44,   64,    80, 150]),
        'm'             : np.array([ 0.0, 0.75, 0.9, 0.45, 0.1, 0.05, 0.005, 0]),
        'f'             : np.array([ 0.0, 0.75, 0.9, 0.45, 0.1, 0.05, 0.005, 0]),
    }

    # Create the age analyzers.
    az1 = hpv.age_results(
        result_args=sc.objdict(
            hpv_prevalence=sc.objdict( # The keys of this dictionary are any results you want by age, and can be any key of sim.results
                years=2019, # List the years that you want to generate results for
                edges=np.array([0., 15., 20., 25., 30., 40., 45., 50., 55., 65., 100.]), #WHAT IS MEANT BY EDGES???
            ),
            hpv_incidence=sc.objdict(
                years=2019,
                edges=np.array([0., 15., 20., 25., 30., 40., 45., 50., 55., 65., 100.]),
            ),
            cancer_incidence=sc.objdict(
                years=2019,
                edges=np.array([0.,20.,25.,30.,40.,45.,50.,55.,65.,100.]),
            ),
            cancer_mortality=sc.objdict(
                years=2019,
                edges=np.array([0., 20., 25., 30., 40., 45., 50., 55., 65., 100.]),
            )
        )
    )

    sim = hpv.Sim(pars, genotypes=[16, 18], analyzers=[az1])
    sim.run()
    a = sim.get_analyzer()
    a.plot();