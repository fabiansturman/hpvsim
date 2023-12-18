#Often we will want to use multisims as they offer the most flexibility.
#However, in some cases, Scenario objects achieve the same thing more simply. 
#Unlike MultiSims, which do not care what sims you include - you just feed sims as a list into the multisim and it runs them -...
#..., scenarios always start from the same basesim (i.e. a sim with the same base parameters), and then modifies...
#... the parameters as we specify, before finally adding uncertainty if desired.

import hpvsim as hpv

if __name__ == "__main__":
    # Set base parameters -- these will be shared across all scenarios
    basepars = {'n_agents':10e3}

    # Configure the settings for each scenario
    scenarios = {'baseline': {
                'name':'Baseline',
                'pars': {}
                },
                'high_rel_trans': {
                'name':'High rel trans (0.75)',
                'pars': {
                    'beta': 0.75,
                    }
                },
                'low_rel_trans': {
                'name':'Low rel trans(0.25)',
                'pars': {
                    'beta': 0.25,
                    }
                },
                }

    # Run and plot the scenarios
    scens = hpv.Scenarios(basepars=basepars, scenarios=scenarios) # each scenario is the combination of our base parameters and the custom parameters for that specific scenario
    scens.run()
    scens.plot();