#This code enhances the nigeria dataset, using a parameter set from a 4000-trial caliration with an 6000-agent sim  with final cost 0.94 to calibrate our sim to this data

import numpy as np
import sciris as sc
import hpvsim as hpv 

if __name__=="__main__":
    #Parameterise our sim with parameters from calibration (marked with a (*))
    genotype_pars = dict(
        hpv16=dict(
            cin_fn=dict(k=0.502955), #(*)
            dur_cin=dict(par1=8.17959) #(*)
        ),
        hpv18=dict(
            cin_fn=dict(k=0.218852), #(*)
            dur_cin=dict(par1=4.608712) #(*)
        ),
        hi5=dict(
            cin_fn=dict(k=0.695355), #(*)
            dur_cin=dict(par1=4.48064) #(*)
        )
    )

    pars = dict(beta=0.674216,  #(*)
                hpv_control_prob=0.376785, #(*)
                n_agents=6*10e3,
                start=1970,  end=2030, dt=0.25,
                location='nigeria',
                genotypes = [16,18,'hi5']
                )


    # Create the age analyzers.
    az = hpv.age_results(
        result_args=sc.objdict(
            # The keys of this dictionary are any results you want by age, and can be any key of sim.results. We are analyzing these 4 values by ages in 2000, 2010, 2015,
       #     hpv_prevalence=sc.objdict( 
        #        years=[2000,2010],
         #       edges=np.array([0., 15., 20., 25., 30., 35. ,40., 45., 50., 55., 65.,70.,75.,80.,85., 100.]), #define the extremities of the age buckets for each analyser
          #  ),
            cancers=sc.objdict(
                years=[2010],
                edges=np.array([0., 15., 20., 25., 30., 35. ,40., 45., 50., 55., 65.,70.,75.,80.,85., 100.]), 
            ),
            n_infected=sc.objdict(
                years=[2010],
                edges=np.array([0., 15., 20., 25., 30., 35. ,40., 45., 50., 55., 65.,70.,75.,80.,85., 100.]), 
            ),
            n_infectious=sc.objdict(
                years=[2010],
                edges=np.array([0., 15., 20., 25., 30., 35. ,40., 45., 50., 55., 65.,70.,75.,80.,85., 100.]), 
            ),
        )
    )

    

    sim = hpv.Sim(pars, genotypes=[16, 18], analyzers=[az])

    
    msim = hpv.MultiSim(sim)
    msim.run(n_runs=3) #here, we are running a single sim over 5 different instances; we get a different random seed each time so can calcualte empirical averages from this 
    

    for sim in msim.sims:
        #Prin the analyzer for each sim, to extract the data by age and manually take average
        a = sim.get_analyzer()
        a.plot()

    msim.mean() #calcualtes the mean over all the sims
    msim.plot()