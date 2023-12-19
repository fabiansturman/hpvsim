#Making my own products: my own screening, treatment, and vaccination, and testing them in a 

#This code is based on: the HPVsim tutorial, the documentation, and the .csv file data for tx, dx, and vx products

import hpvsim as hpv
import pandas as pd

if __name__ == "__main__":
    ##Making my own products##

    #Treatment products (tx) are defined by:
    fabians_tx_data = pd.DataFrame({'name':'Fabian\'s custom treatment', 
                                    'state':['precin', 'cin', 'cancerous'], #the stage ('state') of a HPV infection at which the treatment is applied
                                    'genotype':'all', #the HPV genotypes the treatment can counter
                                    'efficacy':[.2, .9, .9] #the efficacy of the treatment at each stage ('state') - I THINK THIS IS IT, RIGHT??
                                    })
    fabians_tx = hpv.tx(df=fabians_tx_data)
        #Note: in the tutorial, instead of 'cin' it has 'precin','cin1', 'cin2', 'cin3', 'cancerous' which appear to not all exist anymore. Rather, our options appear to be 'precin', 'cin', 'latent', 'cancerous', of which we can pick which ones are relevant to our product

    #Diagnostic products (dx) are defined by:
    fabians_dx_data = pd.DataFrame({'name':'Fabian\'s custom diagnostic test (can detect precin and cin and always detects cancer)',
                                    'state':['susceptible', 'susceptible', 'latent','latent', 'precin','precin', 'cin','cin','cancerous','cancerous'], 
                                    'genotype':'all',
                                    'result':['negative','positive','negative','positive','negative','positive','negative','positive','negative','positive'],
                                    'probability':[ 1,0,1,0,0.3,0.7,0.1, 0.9, 0, 1] 
                                    })
        #The above is best read as giving us tuples (state, result, probability) telling us P(we are diagnosed with result R | we are in state S) for every feasible combination to get us a full distribution for each possible state
    fabians_dx = hpv.dx(df=fabians_dx_data)


    #Vaccination products (vx) are defined by:
    fabians_vx_data = pd.DataFrame({'name':'Fabian\'s vaccine that is just bivalent (just limited to hpv16 and hpv18 but you could type the rest in and get full bivalent)',
                                    'genotype':['hpv16', 'hpv18'],
                                    'rel_imm':[0.95,0.95]
                                    })
    
    fabians_vx = hpv.vx(genotype_pars=fabians_vx_data, #information from our DataFrame (basically an excel clipping) about our vaccination's effectiveness against particular genotypes (only need to specify the relevant genotypes!)
                        imm_init={'dist':'normal', 'par1':0.9, 'par2':0.03} #samples peak immunity from a distribution as defined in the list - see utils.sample for more info
                            )

    fabians_followup_vx_data = pd.DataFrame({'name':'Fabian\'s vaccine ',
                                    'genotype':['hpv16', 'hpv18'], #only need to specifify the genotypes that matter for the vaccine!
                                    'rel_imm':[1,1] #defining immunity as fixed values in the dataframe
                                    })

    fabians_followup_vx = hpv.vx(genotype_pars=fabians_followup_vx_data, #information from our DataFrame (basically an excel clipping) about our vaccination's effectiveness against particular genotypes (only need to specify the relevant genotypes!)
                              imm_boost = 1.3 #our peak immunity is multiplied by this for subsequent vaccinations
                            )


     #IF WE WANT, WE CAN ALWAYS ADD INTERVENTIONS TO THE EXCEL FILES IN hpvsim.data AS THESE FILES ARE LOADED IN EACH TIME!!
     # (...this bypasses the need to create the classes here, though it is messier - ideally those excel files should only contain real life data!!) 



    
    ##Define the interventions themselves that we are using##       . Note how we assign the interventions labels which are chained together to form a treatment process/algorithm
    prob = 0.9        #(For simplicity, I am using the same probability for each intervention)

    screen      = hpv.routine_screening(start_year=2020, prob=prob, product='via', label='screen') #Define the routing screening
    
    to_triage   = lambda sim: sim.get_intervention('screen').outcomes['positive'] #Determine who is eligable for triage to be all those who, when screened, get a positive result
    triage      = hpv.routine_triage(eligibility=to_triage, prob=prob, product='hpv', label='triage') #Defines the triage process (i.e. the preliminary assessment of patients to determine the treatment they need)
                        #i.e. an agent has the hpv product applied to them iff they have a 'positive'-label outcome from the screening intervention
                        #TODO: I *think* the 'hpv' product is a dx product that acts as an indicator function of having hpv, i.e. by using it for the triage process we assume that they perfectly determine who does and doesnt have HPV??

    to_treat    = lambda sim: sim.get_intervention('triage').outcomes['positive'] #Determine who needs treating
    assign_tx   = hpv.routine_triage(eligibility=to_treat, prob=prob, product=fabians_dx, label='assign_tx') #Assign treatments to patients
                        #i.e. an agent is eligable to be assigned a treatment by the tx_assigner product iff they have a 'positive' outcome from the triage intervention
                        #Note: we often use the tx_assigner product. It  is a triage product (i.e. a *dx* product; it is diagnostic), and it is defined clearly in the products_dx.csv file: it will pick a treatment category for the patient to go into according to a probability distribition which is conditional on the actual stage of infection of the patient

    to_fabiantreat   = lambda sim: sim.get_intervention('assign_tx').outcomes['positive']  # Define who's eligible for this treatment (from the treatment assignment stage)
    fabiantreat    = hpv.treat_num(eligibility=to_fabiantreat, prob=prob, product=fabians_tx) # Administer this treatment
    

        #Note: we may further chain treatments together to create a more complex algorithm, as has been done in the tutorial
    

    
    ##Set up a prophylactic vaccination##
    vx = hpv.routine_vx(prob=prob, start_year=2020, age_range=[9,10], product=fabians_vx)


    ##Create and run sims, plot data##
    pars = dict(
        n_agents    = 20e3,
        n_years     = 50,
        verbose     = 0,
        rand_seed   = 2,            #This sets a non-default random seed
        genotypes   = [16,18]       #We are including the two genootypes of greatest general interest in our simulation
    )

    baseline_sim = hpv.Sim(pars, label="Baseline")
    sim_vx_only = hpv.Sim(pars, interventions = [vx] , label='With vaccination only')
    sim_1 = hpv.Sim(pars, interventions = [screen, triage, assign_tx, fabiantreat], label='With screening and treatment')
    sim_2 = hpv.Sim(pars, interventions = [screen, triage, assign_tx, fabiantreat, vx], label='With vaccination, screening and treatment')

    #sim_vx_only.run()
    #sim_vx_only.plot()

    msim = hpv.parallel(baseline_sim, sim_1, sim_vx_only, sim_2)
    msim.plot()