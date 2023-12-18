import hpvsim as hpv

# Custom vaccination intervention
def custom_vx(sim):
    if sim.yearvec[sim.t] == 2000:
        target_group = (sim.people.age>9) * (sim.people.age<14)
        sim.people.peak_imm[0, target_group] = 1

pars = dict(
    location = 'tanzania', # Use population characteristics for Japan
    n_agents = 10e3, # Have 50,000 people total in the population (NOTE: actual tanzanian pop in 1980 was 19.2million, so 1 agent represents about 380 people)
    start = 1980, # Start the simulation in 1980
    n_years = 50, # Run the simulation for 50 years
    burnin = 10, # Discard the first 20 years as burnin period
    verbose = 0, # Do not print any output
)

# Running with multisims -- see Tutorial 3
s1 = hpv.Sim(pars, label='Default')
s2 = hpv.Sim(pars, interventions=custom_vx, label='Custom vaccination')
msim = hpv.MultiSim([s1, s2])
msim.run()
fig = msim.plot(['cancers', 'cins'])