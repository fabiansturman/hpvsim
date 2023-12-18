import hpvsim as hpv

#We define parameters for our simulation as a dictionary
pars = dict(
    n_agents = 10e3,
    genotypes = [16, 18, 'hr'], # Simulate genotypes 16 and 18, plus all other high-risk HPV genotypes pooled together
    start = 1980,
    end = 2030,
)



sim = hpv.Sim(pars)
x = sim.run()
sim.save("Fabian Testing HPVsim informally/my-first-saved-sim.sim") # Note: HPVsim always saves the people if the sim isn't finished running yet. Otherwise, it will not save the people of the model, as they are very large and if the random seed is saved can be regenerated, but we can save the people if we want with the keep_people argument!
print(x)
fig = sim.plot()
sim.to_excel("Fabian Testing HPVsim informally//my-first-sim.xlsx")

