import hpvsim as hpv


#NOTE: MultiSim makes use of a form of concurrency in python which can only be achieved by placing the code inside this sort of 'main' block! We get an error and non-terminating code otherwise!
if __name__ == '__main__':
    hpv.options(verbose=0)

    sim = hpv.Sim()
    msim = hpv.MultiSim(sim)
    msim.run(n_runs=6) #here, we are running a single sim over 5 different instances; we get a different random seed each time so can calcualte empirical averages from this 
    msim.plot()

    for sim in msim.sims:
        sim.brief()
    
    msim.mean() #calcualtes the mean over all the sims
    msim.plot()

    msim.median()
    msim.plot()

    msim.combine() #we use combine when we are treating each of the individual sims as part of a larger single sim, which we then want to combine into a final result
    msim.plot()
    
        #Each  of these ways of combining the constituent sims in our multisim just edit the msim.base_sim object and does not affect the actual list of stored sims, which is why we can go back and forth between them