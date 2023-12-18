import numpy as np
import hpvsim as hpv

if __name__ == '__main__':
    rel_trans_vals = np.linspace(0.0, 0.1, 6) # Sweep from with 6 values (no.linspace(a,b,c) returns c maximally and evenly spaced values in [a,b])
    sims = []
    for rel_trans in rel_trans_vals:
        sim = hpv.Sim(beta=rel_trans, label=f'Rel trans HPV = {rel_trans}') #the beta value is rate at which infectious people spread the disease (beta higher => faster spread)
        sims.append(sim)
    msim = hpv.MultiSim(sims) #creates a multisim out of a list of sims
    msim.run()
    msim.plot()
    msim.plot('infections');