import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import scipy

def normal_confidence_interval(data, alpha):
        '''
        Returns (l,mu,u) where l and u are the lower and upper bounds respectively of the 100(1-alpha)%-confidence interval for the mean of the provided list of data, and mu is the mean.
        We assume the data is normally distributed.

        We use the result that for X sampled iid from N(mu, v), (Xbar-mu)/(S/sqrt(n)) ~ t_{n-1}, where Xbar is the sample mean and S^2 is the sample variance [Niel Laws, 2023, A9 Statistics Lecture Notes Oxford University] 

        Parameters:
            data    = a list of numeric data
            alpha   = a real number in (0,1)
        
        Pre:
            alpha<-(0,1)
            len(data) >= 2 (else its sample variance is ill-defined)
        '''
        n=len(data)
        #Calculate sample mean and sample variance
        Xbar=0
        for x in data:
            Xbar += x
        Xbar /= n

        S2 = 0
        for x in data:
            S2 += (x-Xbar) ** 2
        S2 /= (n-1)
        S = S2 ** 0.5

        #Calculate CI
        offset = scipy.stats.t.cdf(1-alpha/2, n-1) * S/(n ** 0.5)
        return [Xbar-offset,Xbar, Xbar+offset]

a = 0.05
hyperband_times = normal_confidence_interval([12402,12646,12796,12830,12905],a)
median_times = normal_confidence_interval([13021,13031,12679,12699,12745],a)
shrf2_times = normal_confidence_interval([13369,13589,13778,13856,14075],a)
shrf3_times = normal_confidence_interval([11671,11728,11757,11773,11818,11878,11945,11982,12019,12040],a)
shrf4_times = normal_confidence_interval([11826,11917,11929,11970,12224],a)
nop_times = normal_confidence_interval([19816,19888,19934,19939,19944],a)
'''
hyperband_times = [12402,12646,12796,12830,12905]
median_times = [13021,13031,12679,12699,12745]
shrf2_times = [13369,13589,13778,13856,14075]
shrf3_times = [11671,11728,11757,11773,11818,11878,11945,11982,12019,12040]
shrf4_times = [11826,11917,11929,11970,12224]
nop_times = [19816,19888,19934,19939,19944]
'''




dic = {'Time (s)': hyperband_times+median_times+shrf2_times+shrf3_times+shrf4_times+nop_times,
       'Pruning': ['Hyperband']*len(hyperband_times) + ['Median']*len(median_times) + ['Succ. Halving (2)']*len(shrf2_times) + ['Succ. Halving (3)']*len(shrf3_times) + ['Succ. Halving (4)']*len(shrf4_times) + ['No Pruning']*len(nop_times) }


dic1 = {'Time (s)': hyperband_times,
       'Pruning': ['Hyperband']*len(hyperband_times) }

dic2 = {'Time (s)': median_times,
       'Pruning': ['Median']*len(median_times) }

f1 = px.box(pd.DataFrame(dic1), x='Pruning', y="Time (s)" , color="Pruning")
f1 = go.Figure()
f1.add_trace(go.Box(y=hyperband_times))
f2 = px.box(pd.DataFrame(dic2), x='Pruning', y="Time (s)" , color="Pruning")
f1.update_traces(q1=[hyperband_times[0]],
                 lowerfence=[hyperband_times[0]],
                 q3=[hyperband_times[2]])
f2.update_traces(q1=[median_times[0]],
                 q3=[median_times[2]])
f1.show()
f2.show()


tracefig = go.Figure()



df = pd.DataFrame(dic)
print(df)
fig = px.box(df, x='Pruning', y="Time (s)" , color="Pruning",
              width=560, height=560,
              template='simple_white', points='all')
fig.update_layout(boxgap=0, boxgroupgap=0.1)
fig.update_traces(line_width=0.8)



fig.show()



