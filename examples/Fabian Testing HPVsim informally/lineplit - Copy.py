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
        return (Xbar-offset,Xbar, Xbar+offset)

#A plot of the best final cost of 30 calibrations after either 6000 trials with pruning, or 6000 trials without pruning; without, it is a decent bit slower

#NIGERIA
y_p_NG = [1.77,1.98,2.32,2.58,2.62]#0,20,1,16,28
y_np_NG = [1.77,1.98,1.83,2.21,2.15]
stats_y_p_NG = normal_confidence_interval(y_p_NG, 0.05)
stats_n_np_NG = normal_confidence_interval(y_np_NG, 0.05)


fig = go.Figure()
#Add a trace for each seed, showing how it does after 6000 trials in a range of contexts
for i in range(len(y_p_NG)): #iterating through nigeria
    fig.add_trace(go.Scatter(x=['5k Trials','10k Trials'], y=[y_p_NG[i], y_np_NG[i]], mode='lines', line_width=0.4, name=f'Seed={i}'))


#Add a trace for the 95% CI
fig.add_trace(go.Scatter(x=['5k Trials','10k Trials'], y=[stats_y_p_NG[2], stats_n_np_NG[2]], fill=None, line_color='green',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=['5k Trials','10k Trials'], y=[stats_y_p_NG[0], stats_n_np_NG[0]], fill='tonexty',line_color='green', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=['5k Trials','10k Trials'], y=[stats_y_p_NG[1], stats_n_np_NG[1]], name='Average' ,fill=None,line=dict(color='black', width=2)))
fig.update_layout(xaxis_title="Number of trials", yaxis_title='Best Loss', title='Loss after 5k trials against loss after 10k trials')
fig.update_xaxes(type='category')


fig.show()