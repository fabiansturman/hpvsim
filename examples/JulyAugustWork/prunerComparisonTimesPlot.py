#can i take my lineplit-copy file and do little line segment plots and then just combine them? perhaps do fig=go.Figure();fig.add_trace(theplot) for each plot i want to add??


import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import scipy

## DATASET 3

fig=go.Figure()

# Adding min/max bars to plot
hyperband_minmax = [0.462,0.485]
median_minmax = [0.501,0.548]
shrf2_minmax = [0.461,0.510]
shrf3_minmax = [0.431,0.472]
shrf4_minmax = [0.428,0.470]
nop_minmax =  [1, 1.005]

fig.add_trace(go.Scatter(x=[0.5,3.5], y=[hyperband_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='blue', width=2))) #mean
fig.add_trace(go.Scatter(x=[0.5,3.5], y=[hyperband_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='blue', width=2))) #mean

fig.add_trace(go.Scatter(x=[5.5,8.5], y=[median_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='orange', width=2))) #mean
fig.add_trace(go.Scatter(x=[5.5,8.5], y=[median_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='orange', width=2))) #mean

fig.add_trace(go.Scatter(x=[10.5,13.5], y=[shrf2_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='green', width=2))) #mean
fig.add_trace(go.Scatter(x=[10.5,13.5], y=[shrf2_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='green', width=2))) #mean

fig.add_trace(go.Scatter(x=[15.5,18.5], y=[shrf3_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='red', width=2))) #mean
fig.add_trace(go.Scatter(x=[15.5,18.5], y=[shrf3_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='red', width=2))) #mean

fig.add_trace(go.Scatter(x=[20.5,23.5], y=[shrf4_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='purple', width=2))) #mean
fig.add_trace(go.Scatter(x=[20.5,23.5], y=[shrf4_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='purple', width=2))) #mean

fig.add_trace(go.Scatter(x=[25.5,28.5], y=[nop_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='brown', width=2))) #mean
fig.add_trace(go.Scatter(x=[25.5,28.5], y=[nop_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='brown', width=2))) #mean


#also plot mins and maxs!! !!!!!!

# Plot the 95% CIs
hyperband = [0.470,0.476,0.482]
median = [0.512,0.524,0.536]
shrf2 = [0.472,0.484,0.496]
shrf3 = [0.438,0.446,0.453]
shrf4 = [0.433,0.440,0.447]
nop = [0.995,1,1.005]


fig.add_hline(y=shrf4[2], line_width=1.5, line_dash='dash',line=dict(color='purple', width=2))
fig.add_hline(y=median[0], line_width=1.5, line_dash='dash',line=dict(color='orange', width=2))

fig.add_trace(go.Scatter(x=[0,4], y=[hyperband[2]]*2, fill=None, line_color='blue',mode='lines',line_width=0,showlegend=False)) #upper bound of CI
fig.add_trace(go.Scatter(x=[0,4], y=[hyperband[0]]*2, fill='tonexty',line_color='blue', mode='lines', line_width=0,showlegend=False)) #lower bound of CI
fig.add_trace(go.Scatter(x=[0,4], y=[hyperband[1]]*2, name='Average' ,fill=None,line=dict(color='blue', width=2))) #mean

fig.add_trace(go.Scatter(x=[5,9], y=[median[2]]*2, fill=None, line_color='orange',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[5,9], y=[median[0]]*2, fill='tonexty',line_color='orange', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[5,9], y=[median[1]]*2, name='Average' ,fill=None,line=dict(color='orange', width=2)))

fig.add_trace(go.Scatter(x=[10,14], y=[shrf2[2]]*2, fill=None, line_color='green',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[10,14], y=[shrf2[0]]*2, fill='tonexty',line_color='green', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[10,14], y=[shrf2[1]]*2, name='Average' ,fill=None,line=dict(color='green', width=2)))

fig.add_trace(go.Scatter(x=[15,19], y=[shrf3[2]]*2, fill=None, line_color='red',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[15,19], y=[shrf3[0]]*2, fill='tonexty',line_color='red', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[15,19], y=[shrf3[1]]*2, name='Average' ,fill=None,line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=[20,24], y=[shrf4[2]]*2, fill=None, line_color='purple',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[20,24], y=[shrf4[0]]*2, fill='tonexty',line_color='purple', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[20,24], y=[shrf4[1]]*2, name='Average' ,fill=None,line=dict(color='purple', width=2)))

fig.add_trace(go.Scatter(x=[25,29], y=[nop[2]]*2, fill=None, line_color='brown',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[25,29], y=[nop[0]]*2, fill='tonexty',line_color='brown', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[25,29], y=[nop[1]]*2, name='Average' ,fill=None,line=dict(color='brown', width=2)))



fig.update_traces()

fig.update_layout(
     xaxis=dict(
          tickmode='array',
          tickvals=[2,7,12,17,22,27],
          ticktext=['Hyperband', 'Median', 'Succ. Halving rf2','Succ. Halving rf3','Succ. Halving rf4', 'No Pruning']
     )
)

fig.update_layout(yaxis_title='Normalised Time for 5000 Trials', title='Dataset 3',
                  xaxis_title='Pruner',
                  template='simple_white')

fig.update_layout(
    title=dict(font=dict(size=30)),
    yaxis_title=dict(font=dict(size=20)),
    xaxis_title=dict(font=dict(size=20)),
    title_x = 0.5
)

fig.update_yaxes(range = [0,1.005])

fig.show()

## DATASET 2

fig=go.Figure()

# Adding min/max bars to plot
hyperband_minmax = [0.464,0.499]
median_minmax = [0.489, 0.548]
shrf2_minmax = [0.480,0.505]
shrf3_minmax = [0.468, 0.501]
shrf4_minmax = [0.442,0.482]
nop_minmax =  [0.993, 1.007]

fig.add_trace(go.Scatter(x=[0.5,3.5], y=[hyperband_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='blue', width=2))) #mean
fig.add_trace(go.Scatter(x=[0.5,3.5], y=[hyperband_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='blue', width=2))) #mean

fig.add_trace(go.Scatter(x=[5.5,8.5], y=[median_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='orange', width=2))) #mean
fig.add_trace(go.Scatter(x=[5.5,8.5], y=[median_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='orange', width=2))) #mean

fig.add_trace(go.Scatter(x=[10.5,13.5], y=[shrf2_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='green', width=2))) #mean
fig.add_trace(go.Scatter(x=[10.5,13.5], y=[shrf2_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='green', width=2))) #mean

fig.add_trace(go.Scatter(x=[15.5,18.5], y=[shrf3_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='red', width=2))) #mean
fig.add_trace(go.Scatter(x=[15.5,18.5], y=[shrf3_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='red', width=2))) #mean

fig.add_trace(go.Scatter(x=[20.5,23.5], y=[shrf4_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='purple', width=2))) #mean
fig.add_trace(go.Scatter(x=[20.5,23.5], y=[shrf4_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='purple', width=2))) #mean

fig.add_trace(go.Scatter(x=[25.5,28.5], y=[nop_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='brown', width=2))) #mean
fig.add_trace(go.Scatter(x=[25.5,28.5], y=[nop_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='brown', width=2))) #mean


#also plot mins and maxs!! !!!!!!

# Plot the 95% CIs
hyperband = [0.480,0.486,0.492]
median = [0.506,0.517,0.529]
shrf2 = [0.490,0.495,0.501]
shrf3 = [0.476,0.482,0.489]
shrf4 = [0.458,0.465,0.473]
nop = [0.997,1,1.003]

fig.add_hline(y=shrf4[2], line_width=1.5, line_dash='dash',line=dict(color='purple', width=2))
fig.add_hline(y=median[0], line_width=1.5, line_dash='dash',line=dict(color='orange', width=2))

fig.add_trace(go.Scatter(x=[0,4], y=[hyperband[2]]*2, fill=None, line_color='blue',mode='lines',line_width=0,showlegend=False)) #upper bound of CI
fig.add_trace(go.Scatter(x=[0,4], y=[hyperband[0]]*2, fill='tonexty',line_color='blue', mode='lines', line_width=0,showlegend=False)) #lower bound of CI
fig.add_trace(go.Scatter(x=[0,4], y=[hyperband[1]]*2, name='Average' ,fill=None,line=dict(color='blue', width=2))) #mean

fig.add_trace(go.Scatter(x=[5,9], y=[median[2]]*2, fill=None, line_color='orange',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[5,9], y=[median[0]]*2, fill='tonexty',line_color='orange', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[5,9], y=[median[1]]*2, name='Average' ,fill=None,line=dict(color='orange', width=2)))

fig.add_trace(go.Scatter(x=[10,14], y=[shrf2[2]]*2, fill=None, line_color='green',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[10,14], y=[shrf2[0]]*2, fill='tonexty',line_color='green', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[10,14], y=[shrf2[1]]*2, name='Average' ,fill=None,line=dict(color='green', width=2)))

fig.add_trace(go.Scatter(x=[15,19], y=[shrf3[2]]*2, fill=None, line_color='red',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[15,19], y=[shrf3[0]]*2, fill='tonexty',line_color='red', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[15,19], y=[shrf3[1]]*2, name='Average' ,fill=None,line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=[20,24], y=[shrf4[2]]*2, fill=None, line_color='purple',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[20,24], y=[shrf4[0]]*2, fill='tonexty',line_color='purple', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[20,24], y=[shrf4[1]]*2, name='Average' ,fill=None,line=dict(color='purple', width=2)))

fig.add_trace(go.Scatter(x=[25,29], y=[nop[2]]*2, fill=None, line_color='brown',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[25,29], y=[nop[0]]*2, fill='tonexty',line_color='brown', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[25,29], y=[nop[1]]*2, name='Average' ,fill=None,line=dict(color='brown', width=2)))




fig.update_traces()

fig.update_layout(
     xaxis=dict(
          tickmode='array',
          tickvals=[2,7,12,17,22,27],
          ticktext=['Hyperband', 'Median', 'Succ. Halving rf2','Succ. Halving rf3','Succ. Halving rf4', 'No Pruning']
     )
)

fig.update_yaxes(range = [0,1.005])

fig.update_layout(yaxis_title='Normalised time to complete 5000 trials', title='Dataset 6',
                  xaxis_title='Pruner',
                  template='simple_white')

fig.update_layout(
    title=dict(font=dict(size=30)),
    yaxis_title=dict(font=dict(size=20)),
    xaxis_title=dict(font=dict(size=20)),
    title_x = 0.5
)

fig.show()
