#can i take my lineplit-copy file and do little line segment plots and then just combine them? perhaps do fig=go.Figure();fig.add_trace(theplot) for each plot i want to add??


import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import scipy

#Loss after time T plot

fig=go.Figure()


no_datapoints = 19

# Adding min/max bars to plot
median_minmax = [2.167/no_datapoints,2.993/no_datapoints]
shrf4_minmax = [2.518/no_datapoints,3.296/no_datapoints]
nop_minmax =  [2.713/no_datapoints,3.287/no_datapoints]

fig.add_trace(go.Scatter(x=[0.5,3.5], y=[median_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='orange', width=2))) #mean
fig.add_trace(go.Scatter(x=[0.5,3.5], y=[median_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='orange', width=2))) #mean

fig.add_trace(go.Scatter(x=[5.5,8.5], y=[shrf4_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='purple', width=2))) #mean
fig.add_trace(go.Scatter(x=[5.5,8.5], y=[shrf4_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='purple', width=2))) #mean

fig.add_trace(go.Scatter(x=[10.5,13.5], y=[nop_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='brown', width=2))) #mean
fig.add_trace(go.Scatter(x=[10.5,13.5], y=[nop_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='brown', width=2))) #mean


# Plot the 95% CIs
median = [2.535/no_datapoints,2.666/no_datapoints,2.797/no_datapoints]
shrf4 = [2.823/no_datapoints,2.944/no_datapoints,3.065/no_datapoints]
nop = [2.858/no_datapoints,2.956/no_datapoints,3.053/no_datapoints]

fig.add_hline(y=nop[0], line_width=1.5, line_dash='dash',line=dict(color='brown', width=2))
fig.add_hline(y=shrf4[0], line_width=1.5, line_dash='dash',line=dict(color='purple', width=2))


fig.add_trace(go.Scatter(x=[0,4], y=[median[2]]*2, fill=None, line_color='orange',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[0,4], y=[median[0]]*2, fill='tonexty',line_color='orange', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[0,4], y=[median[1]]*2, name='Average' ,fill=None,line=dict(color='orange', width=2)))

fig.add_trace(go.Scatter(x=[5,9], y=[shrf4[2]]*2, fill=None, line_color='purple',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[5,9], y=[shrf4[0]]*2, fill='tonexty',line_color='purple', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[5,9], y=[shrf4[1]]*2, name='Average' ,fill=None,line=dict(color='purple', width=2)))

fig.add_trace(go.Scatter(x=[10,14], y=[nop[2]]*2, fill=None, line_color='brown',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[10,14], y=[nop[0]]*2, fill='tonexty',line_color='brown', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[10,14], y=[nop[1]]*2, name='Average' ,fill=None,line=dict(color='brown', width=2)))



fig.update_traces()

fig.update_layout(
     xaxis=dict(
          tickmode='array',
          tickvals=[2,7,12],
          ticktext=['Median', 'Succ. Halving rf4', 'No Pruning']
     )
)

fig.update_layout(yaxis_title='Loss / datapoint ', title='Best Loss after 20,000s (c. 3.5hrs)',
                  xaxis_title='Pruner',
                  template='simple_white')

fig.update_layout(
    title=dict(font=dict(size=30)),
    yaxis_title=dict(font=dict(size=20)),
    xaxis_title=dict(font=dict(size=20)),
    title_x = 0.5
)

fig.show()

## Trials after time T plots

fig=go.Figure()

# Adding min/max bars to plot
median_minmax = [7993,8709]
shrf4_minmax = [8186,8723]
nop_minmax =  [7127,7847]

fig.add_trace(go.Scatter(x=[0.5,3.5], y=[median_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='orange', width=2))) #mean
fig.add_trace(go.Scatter(x=[0.5,3.5], y=[median_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='orange', width=2))) #mean

fig.add_trace(go.Scatter(x=[5.5,8.5], y=[shrf4_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='purple', width=2))) #mean
fig.add_trace(go.Scatter(x=[5.5,8.5], y=[shrf4_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='purple', width=2))) #mean

fig.add_trace(go.Scatter(x=[10.5,13.5], y=[nop_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='brown', width=2))) #mean
fig.add_trace(go.Scatter(x=[10.5,13.5], y=[nop_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='brown', width=2))) #mean


# Plot the 95% CIs
median = [8102,8252,8403]
shrf4 = [8308,8411,8514]
nop = [7286,7422,7558]


fig.add_trace(go.Scatter(x=[0,4], y=[median[2]]*2, fill=None, line_color='orange',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[0,4], y=[median[0]]*2, fill='tonexty',line_color='orange', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[0,4], y=[median[1]]*2, name='Average' ,fill=None,line=dict(color='orange', width=2)))

fig.add_trace(go.Scatter(x=[5,9], y=[shrf4[2]]*2, fill=None, line_color='purple',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[5,9], y=[shrf4[0]]*2, fill='tonexty',line_color='purple', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[5,9], y=[shrf4[1]]*2, name='Average' ,fill=None,line=dict(color='purple', width=2)))

fig.add_trace(go.Scatter(x=[10,14], y=[nop[2]]*2, fill=None, line_color='brown',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[10,14], y=[nop[0]]*2, fill='tonexty',line_color='brown', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[10,14], y=[nop[1]]*2, name='Average' ,fill=None,line=dict(color='brown', width=2)))



fig.update_traces()

fig.update_layout(
     xaxis=dict(
          tickmode='array',
          tickvals=[2,7,12],
          ticktext=['Median', 'Succ. Halving rf4', 'No Pruning']
     )
)

fig.update_layout(yaxis_title='Number of Trials', title='Trials completed or pruned in 20,000s (c. 3.5hrs)',
                  xaxis_title='Pruner',
                  template='simple_white')

fig.update_layout(
    title=dict(font=dict(size=30)),
    yaxis_title=dict(font=dict(size=20)),
    xaxis_title=dict(font=dict(size=20)),
    title_x = 0.5
)

fig.show()