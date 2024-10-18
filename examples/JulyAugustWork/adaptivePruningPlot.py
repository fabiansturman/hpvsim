#can i take my lineplit-copy file and do little line segment plots and then just combine them? perhaps do fig=go.Figure();fig.add_trace(theplot) for each plot i want to add??


import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import scipy

fig=go.Figure()



no_datapoints = 42

# Adding min/max bars to plot
fullhb_minmax = [5.54/no_datapoints,10.78/no_datapoints]
leakyhb_minmax = [5.471/no_datapoints,6.399/no_datapoints]
adaphb_minmax = [5.153/no_datapoints,6.296/no_datapoints ]
fullsh3_minmax = [5.25/no_datapoints,11.54/no_datapoints]
leakysh3_minmax = [5.314/no_datapoints,6.111/no_datapoints]
adapsh3_minmax = [5.311/no_datapoints,6.159/no_datapoints]
nop_minmax =  [0.130, 0.148]

fig.add_trace(go.Scatter(x=[0.5,3.5], y=[fullhb_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='blue', width=2))) #mean
fig.add_trace(go.Scatter(x=[0.5,3.5], y=[fullhb_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='blue', width=2))) #mean

fig.add_trace(go.Scatter(x=[5.5,8.5], y=[leakyhb_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='mediumblue', width=2))) #mean
fig.add_trace(go.Scatter(x=[5.5,8.5], y=[leakyhb_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='mediumblue', width=2))) #mean

fig.add_trace(go.Scatter(x=[10.5,13.5], y=[adaphb_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='darkblue', width=2))) #mean
fig.add_trace(go.Scatter(x=[10.5,13.5], y=[adaphb_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='darkblue', width=2))) #mean

fig.add_trace(go.Scatter(x=[15.5,18.5], y=[fullsh3_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='orangered', width=2))) #mean
fig.add_trace(go.Scatter(x=[15.5,18.5], y=[fullsh3_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='orangered', width=2))) #mean

fig.add_trace(go.Scatter(x=[20.5,23.5], y=[leakysh3_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='#FF2200', width=2))) #mean
fig.add_trace(go.Scatter(x=[20.5,23.5], y=[leakysh3_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='#FF2200', width=2))) #mean

fig.add_trace(go.Scatter(x=[25.5,28.5], y=[adapsh3_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='red', width=2))) #mean
fig.add_trace(go.Scatter(x=[25.5,28.5], y=[adapsh3_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='red', width=2))) #mean


fig.add_trace(go.Scatter(x=[30.5,33.5], y=[nop_minmax[0]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='brown', width=2))) #mean
fig.add_trace(go.Scatter(x=[30.5,33.5], y=[nop_minmax[1]]*2, name='Average', mode='lines' ,fill=None,line=dict(color='brown', width=2))) #mean



# Plot the 95% CIs


fullhb = [6.059/no_datapoints,6.916/no_datapoints,7.77/no_datapoints]
leakyhb = [5.902/no_datapoints,6.026/no_datapoints,6.149/no_datapoints]
adaphb = [5.527/no_datapoints,5.714/no_datapoints,5.901/no_datapoints]
fullsh3 = [6.488/no_datapoints,7.534/no_datapoints,8.580/no_datapoints]
leakysh3 = [5.590/no_datapoints,5.733/no_datapoints,5.877/no_datapoints]
adapsh3 = [5.542/no_datapoints,5.685/no_datapoints,5.829/no_datapoints]
nop = [5.623/no_datapoints,5.781/no_datapoints,5.94/no_datapoints]

fig.add_hline(y=nop[0], line_width=1.5, line_dash='dash')
fig.add_hline(y=nop[2], line_width=1.5, line_dash='dash')


fig.add_trace(go.Scatter(x=[0,4], y=[fullhb[2]]*2, fill=None, line_color='blue',mode='lines',line_width=0,showlegend=False)) #upper bound of CI
fig.add_trace(go.Scatter(x=[0,4], y=[fullhb[0]]*2, fill='tonexty',line_color='blue', mode='lines', line_width=0,showlegend=False)) #lower bound of CI
fig.add_trace(go.Scatter(x=[0,4], y=[fullhb[1]]*2, name='Average' ,fill=None,line=dict(color='blue', width=2))) #mean

fig.add_trace(go.Scatter(x=[5,9], y=[leakyhb[2]]*2, fill=None, line_color='mediumblue',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[5,9], y=[leakyhb[0]]*2, fill='tonexty',line_color='mediumblue', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[5,9], y=[leakyhb[1]]*2, name='Average' ,fill=None,line=dict(color='mediumblue', width=2)))

fig.add_trace(go.Scatter(x=[10,14], y=[adaphb[2]]*2, fill=None, line_color='darkblue',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[10,14], y=[adaphb[0]]*2, fill='tonexty',line_color='darkblue', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[10,14], y=[adaphb[1]]*2, name='Average' ,fill=None,line=dict(color='darkblue', width=2)))

fig.add_trace(go.Scatter(x=[15,19], y=[fullsh3[2]]*2, fill=None, line_color='orangered',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[15,19], y=[fullsh3[0]]*2, fill='tonexty',line_color='orangered', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[15,19], y=[fullsh3[1]]*2, name='Average' ,fill=None,line=dict(color='orangered', width=2)))

fig.add_trace(go.Scatter(x=[20,24], y=[leakysh3[2]]*2, fill=None, line_color='#FF2200',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[20,24], y=[leakysh3[0]]*2, fill='tonexty',line_color='#FF2200', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[20,24], y=[leakysh3[1]]*2, name='Average' ,fill=None,line=dict(color='#FF2200', width=2)))

fig.add_trace(go.Scatter(x=[25,29], y=[adapsh3[2]]*2, fill=None, line_color='red',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[25,29], y=[adapsh3[0]]*2, fill='tonexty',line_color='red', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[25,29], y=[adapsh3[1]]*2, name='Average' ,fill=None,line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=[30,34], y=[nop[2]]*2, fill=None, line_color='brown',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[30,34], y=[nop[0]]*2, fill='tonexty',line_color='brown', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=[30,34], y=[nop[1]]*2, name='Average' ,fill=None,line=dict(color='brown', width=2)))



fig.update_traces()

fig.update_layout(
     xaxis=dict(
          tickmode='array',
          tickvals=[2,7,12,17,22,27,32],
          ticktext=['Hyperband, full', 'Hyperband, leaky', 'Hyperband, adaptive','SH3, full','SH3, leaky','SH3, adaptive', 'No Pruning']
     )
)

fig.update_layout(yaxis_title='Loss / datapoint', title='Best Loss after 5000 Trials',
                  xaxis_title='Pruner, mode',
                  template='simple_white')

fig.update_layout(
    title=dict(font=dict(size=30)),
    yaxis_title=dict(font=dict(size=20)),
    xaxis_title=dict(font=dict(size=20)),
    title_x = 0.5
)

fig.show()
