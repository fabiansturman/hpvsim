#can i take my lineplit-copy file and do little line segment plots and then just combine them? perhaps do fig=go.Figure();fig.add_trace(theplot) for each plot i want to add??


import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import scipy

## DATASET 1

fig=go.Figure()

# Adding min/max bars to plot
hyperband_minmax = [0.127,0.154]
median_minmax = [0.134,0.155]
shrf2_minmax = [0.124,0.154 ]
shrf3_minmax = [0.127,0.146]
shrf4_minmax = [0.127,0.148]
nop_minmax =  [0.134, 0.155]

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
no_datapoints = 31
hyperband = [4.269/no_datapoints,4.447/no_datapoints,4.625/no_datapoints]
median = [4.35/no_datapoints,4.497/no_datapoints,4.65/no_datapoints]
shrf2 = [4.26/no_datapoints,4.5/no_datapoints,4.73/no_datapoints]
shrf3 = [4.26/no_datapoints,4.37/no_datapoints,4.48/no_datapoints]
shrf4 = [4.11/no_datapoints,4.23/no_datapoints,4.35/no_datapoints]
nop = [4.39/no_datapoints,4.52/no_datapoints,4.65/no_datapoints]

fig.add_hline(y=nop[0], line_width=1.5, line_dash='dash')
fig.add_hline(y=nop[2], line_width=1.5, line_dash='dash')


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

fig.update_layout(yaxis_title='Best Loss', title='Calibration with dataset 1: loss after 5000 trials',
                  template='simple_white')


fig.show()

## DATASET 2

fig=go.Figure()

# Adding min/max bars to plot
hyperband_minmax = [0.138, 0.158]
median_minmax = [0.136, 0.154]
shrf2_minmax = [0.134, 0.159]
shrf3_minmax = [0.135, 0.16]
shrf4_minmax = [0.133, 0.165]
nop_minmax =  [0.135, 0.158]

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
no_datapoints = 42
hyperband = [6.088/no_datapoints,6.272/no_datapoints,6.456/no_datapoints]
median = [5.94/no_datapoints,6.133/no_datapoints,6.32/no_datapoints]
shrf2 = [6.02/no_datapoints,6.304/no_datapoints,6.59/no_datapoints]
shrf3 = [6.05/no_datapoints,6.25/no_datapoints,6.45/no_datapoints]
shrf4 = [5.93/no_datapoints,6.184/no_datapoints,6.44/no_datapoints]
nop = [6.07/no_datapoints,6.264/no_datapoints,6.46/no_datapoints]

fig.add_hline(y=nop[0], line_width=1.5, line_dash='dash')
fig.add_hline(y=nop[2], line_width=1.5, line_dash='dash')

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

fig.update_layout(yaxis_title='Best Loss', title='Calibration with dataset 2: loss after 5000 trials',
                  template='simple_white')


fig.show()

## DATASET 3
fig=go.Figure()

# Adding min/max bars to plot
hyperband_minmax = [0.132, 0.257]
median_minmax = [0.129, 0.160]
shrf2_minmax = [0.132, 0.143]
shrf3_minmax = [0.125, 0.275]
shrf4_minmax = [0.142, 0.215]
nop_minmax =  [0.130, 0.148]

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
no_datapoints = 42
hyperband = [5.79/no_datapoints,6.92/no_datapoints,8.05/no_datapoints]
median = [5.59/no_datapoints,5.84/no_datapoints,6.09/no_datapoints]
shrf2 = [5.53/no_datapoints,5.66/no_datapoints,5.78/no_datapoints]
shrf3 = [6.58/no_datapoints,8.043/no_datapoints,9.51/no_datapoints]
shrf4 = [6.79/no_datapoints,7.313/no_datapoints,7.84/no_datapoints]
nop = [5.623/no_datapoints,5.781/no_datapoints,5.94/no_datapoints]

fig.add_hline(y=nop[0], line_width=1.5, line_dash='dash')
fig.add_hline(y=nop[2], line_width=1.5, line_dash='dash')

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

fig.update_layout(yaxis_title='Best Loss', title='Calibration with dataset 3: loss after 5000 trials',
                  template='simple_white')


fig.show()


## DATASET 4
fig=go.Figure()

# Adding min/max bars to plot
hyperband_minmax = [0.160,0.284]
median_minmax = [0.131, 0.168]
shrf2_minmax = [0.150, 0.176]
shrf3_minmax = [0.158, 0.251]
shrf4_minmax = [0.158, 0.284]
nop_minmax =  [0.143, 0.169]

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
no_datapoints = 30
hyperband = [4.65/no_datapoints,5.334/no_datapoints,6.01/no_datapoints]
median = [4.42/no_datapoints,4.635/no_datapoints,4.85/no_datapoints]
shrf2 = [4.76/no_datapoints,4.936/no_datapoints,5.11/no_datapoints]
shrf3 = [5.21/no_datapoints,5.785/no_datapoints,6.36/no_datapoints]
shrf4 = [5.18/no_datapoints,5.887/no_datapoints,6.6/no_datapoints]
nop = [4.49/no_datapoints,4.655/no_datapoints,4.82/no_datapoints]

fig.add_hline(y=nop[0], line_width=1.5, line_dash='dash')
fig.add_hline(y=nop[2], line_width=1.5, line_dash='dash')

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

fig.update_layout(yaxis_title='Best Loss', title='Calibration with dataset 4: loss after 5000 trials',
                  template='simple_white')


fig.show()