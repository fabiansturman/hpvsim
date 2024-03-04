import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

#This code demonstrates how I can neatly present the times it takes to calirate over several datasets with and without pruning

durations1_p=[1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9]
durations2_p=[11,11,12,12,13,13,14,14,15,16,15,16,17,17,18,18,19,19]
durations1_np=[1,2,3,4,5]
durations2_np=[11,11,12,12,13]


dic = {'Time': durations1_p + durations2_p + durations1_np + durations2_np,
       'Dataset': ["ds1"]*len(durations1_p) + ['ds2']* len(durations2_p) + ['ds1']* len(durations1_np) + ['ds2']* len(durations2_np),
       'Pruning': ['Yes']*len(durations1_p+durations2_p) + ['No']*len(durations1_np+durations2_np)}


df = pd.DataFrame(dic)
print(df)
fig = px.box(df,x="Dataset", y="Time", color="Pruning")
fig.show()


#Below is an example of a line graph (https://plotly.com/python/line-charts/ tells us how to it

xs = [1,2,3,4,5]
ys = [2,4,6,8,10]
ys_up = [3,5,7,9,11]
ys_down = [1,3,5,7,9]


dic2 = {'trial #': xs * 3, 
        'best cost':ys + ys_up + ys_down,
        'legend':['mean']*len(ys) + ['upper']*len(ys_up) + ['lower']*len(ys_down)}
df2 = pd.DataFrame(dic2)
fig=px.line(df2, x='trial #', y='best cost', color='legend')
fig.show()


fig = go.Figure()
fig.add_trace(go.Scatter(x=xs, y=ys_up, fill=None, line_color='green',mode='lines',line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=xs, y=ys_down, fill='tonexty',line_color='green', mode='lines', line_width=0,showlegend=False))
fig.add_trace(go.Scatter(x=xs, y=ys, fill=None,line=dict(color='black', width=2)))
fig.update_layout(xaxis_title="trial #", yaxis_title='loss', title='title here!')
fig.show()