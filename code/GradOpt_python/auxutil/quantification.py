# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def tonumpy(x):
    return x.detach().cpu().numpy()

def signal_T1 (TI, S_0, T1):
    return np.abs(S_0 * (1 - 2*np.exp(-TI/T1)))

def signal2_T1 (TI, S_0, T1, Z_i):
    return np.abs(S_0 - (S_0 - Z_i)*np.exp(-TI/T1))

def signal3_T1 (TI, S_0, T1):
    return np.abs(S_0 * (1 - 2*np.exp(-TI/T1)+np.exp(-0.0152 / T1)))

def quantify_T1 (TI, S, p0):
    popt, pcov = curve_fit(signal2_T1, TI, S, p0 = p0, maxfev=1000000)#, bounds=([S[-1]/2,0.5,-S[-1]*2],[S[-1]*2,5,S[-1]]))
    return popt


T1_map = np.zeros((sz[0],sz[1]))
xdata = np.array([0.5,1,1.5,2,3,4,5,6,8,10])
xdata = np.array([0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1])
xdata = tonumpy(TI)
#S = np.array([mag_echo1, mag_echo2, mag_echo3])

#S = []
#for i in range(extraRep):
#    S.append(magimg(tonumpy(reco_all_rep[i,:,:]).reshape([sz[0],sz[1],2])))
#S = np.array(S)
reco_test = torch.sqrt((reco_testset**2).sum(2))
S = reco_test.reshape(10,32,32)
S = tonumpy(S)
mask = np.flip(real_phantom_resized[:,:,1].transpose(),(0,1)) != 1.e-12

for l in range(sz[0]):
    for m in range(sz[1]):
        try:
            popt = quantify_T1(xdata[0:], S[0:,l,m], p0=[S[-1,l,m],1,-S[-1,l,m]])
            #popt = quantify_T1(xdata[0:], S[0:,l,m], p0=None)
            T1_map[l,m] = popt[1]
        except:
            T1_map[l,m] = 0

T1_map = T1_map*mask

reco_test = torch.sqrt((reco_testset**2).sum(2))
reco_test = reco_test.permute([1,0])
T1_map_CNN = NN(reco_test).reshape([sz[0],sz[1]])
T1_map_CNN = T1_map_CNN*mask

GT = np.flip(real_phantom_resized[:,:,1].transpose(),(0,1))
residuals = (GT-T1_map).flatten()
ss_res = np.sum((residuals)**2)
ss_tot =  np.sum((GT-np.mean(GT))**2)

r_squared = 1 - (ss_res / ss_tot)

GT = np.flip(real_phantom_resized[:,:,1].transpose(),(0,1))
residuals = (GT-tonumpy(T1_map_CNN)*mask).flatten()
ss_res = np.sum((residuals)**2)
ss_tot =  np.sum((GT-np.mean(GT))**2)

r_squared = 1 - (ss_res / ss_tot)

#plt.plot(xdata, signal2_T1(xdata, *popt), 'r-', label='fit: S_0=%5.3f, T1=%5.3f' % tuple(popt))

plt.subplot(131)
plt.imshow(T1_map)
plt.clim(0,4)
plt.colorbar()
#plt.subplot(132)
#plt.imshow(T1map1)
#plt.clim(0,5)
#plt.colorbar()
plt.subplot(132)
#plt.imshow(tonumpy(cnn_output_real))
plt.imshow(np.flip(real_phantom_resized[:,:,1].transpose(),(0,1)))
plt.clim(0,5)
plt.colorbar()
plt.subplot(133)
#plt.imshow(tonumpy(cnn_output_real))
plt.imshow(tonumpy(T1_map_CNN))
plt.clim(0,4)
plt.colorbar()

plt.plot(T1_map.flatten(),np.flip(real_phantom_resized[:,:,1].transpose(),(0,1)).flatten(),'.')
plt.xlim((0,6))
plt.ylim((0,6))
plt.plot(np.arange(0,6))

plt.imshow()


plt.plot(mag_echo1.flatten())
plt.plot(mag_echo2.flatten())
plt.plot(mag_echo3.flatten())

x,y = 17,17 #(R,C)
start_idx = 0
points = S[start_idx:,x,y]
reduced_xdata = xdata[start_idx:]
popt = quantify_T1(reduced_xdata, points, p0=[S[-1,x,y],1,-S[-1,x,y]])#, p0 = [np.flip(real_phantom_resized[:,:,1].transpose(),(0,1))[x,y],0.1,-0.1])
popt = quantify_T1(reduced_xdata, points, p0=None)
plt.plot(reduced_xdata, points, 'b-', label='data, S_0_true = %5.3f, T1_true=%5.3f' % tuple([np.flip(real_phantom_resized[:,:,0].transpose(),(0,1))[x,y],np.flip(real_phantom_resized[:,:,1].transpose(),(0,1))[x,y]]), marker='.')
plt.plot(reduced_xdata, signal2_T1(reduced_xdata, *popt), 'r-', label='fit: S_0=%5.3f, T1=%5.3f, Z_i=%5.3f' % tuple(popt))
plt.legend()
plt.show()

plt.imshow(S[0])

for i in range(extraRep):
    plt.plot(S[i].flatten())
    
import plotly_express as px
from plotly.offline import plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = go.Figure()
for step in range(extraRep):
    fig.add_trace(go.Heatmap(z=S[step,:,:],visible=False,colorscale="haline"))
    #fig.add_trace(px.imshow(S[step]),row=1, col=1)
fig.data[0].visible = True

steps = []
for i in range(10):
    step = dict(
        method="restyle",
        args=["visible", [False] * 10],
    )
    step["args"][1][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)
    
sliders = [dict(
    active=0,
    currentvalue={"prefix": "Slide: "},
    pad={"t": 0},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

plot(fig)

points = [x[15,15] for x in S]
fig.add_trace(go.Scatter(y=points),row=1, col=2)

fig.show()

fig = make_subplots(rows=1, cols=2)
#figure = px.imshow(GT)
#trace1 = figure['data'][0]
#fig.add_trace(trace1, row=1, col=1)
fig.add_trace(go.Heatmap(z=GT,colorscale="haline"),row=1, col=1)
trace = fig['data'][0]


def update_point(trace, points, selector):
    with fig.batch_update():
        fig.add_trace(go.Heatmap(z=GT,colorscale="haline"),row=1, col=2)


trace.on_click(update_point)

plot(fig)
plt.imshow(GT)

import plotly.io as pio

pio.renderers.default ='png'
figure = px.imshow(GT)
figure.show()

import plotly.graph_objects as go

import numpy as np
np.random.seed(1)

x = np.random.rand(100)
y = np.random.rand(100)

f = go.FigureWidget([go.Scatter(x=x, y=y, mode='markers')])

scatter = f.data[0]
colors = ['#a3a7e4'] * 100
scatter.marker.color = colors
scatter.marker.size = [10] * 100
f.layout.hovermode = 'closest'


# create our callback function
def update_point(trace, points, selector):
    c = list(scatter.marker.color)
    s = list(scatter.marker.size)
    for i in points.point_inds:
        c[i] = '#bae2be'
        s[i] = 20
        with f.batch_update():
            scatter.marker.color = c
            scatter.marker.size = s


scatter.on_click(update_point)
plot(f)

fig = make_subplots(1, 2)
fig.add_trace(go.Image(z=S[0,:,:]),1,1)
plot(fig)


plt.subplot(341)
plt.imshow(S[0,:,:])
plt.colorbar()
plt.subplot(342)
plt.imshow(S[1,:,:])
plt.colorbar()
plt.subplot(343)
plt.imshow(S[2,:,:])
plt.colorbar()
plt.subplot(344)
plt.imshow(S[3,:,:])
plt.colorbar()
plt.subplot(345)
plt.imshow(S[4,:,:])
plt.colorbar()
plt.subplot(346)
plt.imshow(S[5,:,:])
plt.colorbar()
plt.subplot(347)
plt.imshow(S[6,:,:])
plt.colorbar()
plt.subplot(348)
plt.imshow(S[7,:,:])
plt.colorbar()
plt.subplot(349)
plt.imshow(S[8,:,:])
plt.colorbar()
plt.subplot(3410)
plt.imshow(S[9,:,:])
plt.colorbar()

plt.subplot(341)
plt.imshow(S[0,:,:])
plt.clim(0,1)
plt.subplot(342)
plt.imshow(S[1,:,:])
plt.clim(0,1)
plt.subplot(343)
plt.imshow(S[2,:,:])
plt.clim(0,1)
plt.subplot(344)
plt.imshow(S[3,:,:])
plt.clim(0,1)
plt.subplot(345)
plt.imshow(S[4,:,:])
plt.clim(0,1)
plt.subplot(346)
plt.imshow(S[5,:,:])
plt.clim(0,1)
plt.subplot(347)
plt.imshow(S[6,:,:])
plt.clim(0,1)
plt.subplot(348)
plt.imshow(S[7,:,:])
plt.clim(0,1)
plt.subplot(349)
plt.imshow(S[8,:,:])
plt.clim(0,1)
plt.subplot(3410)
plt.imshow(S[9,:,:])
plt.clim(0,1)

reco_testset_complex = tonumpy(reco_testset[:,:,0])+1j*tonumpy(reco_testset[:,:,1])
phase = np.angle(reco_testset_complex)

plt.subplot(341)
plt.imshow(phase[0,:].reshape(32,32))
plt.colorbar()
plt.subplot(342)
plt.imshow(phase[1,:].reshape(32,32))
plt.colorbar()
plt.subplot(343)
plt.imshow(phase[2,:].reshape(32,32))
plt.colorbar()
plt.subplot(344)
plt.imshow(phase[3,:].reshape(32,32))
plt.colorbar()
plt.subplot(345)
plt.imshow(phase[4,:].reshape(32,32))
plt.colorbar()
plt.subplot(346)
plt.imshow(phase[5,:].reshape(32,32))
plt.colorbar()
plt.subplot(347)
plt.imshow(phase[6,:].reshape(32,32))
plt.colorbar()
plt.subplot(348)
plt.imshow(phase[7,:].reshape(32,32))
plt.colorbar()
plt.subplot(349)
plt.imshow(phase[8,:].reshape(32,32))
plt.colorbar()
plt.subplot(3410)
plt.imshow(phase[9,:].reshape(32,32))
plt.colorbar()