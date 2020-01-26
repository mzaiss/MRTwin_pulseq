# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:02:42 2020

@author: mzaiss
"""

# formatting
#https://pyformat.info/
"f={:.2}*exp(-{:.2}*t)+{}".format(5.0, 7.1,'constant')


TypeError: can't assign a numpy.ndarray to a torch.FloatTensor
solution: torch.from_numpy(Trec)
vice-versa: tonumpy()


# np.flatten('F')
tonumpy(event_time).flatten('F')
# this is the same as 
np.transpose(tonumpy(event_time)).flatten()
# or
np.transpose(tonumpy(event_time)).ravel()
# or
np.ravel(tonumpy(event_time),order='F')


# plot signal as a function of total time:
plt.subplot(413); plt.ylabel('signal')
time_axis=np.cumsum(tonumpy(event_time).flatten('F'))
plt.plot(time_axis,tonumpy(scanner.signal[0,:,:,0,0]).flatten('F'),label='real')
plt.plot(time_axis,tonumpy(scanner.signal[0,:,:,1,0]).flatten('F'),label='imag')
ax=plt.gca(); ax.set_xticks(time_axis[major_ticks]); ax.grid()