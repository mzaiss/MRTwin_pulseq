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