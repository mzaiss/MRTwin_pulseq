# -*- coding: utf-8 -*-


np.save(r'C:\Users\danghi\Documents\MRzero\export\S_'+str(iternum),S)
np.save(r'C:\Users\danghi\Documents\MRzero\export\CNN_'+str(iternum),tonumpy(T1_map_CNN))
np.save(r'C:\Users\danghi\Documents\MRzero\export\FIT_'+str(iternum),T1_map)

sort_idx=np.argsort(np.abs(xdata))
xdata = xdata[sort_idx]

#csf
x,y = 17,15 #(R,C)
points = S[sort_idx,x,y]
plt.plot(np.abs(xdata),points,'x')

popt = quantify_T1(xdata, points, p0=[S[-1,x,y],1,-S[-1,x,y]])#, p0 = [np.flip(real_phantom_resized[:,:,1].transpose(),(0,1))[x,y],0.1,-0.1])
#popt = quantify_T1(np.abs(xdata), points, p0=None)
plt.plot(np.abs(xdata), points, '.b', marker='.')
plt.plot(np.abs(xdata), signal2_T1(np.abs(xdata), *popt), '-r', label='fit: S_0=%5.3f, T1=%5.3f, Z_i=%5.3f' % tuple(popt))
plt.legend()
plt.show()

np.save(r'C:\Users\danghi\Documents\MRzero\export\s_csf_'+str(iternum),points)
np.save(r'C:\Users\danghi\Documents\MRzero\export\fit_csf_'+str(iternum),signal2_T1(np.abs(xdata), *popt))

#white matter
x,y = 21,11 #(R,C)
points = S[sort_idx,x,y]
plt.plot(np.abs(xdata),points,'x')

popt = quantify_T1(xdata, points, p0=[S[-1,x,y],1,-S[-1,x,y]])#, p0 = [np.flip(real_phantom_resized[:,:,1].transpose(),(0,1))[x,y],0.1,-0.1])
#popt = quantify_T1(np.abs(xdata), points, p0=None)
plt.plot(np.abs(xdata), points, '.b', marker='.')
plt.plot(np.abs(xdata), signal2_T1(np.abs(xdata), *popt), '-r', label='fit: S_0=%5.3f, T1=%5.3f, Z_i=%5.3f' % tuple(popt))
plt.legend()
plt.show()

np.save(r'C:\Users\danghi\Documents\MRzero\export\s_white_'+str(iternum),points)
np.save(r'C:\Users\danghi\Documents\MRzero\export\fit_white_'+str(iternum),signal2_T1(np.abs(xdata), *popt))

#grey
x,y = 14,6 #(R,C)
points = S[sort_idx,x,y]
plt.plot(np.abs(xdata),points,'x')

popt = quantify_T1(xdata, points, p0=[S[-1,x,y],1,-S[-1,x,y]])#, p0 = [np.flip(real_phantom_resized[:,:,1].transpose(),(0,1))[x,y],0.1,-0.1])
#popt = quantify_T1(np.abs(xdata), points, p0=None)
plt.plot(np.abs(xdata), points, '.b', marker='.')
plt.plot(np.abs(xdata), signal2_T1(np.abs(xdata), *popt), '-r', label='fit: S_0=%5.3f, T1=%5.3f, Z_i=%5.3f' % tuple(popt))
plt.legend()
plt.show()

np.save(r'C:\Users\danghi\Documents\MRzero\export\s_grey_'+str(iternum),points)
np.save(r'C:\Users\danghi\Documents\MRzero\export\fit_grey_'+str(iternum),signal2_T1(np.abs(xdata), *popt))

