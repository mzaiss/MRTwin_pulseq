spectrum=np.zeros(128)
spectrum[64:70]=1
#spectrum = np.roll(spectrum,szread//2+1,axis=0)
plt.plot(np.abs(spectrum))

space = np.fft.fft(spectrum)
plt.plot(np.real(space))

space= space[::2]
plt.plot(np.real(space))
spec2 = np.fft.ifft(space)

plt.plot(np.abs(spec2),'x')
# fftshift
#    space = roll(space,szread//2-1,0)
#    space = roll(space,NRep//2-1,1)
#space= np.roll(space,szread//2-1,axis=0)
plt.subplot(312)
plt.plot(20*np.abs(np.transpose(space).ravel()))
ax=plt.gca(); ax.set_xticks(major_ticks); ax.grid()