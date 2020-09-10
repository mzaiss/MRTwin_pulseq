from scipy import signal

samplepoints=112
func=np.zeros((samplepoints,))
func[40:70]=1

#func = signal.gaussian(samplepoints, std=5)
#func = np.sin(0.1*np.arange(1,samplepoints,1)) + 0.25*np.sin(0.5*np.arange(1,samplepoints,1))

#func = np.roll(func,samplepoints//2+11,axis=0)

FFT_func = np.fft.ifft(func)
FFT_FFT_func = np.fft.fft(FFT_func)

plt.subplot(321); plt.title('func')
plt.plot(np.real(func))

plt.subplot(323); plt.title('ifft(func)')
plt.plot(np.abs(FFT_func))

plt.subplot(325); plt.title('fft(ifft(func))')
plt.plot(np.abs(FFT_FFT_func))
plt.plot(np.imag(FFT_FFT_func))
# fftshift
func_echolike = np.roll(FFT_func,samplepoints//2,0)

plt.subplot(322); plt.title('func_echolike')
plt.plot(np.abs(func_echolike))
plt.plot(np.imag(func_echolike))


# fft of wrongly fft-shifted
FFT_func_echolike = np.fft.fft(func_echolike)

plt.subplot(324); plt.title('FFT_echo')
plt.plot(np.abs(FFT_func_echolike))
plt.plot(np.imag(FFT_func_echolike))

# fft of rolled fft-shifted

FFT_func_echolike_rolled = np.fft.fft(np.roll(func_echolike,samplepoints//2,0))

plt.subplot(326); plt.title('FFT_echo_rolled')
plt.plot(np.abs(FFT_func_echolike_rolled))
plt.plot(np.imag(FFT_func_echolike_rolled))