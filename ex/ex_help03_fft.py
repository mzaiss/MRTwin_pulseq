from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

samplepoints = 112
func = np.zeros((samplepoints,),dtype=np.complex128)
func[20:70] = 1+0.1j

# func = signal.gaussian(samplepoints, std=5)
# func = (np.sin(0.1 * np.arange(1, samplepoints, 1)) +
#         0.25 * np.sin(0.5 * np.arange(1, samplepoints, 1)))
# func = np.roll(func, samplepoints // 2 + 11, axis=0)

FFT_func = np.fft.ifft(func)
FFT_FFT_func = np.fft.fft(FFT_func)

plt.subplot(321)
plt.title('func')
plt.plot(np.real(func))
plt.plot(np.imag(func)); plt.legend(['real','imag'])

plt.subplot(323)
plt.title('ifft(func)')
plt.plot(np.real(FFT_func))
plt.plot(np.imag(FFT_func)); plt.legend(['real','imag'])
plt.subplot(325)
plt.title('fft(ifft(func))')
plt.plot(np.abs(FFT_FFT_func))
plt.plot(np.imag(FFT_FFT_func)); plt.legend(['real','imag'])



## echo like function
func_echolike = np.fft.fftshift(FFT_func, 0) #an echo is the shifted ifft of an object

plt.subplot(322)
plt.title('echo_func \n(an echo is the shifted ifft of an object) ')
plt.plot(np.abs(func_echolike))
plt.plot(np.imag(func_echolike)); plt.legend(['real','imag'])


# fft of wrongly fft-shifted
FFT_func_echolike = np.fft.fft(func_echolike)

plt.subplot(324)
plt.title('FFT(echo)')
plt.plot(np.abs(FFT_func_echolike))
plt.plot(np.imag(FFT_func_echolike)); plt.legend(['real','imag'])

# fft of rolled fft-shifted

FFT_func_echolike_shifted = np.fft.fft(np.fft.ifftshift(func_echolike, 0))

plt.subplot(326)
plt.title('FFT(FFTshift(echo)')
plt.plot(np.abs(FFT_func_echolike_shifted))
plt.plot(np.imag(FFT_func_echolike_shifted)); plt.legend(['real','imag'])
