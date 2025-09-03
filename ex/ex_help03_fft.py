import scipy
import torch 
import matplotlib.pyplot as plt
case="gauss"
if case=="block":
    samplepoints = 112
    func = torch.zeros((samplepoints,),dtype=torch.complex128)
    func[20:70] = 1+0.1j
elif case=="gauss":
    func = scipy.signal.gaussian(samplepoints, std=5)*(1+0.1j)

func = torch.tensor(func,dtype=torch.complex128)

FFT_func = torch.fft.ifft(func)
FFT_FFT_func = torch.fft.fft(FFT_func)

plt.subplot(321)
plt.title('func')
plt.plot(torch.real(func))
plt.plot(torch.imag(func)); plt.legend(['real','imag'])

plt.subplot(323)
plt.title('ifft(func)')
plt.plot(torch.real(FFT_func))
plt.plot(torch.imag(FFT_func)); plt.legend(['real','imag'])
plt.subplot(325)
plt.title('fft(ifft(func))')
plt.plot(torch.abs(FFT_FFT_func))
plt.plot(torch.imag(FFT_FFT_func)); plt.legend(['real','imag'])



## echo like function
func_echolike = torch.fft.fftshift(FFT_func, axis=0) #an echo is the shifted ifft of an object

plt.subplot(322)
plt.title('echo_func \n(an echo is the shifted ifft of an object) ')
plt.plot(torch.abs(func_echolike))
plt.plot(torch.imag(func_echolike)); plt.legend(['real','imag'])


# fft of wrongly fft-shifted
FFT_func_echolike = torch.fft.fft(func_echolike, axis=0)

plt.subplot(324)
plt.title('FFT(echo)')
plt.plot(torch.abs(FFT_func_echolike))
plt.plot(torch.imag(FFT_func_echolike)); plt.legend(['real','imag'])

# fft of rolled fft-shifted

FFT_func_echolike_shifted = torch.fft.fft(torch.fft.ifftshift(func_echolike, axis=0))

plt.subplot(326)
plt.title('FFT(FFTshift(echo)')
plt.plot(torch.abs(FFT_func_echolike_shifted))
plt.plot(torch.imag(FFT_func_echolike_shifted)); plt.legend(['real','imag'])
