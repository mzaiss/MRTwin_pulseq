from math import ceil, sqrt, pow

from pypulseq.holder import Holder
from pypulseq.opts import Opts

def make_trapezoid(kwargs):
    """
    Makes a Holder object for an trapezoidal gradient Event.

    Parameters
    ----------
    kwargs : dict
        Key value mappings of trapezoidal gradient Event parameters_params and values.

    Returns
    -------
    grad : Holder
        Trapezoidal gradient Event configured based on supplied kwargs.
    """

    channel = kwargs.get("channel", "z")
    system = kwargs.get("system", Opts())
    duration = kwargs.get("duration", 0)
    area_result = kwargs.get("area", -1)
    flat_time_result = kwargs.get("flat_time", 0)
    flat_area_result = kwargs.get("flat_area", -1)
    amplitude_result = kwargs.get("amplitude", -1)
    max_grad = kwargs.get("max_grad", 0)
    max_slew = kwargs.get("max_slew", 0)
    rise_time = kwargs.get("rise_time", 0)
    delay = kwargs.get("delay", 0)

    max_grad = max_grad if max_grad > 0 else system.max_grad
    max_slew = max_slew if max_slew > 0 else system.max_slew
    rise_time = rise_time if rise_time > 0 else system.rise_time

    if area_result == -1 and flat_area_result == -1 and amplitude_result == -1:
        raise ValueError('Must supply either ''area'', ''flat_area'' or ''amplitude''')

    if flat_time_result > 0:
        amplitude = amplitude_result if (amplitude_result != -1) else (flat_area_result / flat_time_result)
        if rise_time == 0:
            rise_time = abs(amplitude) / max_slew
            rise_time = ceil(rise_time / system.grad_raster_time) * system.grad_raster_time
        fall_time, flat_time = rise_time, flat_time_result
    elif duration > 0:
        if amplitude_result != -1:
            amplitude = amplitude_result
        else:
            if rise_time == 0:
                dC = 1 / abs(2 * max_slew) + 1 / abs(2 * max_slew)
                amplitude = (duration - sqrt(pow(duration, 2) - 4 * abs(area_result) * dC)) / (2 * dC)
                possible = duration**2 > 4*abs(area_result)*dC
            else:
                amplitude = area_result / (duration - rise_time)
                possible = duration>2*rise_time and abs(amplitude)<max_grad;
                
            if not possible:
                raise ValueError('Requested area is too large for this gradient.')

        if rise_time == 0:
            rise_time = ceil(
                abs(amplitude) / max_slew / system.grad_raster_time) * system.grad_raster_time

        fall_time = rise_time
        flat_time = (duration - rise_time - fall_time)

        amplitude = area_result / (rise_time / 2 + fall_time / 2 + flat_time) if amplitude_result == -1 else amplitude

    else:
        if area_result == -1:
            raise ValueError("makeTrapezoid:invalidArguments','Must supply area at least")
        else:
            #
            # find the shortest possible duration
            # first check if the area can be realized as a triangle
            # if not we calculate a trapezoid
            rise_time=ceil(sqrt(abs(area_result)/max_slew)/system.grad_raster_time)*system.grad_raster_time
            amplitude=area_result/rise_time
            
            if abs(rise_time) < 1e-12:
                raise ValueError("rise_time = 0")
            
            tEff=rise_time
            if abs(amplitude)>max_grad:
                tEff=ceil((abs(area_result)/max_grad)/system.grad_raster_time)*system.grad_raster_time
                amplitude=area_result/tEff
                rise_time=ceil((abs(amplitude)/max_slew)/system.grad_raster_time)*system.grad_raster_time
                
            flat_time=tEff-rise_time
            fall_time=rise_time        

    if abs(amplitude) > max_grad:
        raise ValueError("Amplitude violation")

    grad = Holder()
    grad.type = "trap"
    grad.channel = channel
    grad.amplitude = amplitude
    grad.rise_time = rise_time
    grad.flat_time = flat_time
    grad.fall_time = fall_time
    grad.area = amplitude * (flat_time + rise_time / 2 + fall_time / 2)
    grad.flat_area = amplitude * flat_time
    grad.delay = delay

    return grad    

#def make_trapezoid(kwargs):
#    """
#    Makes a Holder object for an trapezoidal gradient Event.
#
#    Parameters
#    ----------
#    kwargs : dict
#        Key value mappings of trapezoidal gradient Event parameters_params and values.
#
#    Returns
#    -------
#    grad : Holder
#        Trapezoidal gradient Event configured based on supplied kwargs.
#    """
#
#    channel = kwargs.get("channel", "z")
#    system = kwargs.get("system", Opts())
#    duration = kwargs.get("duration", 0)
#    area_result = kwargs.get("area", -1)
#    flat_time_result = kwargs.get("flat_time", 0)
#    flat_area_result = kwargs.get("flat_area", -1)
#    amplitude_result = kwargs.get("amplitude", -1)
#    max_grad = kwargs.get("max_grad", 0)
#    max_slew = kwargs.get("max_slew", 0)
#    rise_time = kwargs.get("rise_time", 0)
#    delay = kwargs.get("delay", 0)
#
#    max_grad = max_grad if max_grad > 0 else system.max_grad
#    max_slew = max_slew if max_slew > 0 else system.max_slew
#    rise_time = rise_time if rise_time > 0 else system.rise_time
#
#    if area_result == -1 and flat_area_result == -1 and amplitude_result == -1:
#        raise ValueError('Must supply either ''area'', ''flat_area'' or ''amplitude''')
#
#    if flat_time_result > 0:
#        amplitude = amplitude_result if (amplitude_result != -1) else (flat_area_result / flat_time_result)
#        if rise_time == 0:
#            rise_time = abs(amplitude) / max_slew
#            rise_time = ceil(rise_time / system.grad_raster_time) * system.grad_raster_time
#        fall_time, flat_time = rise_time, flat_time_result
#    elif duration > 0:
#        if amplitude_result != -1:
#            amplitude = amplitude_result
#        else:
#            if rise_time == 0:
#                dC = 1 / abs(2 * max_slew) + 1 / abs(2 * max_slew)
#                amplitude = (duration - sqrt(pow(duration, 2) - 4 * abs(area_result) * dC)) / (2 * dC)
#            else:
#                amplitude = area_result / (duration - rise_time)
#
#        if rise_time == 0:
#            rise_time = ceil(
#                amplitude / max_slew / system.grad_raster_time) * system.grad_raster_time
#
#        fall_time = rise_time
#        flat_time = (duration - rise_time - fall_time)
#
#        amplitude = area_result / (rise_time / 2 + fall_time / 2 + flat_time) if amplitude_result == -1 else amplitude
#    else:
#        if area_result == -1:
#            raise ValueError("makeTrapezoid:invalidArguments','Must supply area at least")
#        else:
#            #
#            # find the shortest possible duration
#            # first check if the area can be realized as a triangle
#            # if not we calculate a trapezoid
#            rise_time=ceil(sqrt(abs(area_result)/max_slew)/system.grad_raster_time)*system.grad_raster_time
#            amplitude=area_result/rise_time
#            tEff=rise_time
#            if abs(amplitude)>max_grad:
#                tEff=ceil((abs(area_result)/max_grad)/system.grad_raster_time)*system.grad_raster_time
#                amplitude=area_result/tEff
#                rise_time=ceil((abs(amplitude)/max_slew)/system.grad_raster_time)*system.grad_raster_time
#                
#            flat_time=tEff-rise_time
#            fall_time=rise_time        
#
#    if abs(amplitude) > max_grad:
#        raise ValueError("Amplitude violation")
#
#    grad = Holder()
#    grad.type = "trap"
#    grad.channel = channel
#    grad.amplitude = amplitude
#    grad.rise_time = rise_time
#    grad.flat_time = flat_time
#    grad.fall_time = fall_time
#    grad.area = amplitude * (flat_time + rise_time / 2 + fall_time / 2)
#    grad.flat_area = amplitude * flat_time
#    grad.delay = delay
#
#    return grad    
