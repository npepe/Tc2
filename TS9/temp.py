# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 00:01:33 2022
@author: Deste
"""

# Inicialización e importación de módulos

# Módulos para Jupyter
import warnings
warnings.filterwarnings('ignore')

# Módulos importantantes
import scipy.signal as sig
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
from splane import plot_plantilla
from pylab import *

def impz(b,a=1):
    l = len(b)
    impulse = repeat(0.,l); impulse[0] =1.
    x = arange(0,l)
    response = sig.lfilter(b,a,impulse)
    subplot(211)
    stem(x, response)
    ylabel('Amplitude')
    xlabel(r'n (samples)')
    title(r'Impulse response')




fig_sz_x = 10
fig_sz_y = 7
fig_dpi = 100 # dpi

fig_font_size = 16

mpl.rcParams['figure.figsize'] = (fig_sz_x,fig_sz_y)
plt.rcParams.update({'font.size':fig_font_size})

###
## Señal de ECG registrada a 1 kHz, con contaminación de diversos orígenes.
###

# para listar las variables que hay en el archivo
#io.whosmat('ecg.mat')
mat_struct = sio.loadmat('ecg.mat')

ecg_one_lead = mat_struct['ecg_lead']
ecg_one_lead = ecg_one_lead.flatten()
cant_muestras = len(ecg_one_lead)

fs = 1000 # Hz
nyq_frec = fs / 2
# Plantilla

# filter design
ripple = 0 # dB
atenuacion = 40 # dB

ws1 = 1.0 #Hz
wp1 = 3.0 #Hz
wp2 = 25.0 #Hz
ws2 = 35.0 #Hz

frecs = np.array([0.0,         ws1,         wp1,     wp2,     ws2,         nyq_frec   ]) / nyq_frec
gains = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])
gains = 10**(gains/20)
# Design IIR



iir_sos_cauer= sig.iirdesign(wp=np.array([wp1-1, wp2+1]) / nyq_frec, ws=np.array([ws1, ws2]) / nyq_frec, gpass=0.01, gstop=40., analog=False, ftype='ellip', output='sos')

w,h=sig.sosfreqz(iir_sos_cauer)
a,b=sig.sos2tf(iir_sos_cauer)
# renormalizo el eje de frecuencia
w = w / np.pi * nyq_frec
plt.figure(1)
print(iir_sos_cauer.shape)
plt.plot(w, 20 * np.log10(abs(h)), label='IIR_cauer')
plt.title('Filtros diseñados')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo [dB]')
plt.grid()
plt.axis([0, 100, -60, 5 ]);
axes_hdl = plt.gca()
axes_hdl.legend()
plot_plantilla(filter_type = 'bandpass', fpass = frecs[[2, 3]]* nyq_frec, ripple = ripple , fstop = frecs[ [1, 4] ]* nyq_frec, attenuation = atenuacion, fs = fs)

plt.figure(2)
h_Phase = np.unwrap(pylab.arctan2(imag(h),real(h)))
plt.plot(w,h_Phase)
plt.ylabel('Phase (radians)')
plt.xlabel('Frecuencia [Hz]')
plt.title(r'Phase response')

plt.figure(3)
# gd = -np.diff(np.angle(h))/np.diff(h)
# n_gd = np.arange(len(gd))
# plt.plot(n_gd,gd)
w_gd,gd = sig.group_delay((b,a))
plt.plot(2*pi*w_gd,gd)
plt.ylabel('Group Delay [Samples]')
plt.xlabel('Frecuencia [rad/sample]')
plt.title(r'Phase response')


plt.figure(4)
impz(b,a)
