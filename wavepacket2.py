# wave_packet.py - propagates a wave packet with a dispersion relation omega(k)
# by Bjoern Malte Schaefer, GSFP+/Heidelberg, bjoern.malte.schaefer@uni-heidelberg.de

#Modifikation für Aufgabe 2 Experimentalpysik 3 Ws14/15 von Gerrit Anders und Christiane Klein
#DIe unterschiedlichen Graphen sind auszukommentieren und nach einander zu plotten

# import libraries numpy and matplotlib
import numpy as np
import pylab as plt
from numpy import *

plt.close()

# set width sigma and the number of pixels ngrid
sigma = 0.1	    # width of the wave packet
ngrid = 1024	# number of pixels
velocity = 20	# group velocity
t = 10 		    # time 

# definition of the Gaussian envelope
def psi_gauss(x,sigma):
	aux = x / sigma
	result = np.exp(-aux**2 / 2.0)
	return(result)
	
# definition of the tophat envelope
def psi_tophat(x,sigma):
	return(np.asarray(np.abs(x/sigma)<1.0,dtype=float))
	
# dispersion relation omega(k)
def omega(k):
	result = velocity * k
	return(result)

# set up wave packet in real space and plot at t=0
x = np.linspace(-1.0,1.0,ngrid)
psi_initial = psi_gauss(x,sigma)
#Plot 0a: Wellenpaktet
#plt.plot(x,psi_initial,'r--',label='initial wave packet at 0')

# Fourier transform
psi_fourier = np.fft.fft(psi_initial)

# multiply with k-dependent phase factor
k = np.fft.fftfreq(ngrid)
phase = np.exp(-1j * omega(k) * t)
psi_fourier *= phase

# Fourier transform back to real space
psi_final = np.fft.ifft(psi_fourier)

# plot wavepacket at t
#Plot 0b: Wellenpaket verschoben
#plt.plot(x,psi_final,'g-',labelh='final wave packet at t')

#Plots der Aufgabenlösungen

#Plot 1: Spektrum Gaussimpuls
well = np.exp(-1j*(k*x-omega(k)*t))
specG = np.abs(psi_initial * well)**2
#plt.plot(x,specG,'g-',label='Spektrum Gaussimpuls')

#Plot 2: Spektrum Rechteckeck
psi_reck = psi_tophat(x,sigma)
specR= np.abs(psi_reck*well)**2
#plt.plot(x,specR,'g-', label='Spektrum Rechteckimpuls')


#Plot 3: Fouriertransformation mit psi(k)*exp(-iw(k)t) fuer Gaussimpuls
psiG = psi_initial * phase
#plt.plot(x,psiG,'g-',label='Fouriertrafo Gauss Envelope')

#Plot 4: Fouriertransformation fuer Rechteckimpuls
psiR = psi_reck * phase
#plt.plot(x,psiR, 'g-',label='Fouriertrafo Rechteck Envelope')


#Plot 5: Ruecktrafo Gauss
psiGr = np.fft.ifft(psiG)
#plt.plot(x,psiGr,'g-',label='Ruecktrafo Gauss')

#Plot 6: Ruecktrafo Rechteck
psiRr = np.fft.ifft(psiR)
#plt.plot(x,psiRr,'g-',label='Ruecktrafo Rechteck')

# add nice things
plt.legend(loc='upper left')
plt.xlabel('$x$-axis')
plt.ylabel('$\psi$-axis')
plt.ylim([-0.2,2.0])
plt.show()




