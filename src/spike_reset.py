#!/usr/bin/env python
#encoding: utf-8
import numpy as np
from pylab import *

dt=0.01 # msec
tau=40.0 # msec
tmax=1000 # msec
V_spk=-20
V_thres=-50.0
V_reset=-70.0
E_leak=V_reset
R_m=10.0 # MΩ

tt=np.arange(0, tmax, dt) #0:dt:tmax
Nt=len(tt) #length(tt)
V=np.zeros((Nt,))
V2=np.zeros((Nt,))
S=np.zeros((Nt,))
S2=np.zeros((Nt,))
#I0=np.zeros((Nt,))

# Plot characteristics
Vlim=E_leak-10,V_spk+10
# tlim=0,1000 #msec
tlim=200,800 #msec
nrows=4
LW=2
colors=[]
cmap = cm.hsv

# Solved Dayan & Abbott (2001) Ch.5 Eq. 5.12 for I_e using r_isi = 7 Hz:
theta_freq = 7
def I_e(f):
    tau_isi = 1000.0/f
    return -(1/R_m) * (E_leak + (V_reset - V_thres*exp(tau_isi/tau))/(exp(tau_isi/tau) - 1))
    
I_const=I_e(theta_freq) # 2.0578580 # 2.1 # constant current 
print 'I_const = %.4f nA'%I_const

Dt=25 # msec: STDP half window
n=int(Dt/dt)
hPlus=1.0*I_const # max height
hMinus=2.0*hPlus
dI=np.r_[np.linspace(0,hPlus,n),0,np.linspace(-hMinus,0,n)]

## first simulation
V[0]=V_reset

for i in xrange(1, Nt): #=2:Nt
    V[i]=((tau-dt)/tau)*V[i-1]+(dt/tau)*(E_leak+R_m*I_const)
    if V[i]>=V_thres:
        V[i]=V_reset
        S[i]=1

k=np.nonzero(S>0)[0]
Nspk=len(k)

ioff()


figure(1, figsize=(10.0, 14.7625))
clf()
subplot(nrows,1,1)
plot(tt,V,'k-',lw=LW)
# hold(True)
# plot([[k*dt,k*dt]*Nspk,[V_reset,V_spk],'b-',lw=LW)
title('control')
xlim(tlim)
ylim(Vlim)

## second simulation
T=(k[2]-k[1])*dt # period

Nsuper=5 # number of super-cycle for testing different timing

timeList=np.linspace((-T/2), T/2,Nsuper)
phaseList=np.zeros((Nsuper,))
plot_spikes =True

for i_super in xrange(Nsuper): #=1:Nsuper

    k0=k[2]+int(timeList[i_super]/dt)
    I=np.zeros((Nt,))
    I[k0-n:k0+n+1]=dI

    V2[0]=V_reset
    S2=np.zeros((Nt,))

    for i in xrange(1, Nt): #=2:Nt
        V2[i]=((tau-dt)/tau)*V2[i-1]+(dt/tau)*(E_leak+R_m*(I_const+I[i]))
        if V2[i]>=V_thres:
            V2[i]=V_reset
            S2[i]=1

    k2=np.nonzero(S2>0)[0]
    Nspk2=len(k2)

    subplot(nrows,1,2)
    color = cmap(i_super/float(Nsuper))
    colors.append(color)
    plot(tt,V2,'-',zorder=-Nsuper+i_super,lw=LW,c=color)
    if plot_spikes:
        hold(True)
        plot([k2*dt]*2, [V_reset,V_spk], '-',zorder=-Nsuper+i_super,c=color,lw=LW)
    title('Adding input')

    subplot(nrows,1,3)
    plot(tt,I,c=color,lw=LW,zorder=-Nsuper+i_super)
    draw()
    
    # Wrap new phase around half-cycles
    newphase=(k2[4]-k[4])*2*dt/T
    if newphase<-1:
        newphase+=2
    elif newphase >=1:
        newphase-=2
    phaseList[i_super]=newphase

subplot(nrows,1,2)
plot([k*dt]*2, [V_reset,V_spk], 'k-',lw=LW,zorder=-50)
xlim(tlim)
ylim(Vlim)
ylabel('V')

subplot(nrows,1,3)
xlim(tlim)
ylim(-25, 25)
ylabel(r'$I_e$ (pA)')

# plot(timeList/T, phaseList,'o-')
# xlabel('Pulse timing (Period)')
# ylabel('Phase reset (degree)')
# grid(True)

subplot(nrows,2,7)

X=2*timeList/T
Y=phaseList+0.0
    
# Unwrap phases
jump_ix = np.argmax(np.abs(np.diff(Y)))+1
X = r_[X[jump_ix:]-2, X[:jump_ix]]
Y = r_[Y[jump_ix:], Y[:jump_ix]]
colors = colors[jump_ix:] + colors[:jump_ix]
midX = X[int(Nsuper/2)+1]

for i_super in xrange(Nsuper):
    plot(X[i_super],Y[i_super],'o',mec='k',
        mfc=colors[i_super],ms=6,mew=1,zorder=i_super)
    print X[i_super],Y[i_super]
# p=np.polyfit(x,y,1)
# yp=np.polyval(p,x)
# plot(x,yp,'r-',zorder=0)
# plot(X,Y,'b-',lw=1,zorder=0)
ylabel(r'Phase Reset ($\pi$)')
ax = gca()
ax.set_xticks(linspace(-1, 1, 5))
ax.set_yticks(linspace(-1, 1, 5))
axis('equal')
axis('image')
xlim(midX-1.2, midX+1.2)
ylim(-1.2, 1.2)

ion()
show()