#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:00:52 2020

@author: Carlos and Alec
"""
import numpy as np #for arrays and math
import scipy
from scipy.io import loadmat
import matplotlib.pyplot as plt #for plotting
import pickle #for saving data
import matplotlib.animation as animation
import time #for timing the run
from scipy.interpolate import RegularGridInterpolator #for estimating velocities between grid points
plt.rcParams['animation.ffmpeg_path'] = '/anaconda3/bin/ffmpeg' # Important
import copy
import math
#------------------------------------------------------------------------------
# ------------------------------functions--------------------------------------
#------------------------------------------------------------------------------

# Class for evolving trajectories of Kolmogorov flow
class Evolve:
    def __init__(self,Nx,Ny,n,Re,dt,D=0.001,a=0.1,sigma=0.05):
        # Initialize wavenumber 
        self.Nx=Nx
        self.Ny=Ny
        self.x=np.linspace(0,2*np.pi-2*np.pi/Nx,Nx)
        self.y=np.linspace(0,2*np.pi-2*np.pi/Ny,Ny)
        self.n=n
        [self.kx,self.ky]=np.meshgrid(np.fft.fftfreq(Nx, d=1/Nx),np.fft.fftfreq(Ny, d=1/Ny),indexing='ij')
        self.D=D # Added diffusion coeff
        self.a=a # Added Gaussian amplitude
        self.sigma=sigma # Added Gaussian width
        
        # For impementing the forcing 
        self.kron_kx = np.zeros((Nx,Ny))
        self.kron_kx[0,:] = 1
        self.kron_ky = np.zeros((Nx,Ny))
        self.kron_ky[:,n] = 1
        self.kron_ky[:,-n]=1
        self.force=(self.n/2)*self.kron_kx*self.kron_ky
        
        # All these equations are for converting vorticity to velocity
        alphainner = np.divide(self.ky[1:,1:],self.kx[1:,1:])
        onemat = np.ones((self.Nx-1,self.Ny-1))
        d1 = 1j*self.ky[1:,1:]*alphainner
        d2 = 1j*self.kx[1:,1:]
        self.d3 = np.divide(onemat,d2+d1)
        self.d4 = -alphainner*self.d3
        
        
        self.dt=dt
        self.workers=1
        
        # Used for solver

        self.G = (self.kx**2+self.ky**2)/Re #G
        self.inv=1/(1/self.dt+self.G/2) 
        self.lin=(1/self.dt-self.G/2)
        
        self.Gc=self.D*(self.kx**2+self.ky**2) #Gc
        self.invc=1/(1/self.dt+self.Gc/2) 
        self.linc=(1/self.dt-self.Gc/2)
        
        # Initialize F(u) and F(v) with zeros
        self.Fu = np.zeros((Nx,Ny),complex)
        self.Fv = np.zeros((Nx,Ny),complex)
        
    def init_C(self):
        # Initialize tracer
        x,y=np.meshgrid(self.x,self.y,indexing='ij')
        C=np.zeros((self.Nx,self.Ny))
        C_hat=(1/self.Nx)*(1/self.Ny)*np.fft.fftn(C,axes=(0,1))
        return C, C_hat
        
    # Convert vorticity to velocity
    def w2u(self,w):
        W=1/Nx*1/Ny*np.fft.fftn(w,axes=(0,1))
        
        # case kx = 0, ky /= 0
        self.Fu[0,1:] = (-1/(1j*self.ky[0,1:]))*W[0,1:]
        # case kx /= 0, ky = 0
        self.Fv[1:,0] = (1/(1j*self.kx[1:,0]))*W[1:,0]        
        # Build final matrices
        self.Fu[1:,1:] = self.d4*W[1:,1:]
        self.Fv[1:,1:] = self.d3*W[1:,1:]
        
        u = Nx*Ny*np.real(np.fft.ifftn(self.Fu,axes=(0,1)))
        v = Nx*Ny*np.real(np.fft.ifftn(self.Fv,axes=(0,1)))
        
        return u,v
    
    # Convert fft vorticity to velocity
    def W2u(self,W):        
        # case kx = 0, ky /= 0
        self.Fu[0,1:] = (-1/(1j*self.ky[0,1:]))*W[0,1:]
        # case kx /= 0, ky = 0
        self.Fv[1:,0] = (1/(1j*self.kx[1:,0]))*W[1:,0]        
        # Build final matrices
        self.Fu[1:,1:] = self.d4*W[1:,1:]
        self.Fv[1:,1:] = self.d3*W[1:,1:]
        
        u = Nx*Ny*np.real(np.fft.ifftn(self.Fu,axes=(0,1)))
        v = Nx*Ny*np.real(np.fft.ifftn(self.Fv,axes=(0,1)))
        
        return u,v
 
    # Convert velocity to vorticity
    def u2w(self,u,v):
        # Fourier transform of u
        u_fft=1/Nx*1/Ny*np.fft.fftn(u,axes=(0,1))
        v_fft=1/Nx*1/Ny*np.fft.fftn(v,axes=(0,1))
        
        # calculate vorticity in fourier space
        W = 1j*self.kx*v_fft - 1j*self.ky*u_fft
        w = Nx*Ny*np.real(np.fft.ifftn(W,axes=(0,1)))
        
        return w
    
    def forcing2f(self):
        x,y=np.meshgrid(self.x,self.y,indexing='ij')
        forcing=self.a*np.exp(-(((x-np.pi)**2)+(y-np.pi)**2)/(2*self.sigma**2))
        forcing_hat=(1/self.Nx)*(1/self.Ny)*np.fft.fftn(forcing,axes=(0,1))
        
        return forcing_hat
        
    # Defines governing MDE for tracer
    def C2f(self,C_hat,W,forcing_hat):
        C=self.Nx*self.Ny*np.real(np.fft.ifftn(C_hat,axes=(0,1)))
        u,v=self.W2u(W)
        fftuC=(1/self.Nx)*(1/self.Ny)*np.fft.fftn(u*C,axes=(0,1))
        fftvC=(1/self.Nx)*(1/self.Ny)*np.fft.fftn(v*C,axes=(0,1))
        adv_hat = -1j*((self.kx*fftuC)+(self.ky*fftvC))
        # diff_hat = -self.D*(self.kx**2+self.ky**2)*C_hat
        
        return adv_hat + forcing_hat
    
       
    def dCdt_CN_Heunn(self,C_hat,W,forcing_hat,T):
        tsteps=round(T/self.dt)
        
        for _ in range(tsteps):
            
            Wf=self.W2f(W)
            invlinW=self.inv*self.lin*W
            W1=invlinW+self.inv*Wf #Equation 3.4
            W_next=invlinW+.5*self.inv*(self.W2f(W1)+Wf) #Equation 3.5
            
            fc=self.C2f(C_hat,W,forcing_hat)
            C_next=self.invc*(self.linc*C_hat+fc)
            
            fcf=self.C2f(C_next,W_next,forcing_hat)
            C_hat=self.invc*(self.linc*C_hat+0.5*(fc+fcf))
            
            W=W_next
            
        return C_hat, W_next
   
    #Equation 3.3
    def W2f(self,W): 
        # Get the realspace w and u,v
        w = Nx*Ny*np.real(np.fft.ifftn(W,axes=(0,1)))
        [u,v]=self.W2u(W)

        # Calculate the fft 
        fftuw = 1/Nx*1/Ny*np.fft.fftn(u*w,axes=(0,1))
        fftvw = 1/Nx*1/Ny*np.fft.fftn(v*w,axes=(0,1))
        
        # Calculate f
        f=-1j*(self.kx*fftuw+self.ky*fftvw) - self.force
        
        return f
    
    def dWdt(self,W,T):
        tsteps=round(T/self.dt)
        for _ in range(tsteps):
            Wf=self.W2f(W)
            invlinW=self.inv*self.lin*W
            W1=invlinW+self.inv*Wf #Equation 3.4
            W=invlinW+.5*self.inv*(self.W2f(W1)+Wf) #Equation 3.5
    
        return W

    def dWdtStreak(self,W,T):
        tsteps=round(T/self.dt)
        
        # Compute the current velocity at the point
        [u,v]=self.W2u(W)
        
        # Interpolate to the point location
        ui = RegularGridInterpolator((self.x,self.y), u)
        vi = RegularGridInterpolator((self.x,self.y), v)

        # Particle positions for streakline
        xIC= np.pi
        yIC= np.pi
        xs=[[xIC,yIC]]
        for k in range(tsteps):
            Wf=self.W2f(W)
            invlinW=self.inv*self.lin*W
            W1=invlinW+self.inv*Wf #Equation 3.4
            W=invlinW+.5*self.inv*(self.W2f(W1)+Wf) #Equation 3.5
            
            # Compute the new velocity field
            [u2,v2]=self.W2u(W)
            # Interpolate to the point location
            ui2 = RegularGridInterpolator((self.x,self.y), u2)
            vi2 = RegularGridInterpolator((self.x,self.y), v2)

            for i in range(len(xs)):
                ux=ui(np.asarray([xs[i][0],xs[i][1]])[np.newaxis,:]).squeeze()
                vx=vi(np.asarray([xs[i][0],xs[i][1]])[np.newaxis,:]).squeeze()
                
                xe=xs[i][0]+ux*self.dt
                ye=xs[i][1]+vx*self.dt
                # If the data leaves the domain wrap around
                xe = xe % self.x[-1]
                ye = ye % self.y[-1]
                
                ux2=ui2(np.asarray([xe,ye])[np.newaxis,:]).squeeze()
                vx2=vi2(np.asarray([xe,ye])[np.newaxis,:]).squeeze()
                
                xs[i][0]+=.5*(ux+ux2)*self.dt
                xs[i][1]+=.5*(vx+vx2)*self.dt
                
                # If the data leaves the domain wrap around
                xs[i][0]=xs[i][0] % self.x[-1]
                xs[i][1]=xs[i][1] % self.y[-1]

            ui=copy.deepcopy(ui2)
            vi=copy.deepcopy(vi2)
            
            if  k>0 and k%10==0:
                xs.append([xIC,yIC])
            
            
        return W,xs
    
    def AdvectParticles(self, xs, W):
        [u,v] = self.W2u(W)

        ui = RegularGridInterpolator((self.x,self.y), u)
        vi = RegularGridInterpolator((self.x,self.y), v)


        for i in range(len(xs)):
            ux = ui([xs[i]]).squeeze()
            vx = vi([xs[i]]).squeeze()

            xs[i][0] = (xs[i][0] + ux*self.dt) % self.x[-1]
            xs[i][1] = (xs[i][1] + vx*self.dt) % self.y[-1]

        return xs
    
    def Plotting(self,u,title=''):
        plt.pcolormesh(self.x,self.y, u, vmin=-5,vmax=5,shading='gouraud')
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        
#------------------------------------------------------------------------------
# ------------------------------main script------------------------------------
#------------------------------------------------------------------------------
if __name__=='__main__':

    Nx = int(128)
    Ny = int(128)
    dt = 0.01 # If things blow up dt appears most important to preventing that (.01 40 .005 80)
    #tfin=dt*8000
    Re = 14.4
    n  = 2
    D = 0.0
    
    x=np.linspace(0,2*np.pi-2*np.pi/Nx,Nx)
    y=np.linspace(0,2*np.pi-2*np.pi/Ny,Ny)
    
    u0=(Re/n**2)*np.sin(n*y)
    u0=np.repeat(u0[np.newaxis,:],Nx,axis=0)    
    v0 = np.zeros((Nx,Ny))

    # Initialize the class
    KFlow=Evolve(Nx,Ny,n,Re,dt,D=D)
    
    ########################### Initial Conditon ##############################
    #Calculate the first field in fourier space
    np.random.seed(0) #same IC
    w0=KFlow.u2w(u0,v0)+.01*Re*np.random.randn(Nx,Ny)

    ########################## Evolve Trajectory ##############################
    #plotw=np.zeros(tsteps)
    W=1/Nx*1/Ny*np.fft.fftn(w0,axes=(0,1))
    # Evolve
    T = 1000
    start = time.time()
    W=KFlow.dWdt(W,T*dt)
    end = time.time()
    pickle.dump(W,open('Wini.p','wb'))

    
    W=pickle.load(open('Wini.p','rb'))
    
    # Evolve tracer concentration profile
    Cs=[]
    Ws=[]
    times=[]
    C,C_hat=KFlow.init_C()
    forcing_hat=KFlow.forcing2f()

    dT = 2.0

    xs = [[np.pi, np.pi]]   # initial particle
    particle_history = []

    for i in range(T):

        print(i)

        # save current fields
        Cs.append(Nx*Ny*np.real(np.fft.ifftn(C_hat,axes=(0,1))))
        Ws.append(Nx*Ny*np.real(np.fft.ifftn(W,axes=(0,1))))
        times.append(i)

        # evolve tracer + flow
        C_hat, W = KFlow.dCdt_CN_Heunn(C_hat, W, forcing_hat, dT)

        # advect particles
        xs = KFlow.AdvectParticles(xs, W)

        # inject new particle every 10 frames
        if i % 10 == 0:
            xs.append([np.pi, np.pi])

        # save streakline snapshot
        particle_history.append(copy.deepcopy(xs))

# Plot tracer concentrations 
    sp=4
    for i in range(1,len(Cs)-1):
        plt.figure()
        plt.pcolormesh(KFlow.x, KFlow.y, Cs[i].T/np.max(Cs[i]),cmap='Blues', shading='gourad',vmin=0,vmax=1)
        pts = np.asarray(particle_history[i])
        plt.plot(pts[:, 0],pts[:, 1],"ro-",markersize=3,linewidth=1.0,label="streakline")
        plt.colorbar()
        # plt.contour(KFlow.x, KFlow.y, Ws[i].T,levels=[-3,-1,1, 3],colors='k')
        [u,v]=KFlow.w2u(Ws[i])
        plt.quiver(KFlow.x[::sp], KFlow.y[::sp], u[::sp,::sp].T, v[::sp,::sp].T, color='k', scale=50)
        plt.title(f'Tracer concentration at t = {times[i]}')
        plt.xlabel('x')
        plt.ylabel('y')
        #plt.show()
        plt.savefig(f'./Video/{i:04d}.png')
        
    # for i in range(len(Cs)):
    #     plt.figure()
    #     plt.pcolormesh(KFlow.x, KFlow.y, Ws[i].T, shading='gourad')
    #     plt.colorbar()
    #     plt.title(f'Tracer concentration at t = {times[i]}')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.show()
    

    # Plot after evolving
    #Ws=[W]
  #  ws=[]
  # xs=[]
  #  T=100
  #  for i in range(T):
  #      print(i)
  #      W=KFlow.dWdt(W,400*dt)
  #      W,xp=KFlow.dWdtStreak(W,201*dt)
  #      #Ws.append(W)
  #      xs.append(xp)
  #      ws.append(Nx*Ny*np.real(np.fft.ifftn(W,axes=(0,1))))
  #      
        
        # Evolve the field forward more time before sampling again
     #   W=KFlow.dWdt(W,300*dt) # This means every 5 time units I sample 1 time unit of streak data
    
#import os

#path = r'C:\Users\aaron\OneDrive\Desktop\Linot Lab'
#os.makedirs(path, exist_ok=True)

#filename = os.path.join(path, 'KFlowStreakline32x32Re14p4n2T50000dt0p01.pkl')
#with open(filename, 'wb') as f:
#    pickle.dump((ws, xs), f)