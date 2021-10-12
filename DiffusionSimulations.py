# -*- coding: utf-8 -*-
"""
Created on Mon Nov 04 13:46:09 2013

@author: Colleen Bailey
A set of classes and methods intended to model the diffusion signal in a range
of environments. Restricted  diffusion inside spheres and cylinders can be 
modelled, as well as a tortuosity approximation of the extacellular space
2019-09-18: adjusting inputs so that parameters passed are scaled near 1.
"""

import numpy as np
gammap=42.576e6*2*np.pi

class Tortuosity():
    """
    Tortuosity(fs=0.5,ft=0.3,acyl=np.r_[5,50],DE=1.7, fibreDir=[1,0,0], AngleDist='random')
    Set up as a tortuosity approximation for aligned cylinders (Stanisz et al., 
    1997 and Szafer et al., 1995), but there exist formulae for spheres (Stanisz, 2003).
    """
    def __init__(self, fs=0.5,ft=0.3,acyl=np.r_[5,50],DE=1.7, fibreDir=[1,0,0], AngleDist='random'):
        """
        acyl in um, DE in um^2/ms
        """
        self.fs=fs
        self.ft=ft
        self.acyl=acyl*1e-6
        self.DE=DE*1e-9
        self.AngleDist=AngleDist
        self.Lpar=np.log(2*self.acyl[1]/self.acyl[0]-1)*(self.acyl[0]/self.acyl[1])**2
        self.Lperp=(1-self.Lpar)/2
        self.ADCEpar=self.DE*(1-self.fs-self.ft)**((self.fs/2+self.ft*self.Lpar/(1-self.Lpar))/(self.fs+self.ft))
        self.ADCEperp=self.DE*(1-self.fs-self.ft)**((self.fs/2+self.ft*self.Lperp/(1-self.Lperp))/(self.fs+self.ft))
    def GetSig(self,pulseseq):
        """
        Function to calculate the diffusion signal for a particular pulse 
        sequence (see PulseSequences.py module)
        (Right now angule distribution is unused because I've assumed aligned 
        cylinders)
        """
        bVals=pulseseq.GetB()
        SigE=np.exp(-bVals*(self.ADCEpar+self.ADCEperp))
        return SigE
    def GetJac(self,pulseseq):
        bVals=pulseseq.GetB()
        return -1*bVals*np.exp(-bVals*(self.ADCEpar+self.ADCEperp))

class Sphere(object):
    """
    Sphere(rad=5,DI=1.1,ApproxType='GPD')
    Water motion inside restricted sphere with defined radius
    Inputs:
    rad - The radius of the cell in um
    DI - Intracellular diffusion time for this group of cells in um^2/s
    ApproxType - Approximation used to calculate dephasing by gradients. Choices
        are GPD (Gaussian Phase Distribution, default) or SPG (short pulse gradient)
    """
    def __init__(self, rad=5,DI=1.1,ApproxType='GPD'):
        self.rad=rad*1e-6
        self.DI=DI*1e-9
        self.ApproxType=ApproxType
        
    def GetSig(self,pulseseq):
        """
        Function to calculate the diffusion signal for a particular pulse 
        sequence (GPD is set up for either TRSE or PGSE; see PulseSequences.py 
        module). The SPG calculation involves an infinite series, which is 
        terminated after 60 steps, hard-coded in vec below. The GPD approximation 
        involves the BesselRoots, which are hard-coded below.
        """
        lval=self.rad
        Sig=np.ones(pulseseq.gMag.shape)
        if self.ApproxType=="SPG":
            #Short pulse gradient approximation
            for gct,(gval,dval,Dval) in enumerate(zip(pulseseq.gMag,pulseseq.delta,pulseseq.DELTA)):
                if gval!=0:
                    qval=gammap*gval*dval
                    vec=np.r_[1:60] #accuracy of convergence
                    qsum=sum(np.exp(-1*vec**2*np.pi*np.pi*self.DI*Dval/lval/lval)*(1-(-1)**vec*np.cos(qval*lval))/((qval*lval)**2-(vec*np.pi)**2)**2)
                    Sig[gct] = (2*(1-np.cos(qval*lval))/(qval*lval)**2+4*(qval*lval)**2*qsum)
        elif self.ApproxType=="GPD":
            #GPD approximation for spheres
            BesselRoots=np.r_[2.08157597782, 5.94036999057, 9.20584014294, 12.4044450219, 15.5792364104, 18.7426455848, 21.8996964795, 25.052825281, 
                           28.203361004, 31.3520917266, 34.4995149214, 37.6459603231, 40.7916552313, 43.9367614714, 47.0813974122, 50.2256516492, 
                           53.3695918205, 56.5132704622, 59.6567290035, 62.8000005565, 65.9431119047, 69.0860849466, 72.228937762, 75.3716854093, 
                           78.5143405319, 81.656913824, 84.7994143922, 87.9418500397, 91.0842274915, 94.2265525746, 97.3688303629, 100.511065295, 
                           103.653261272, 106.795421733, 109.937549726, 113.079647959, 116.221718846, 119.363764549, 122.505787005, 125.647787961, 
                           128.789768989, 131.931731515, 135.073676829, 138.215606107, 141.357520417, 144.499420737, 147.64130796, 150.783182905, 
                           153.925046323, 157.066898908, 160.208741296, 163.350574075, 166.492397791, 169.634212946, 172.776020008, 175.917819411, 
                           179.059611558, 182.201396824, 185.343175559, 188.484948089, 191.626714721, 194.76847574, 197.910231412, 201.05198199]
            Betam=BesselRoots/lval
            Yfunc=lambda x: np.exp(-1*self.DI*Betam**2*x)  
            try: #TRSE pulse sequence is the only one with this format
                gDirsvec=pulseseq.gDirs
                gMagvec=pulseseq.gMag
                del1vec=pulseseq.del1
                del2vec=pulseseq.del2
                del3vec=pulseseq.del3
                t1vec=pulseseq.t1
                t2vec=pulseseq.t2
                t3vec=pulseseq.t3
                for gct,(gDir,gMag,del1,del2,del3,t1,t2,t3) in enumerate(zip(gDirsvec,gMagvec,del1vec,del2vec,del3vec,t1vec,t2vec,t3vec)):
                    numer=2*self.DI*Betam**2*(del1+del2)-5-(Yfunc(t2-t1)-Yfunc(t3-t1)-Yfunc(t3-t2)-Yfunc(del1)-Yfunc(t2-t1-del1)+
                        +Yfunc(t3-t1-del1)-2*Yfunc(del2)-2*Yfunc(t2-t1+del2)+2*Yfunc(t2-t1+del2-del1)+2*Yfunc(t3-t2-del2)-
                        2*Yfunc(del3)+Yfunc(del2+del3)+Yfunc(t2-t1+del2+del3)-Yfunc(t2-t1+del2+del3-del1)-
                        2*Yfunc(t3-t2+del1-del3)-Yfunc(t3-t1+del2-del3)-Yfunc(del1+del2-del3)+Yfunc(t3-t1+del1+del2-del3)+
                        Yfunc(t3-t2+del1+del2-del3)-Yfunc(t3-t2-del2-del3)+Yfunc(t3-t2+del1-2*del3))
                    denom=self.DI**2*Betam**6*(lval**2*Betam**2-2)
                    Sig[gct]=np.exp(-2*gammap**2*sum(numer/denom)*gMag**2)
            except AttributeError: # pulse sequence is PGSE not TRSE
                gDirsvec=pulseseq.gDirs
                gMagvec=pulseseq.gMag
                delvec=pulseseq.delta
                DELTA=pulseseq.DELTA
                for gct,(gDir,gMag,del1,t1) in enumerate(zip(gDirsvec,gMagvec,delvec,DELTA)):
                    numer=2*self.DI*del1*(Betam)**2-2+2*Yfunc(del1)+2*Yfunc(t1)-Yfunc(t1-del1)-Yfunc(t1+del1)
                    denom=self.DI**2*Betam**6*(self.rad**2*Betam**2-2)
                    Sig[gct]=np.exp(-2*gammap**2*sum(numer/denom)*gMag**2)
        return Sig
    def GetJac(self,pulseseq):
        dp=1e-5
        pList=[self.rad,self.DI]
        npar=len(pList)
        jac=np.ones([npar,len(pulseseq.ScanPars['gMag'].squeeze())])
        # compute numerically
        S0=self.GetSig(pulseseq)
        for jct in range(npar):
            phold=pList[jct]
            pList[jct]=pList[jct]*(1+dp)
            S1=self.GetSig(pulseseq)
            pList[jct]=phold
            jac[jct,:]=(S1-S0)/(dp*phold)
        return jac    
        
class Cylinder(object):
    """
    Cylinder(rad=5, length=50, DI=1.1, fibreDir=[1,0,0],ApproxType='GPD')
    Water motion inside a cylinder with radius and particular orientation
    Inputs:
    rad - The radius of the cylinder in um
    length - length of cylinder in um (currently unused in signal calculations,
            which assuming infinitely long cylinders)
    DI - Intracellular diffusion time for this cylinder in um^2/s
    fibreDir - direction of main fibre axis (same co-ordienate system as gradient
                directions use in pulse sequence)
    ApproxType - Approximation used to calculate dephasing by gradients. Choices
        are GPD (Gaussian Phase Distribution, default) or SPG (short pulse gradient)
    """
    def __init__(self, rad=5, length=50, DI=1.1, fibreDir=[1,0,0],ApproxType='GPD'):
        self.rad=rad*1e-6
        self.length=length*1e-6
        self.DI=DI*1e-9
        self.fibreDir=fibreDir
        self.ApproxType=ApproxType
        
    def GetSig(self,pulseseq,numSteps=60):
        """
        Function to calculate the diffusion signal for a pulse sequence (only
        set up for PGSE currently, see PulseSequences.py module). The SPG 
        calculation involves an infinite series, which is terminated after
        numSteps steps.
        The GPD approximation involves the BesselRoots, which are hard-coded below.
        Currently using the assumption that the signal is the product of the 
        parallel and perpendicular signals (Assaf et al., 2004). The calculation 
        can be made using different approximations. 
        """
        lval=self.rad
        gradDir=pulseseq.gDirs
        gradMag=pulseseq.gMag
        delta=pulseseq.delta
        Delta=pulseseq.DELTA
        DI=self.DI
        if type(gradMag) is not np.ndarray:
            gradMag=np.r_[gradMag]
        if type(Delta) is not np.ndarray:
            Delta=Delta*np.ones(gradMag.shape)
        if type(delta) is not np.ndarray:
            delta=delta*np.ones(gradMag.shape)
        Sperp=np.ones(gradMag.shape)
        Spar=np.exp(-1*(gammap*gradMag*np.dot(gradDir,self.fibreDir)*delta)**2*(Delta-delta/3)*DI)
        sinThetasq=1-np.dot(gradDir,self.fibreDir)**2
        if self.ApproxType=="SPG":
            #Short pulse gradient approximation
            for gct,(gval,dval,Dval) in enumerate(zip(gradMag*np.sqrt(1-np.dot(gradDir,self.fibreDir)**2),delta,Delta)):
                if gval!=0:
                    qval=gammap*gval*dval
                    vec=np.r_[1:60] #accuracy of convergence
                    qsum=np.sum(np.exp(-1*vec**2*np.pi*np.pi*DI*Dval/lval/lval)*(1-(-1)**vec*np.cos(qval*lval))/((qval*lval)**2-(vec*np.pi)**2)**2)
                    Sperp[gct] = (2*(1-np.cos(qval*lval))/(qval*lval)**2+4*(qval*lval)**2*qsum)
        elif self.ApproxType=="GPD":
            #GPD approximation for cylinders
            import scipy.special
            for gct,(gval,dval,Dval) in enumerate(zip(gradMag,delta,Delta)):
                BesselRoots=scipy.special.jnp_zeros(1,numSteps) #can adjust for better convergence
                Betam=BesselRoots/lval
                Yfunc=lambda x: np.exp(DI*Betam**2*x)
                numer=2*DI*Betam**2*dval-2+2*Yfunc(-1*dval)+2*Yfunc(-1*Dval)-Yfunc(dval-Dval)-Yfunc(-1*(Dval+dval))
                denom=DI**2*Betam**6*(lval**2*Betam**2-1)
                Sperp[gct]=np.exp(-2*gammap**2*np.sum(numer/denom)*sinThetasq[gct]*gval**2)
        return Spar*Sperp        

class IsotropicFree(object):
    """
    IsotropicFree(ADC=1)
    Free diffusion in an isotropic environment with coeffecient ADC
    """
    def __init__(self,ADC=1):
        self.ADC=ADC*1e-9
        
    def GetSig(self,pulseseq):
        bVals=pulseseq.GetB()
        return np.exp(-1*bVals*self.ADC)
        
    def GetJac(self,pulseseq):
        bVals=pulseseq.GetB()
        return -1*bVals*np.exp(-1*bVals*self.ADC)
        
class Stick(object):
    """
    Stick(thetaf,phif,DI=1.1)
    Diffusion along a 1D "stick" (infinitessimally think cyclinder) with directions
    defined by thetaf (angle from z) in radians and phif (rotation of xy-projection from x-axis)
    """
    def __init__(self,thetaf,phif,DI=1.1):
        self.DI=DI*1e-9
        self.thetaf=thetaf
        self.phif=phif
    @property
    def fibreDir(self):
        return np.r_[np.sin(self.thetaf)*np.cos(self.phif),np.sin(self.thetaf)*np.sin(self.phif),np.cos(self.thetaf)]
        
    def GetSig(self,pulseseq):
        gFact=np.zeros(pulseseq.gMag.shape)
        for gct,gd in enumerate(pulseseq.gDirs):
            gFact[gct]=np.dot(pulseseq.gDirs[gct],self.fibreDir)
        bVals=pulseseq.GetB()*gFact
        return np.exp(-1*bVals*self.DI)

        
class StretchedExp(object):
    """
    StretchedExp(ADC=1,alpha=1)
    Stretched exponential (phenomenological) diffusion model
    ADC - apparent diffusion coefficient in um^2/ms
    alpha - exponent
    """
    def __init__(self,ADC=1,alpha=1):
        self.ADC=ADC*1e-9
        self.alpha=alpha
        
    def GetSig(self,pulseseq):
        bVals=pulseseq.GetB()
        return np.exp(-1*(bVals*self.ADC)**self.alpha)
        
class Kurtosis(object):
    """
    Kurtosis(ADC=1,kurt=0)
    Kurtosis (phenomenological) diffusion model
    ADC - apparent diffusion coefficient in um^2/ms
    kurt - kurtosis parameter. A value of 0 is Gaussian diffusion
    """
    def __init__(self,ADC=1,kurt=0):
        self.ADC=ADC*1e-9
        self.kurt=kurt
        
    def GetSig(self,pulseseq):
        bVals=pulseseq.GetB()
        return np.exp(-1*bVals*self.ADC+(bVals)**2*self.kurt*self.ADC**2/6)
        
class DiffTensor(object):
    """
    DiffTensor(lambda1,lambda2,lambda3,theta,phi,alpha)
    lambda 1 - diffusion coefficient along primary direction (defined by theta 
            and phi below) in um^2/ms
    lambda 2 - diffusion coefficient along secondary directions in um^2/ms
    lambda 3 - diffusion coefficient along tertiary direction
    theta - angle of primary diffusion direction relative to z-axis, in radians
    phi - angle of primary diffusion direction projection in xy-plan relative to x-axis, in radians
    alpha - angle of secondary diffusion direction (third direction is determined by orthogonality constraint)
    """
    def __init__(self,lambda1,lambda2,lambda3,theta,phi,alpha):
        self.theta=theta
        self.phi=phi
        self.alpha=alpha
        self.lambda1=lambda1*1e-9
        self.lambda2=lambda2*1e-9
        self.lambda3=lambda3*1e-9
    @property
    def e1(self):
        return np.r_[np.sin(self.theta)*np.cos(self.phi),np.sin(self.theta)*np.sin(self.phi),np.cos(self.theta)]
    @property
    def e2(self):
        # first, find a vector perpendicular to e1. Then rotate by alpha around e1
        if (np.dot(self.e1,np.r_[0,1,0])-1)==0:
            rotvec=np.dot(np.array([[1,0,0],[0,0,-1],[0,1,0]]),self.e1)
        else:
            rotvec=np.dot(np.array([[0,0,1],[0,1,0],[-1,0,0]]),self.e1)
        RotMat=np.cos(self.alpha)*np.eye(3) + np.sin(self.alpha)*np.array([[0,-1*self.e1[2],self.e1[1]],[self.e1[2],0,-1*self.e1[0]],
            [-1*self.e1[1],self.e1[0],0]]) + (1-np.cos(self.alpha))*np.array([[self.e1[0]**2,self.e1[0]*self.e1[1],self.e1[0]*self.e1[2]],
            [self.e1[0]*self.e1[1],self.e1[1]**2,self.e1[1]*self.e1[2]],[self.e1[0]*self.e1[2],self.e1[1]*self.e1[2],self.e1[2]**2]])
        return np.dot(RotMat,rotvec)
    @property
    def e3(self):
        return np.cross(self.e1,self.e2)/np.linalg.norm(np.cross(self.e1,self.e2))
    def GetSig(self,pulseseq):
        bVals=pulseseq.GetB()
        SigVals=np.zeros(bVals.shape)
        for gct,gd in enumerate(pulseseq.gDirs):
            SigVals[gct]=np.exp(-1*bVals[gct]*(self.lambda1*np.dot(self.e1,gd)**2+self.lambda2*np.dot(self.e2,gd)**2
                +self.lambda3*np.dot(self.e3,gd)**2))
        return SigVals
    
class Zeppelin(object):
    """
    Zeppelin(lambda1,lambda2,theta,phi)
    Diffusion calculation for spherically symmetric tensor
    lambda 1 - diffusion coefficient along primary direction (defined by theta 
            and phi below) in um^2/ms
    lambda 2 - diffusion coefficient along secondary directions in um^2/ms
    theta - angle of primary diffusion direction relative to z-axis, in radians
    phi - angle of primary diffusion direction projection in xy-plan relative to x-axis, in radians
    """
    def __init__(self,lambda1,lambda2,theta,phi):
        self.theta=theta
        self.phi=phi
        self.lambda1=lambda1
        self.lambda2=lambda2
    @property
    def e1(self):
        return np.r_[np.sin(self.theta)*np.cos(self.phi),np.sin(self.theta)*np.sin(self.phi),np.cos(self.theta)]

    def GetSig(self,pulseseq):
        bVals=pulseseq.GetB()
        SigVals=np.zeros(bVals.shape)
        for gct,gd in enumerate(pulseseq.gDirs):
            SigVals[gct]=np.exp(-1*bVals[gct]*self.lambda1*np.dot(self.e1,gd)**2-1*bVals[gct]*self.lambda2*(1-np.dot(self.e1,gd)**2))
        return SigVals

if __name__ == '__main__':
    """
    Examples/debugging
    """
    import PulseSequences as psq
    import matplotlib.pyplot as plt
    
    psfull=psq.get_sample_PGSE()
    ps1=psq.slice_ps(psfull,np.r_[0:10])
    ps2=psq.slice_ps(psfull,np.r_[10:20])
    DiffI=Sphere(DI=1,rad=5)
    SigI1=DiffI.GetSig(ps1)
    DiffE=IsotropicFree(ADC=1)
    SigE1=DiffE.GetSig(ps1)
    SigI2=DiffI.GetSig(ps2)
    SigE2=DiffE.GetSig(ps2)
    f1,ax1=plt.subplots(1,1)
    f1.set_size_inches([4,4])
    ax1.semilogy(ps1.GetB()/1e6,0.25*SigI1+0.75*SigE1,'-b',label='Delta={:.2f} ms'.format(ps1.DELTA[0]*1000))
    ax1.semilogy(ps2.GetB()/1e6,0.25*SigI2+0.75*SigE2,'-r',label='Delta={:.2f} ms'.format(ps2.DELTA[0]*1000))
    ax1.legend()
    ax1.set_xlabel('b ($s/mm^2$)')
    ax1.set_ylabel('Normalized Signal')
    f1.set_facecolor('w')