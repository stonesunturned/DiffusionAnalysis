# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 21:21:15 2014
Pulse Sequence module for storing scan parameters that can be use to calculate 
diffusion MRI signal. (Designed for use with DiffusionSimulations.py but should
work generally)

@author: Colleen Bailey (colleen.em.bailey@gmail.com)
Classes:
    TRSE - twice refocused spin echo
    PGSE - pulsed gradient spin echo (conventional Stejskal-Tanner)
    fromB - basic class for storing the b-values from a sequence when non-restricted
        signals (IsotropicFree, Tortuosity, DiffTensor, Zeppelin)
Create an instance of the pulse sequence obvious first and then it can be filled
by various methods (loading a scheme file, dictionary)

Methods:
    from_b_to_g - calculates gradient magnitudes for PGSE given b-values and gradient timings
    rep_ps - repeat the parameters in a PulseSequence n times and create new PulseSequence object
    add_zeros_ps - add zeros to the start of a PulseSequence object and return new object
    slice_ps - select a subset of PulseSequence measurements and return as new PulseSequence object
"""

import numpy as np
gammap=2.675987E8


class TRSE(object):
    """
    Twice-refocused spin echo
    Generate as:
        NewTRSE=TRSE()
        NewTRSE.fill_from_file(fname)
        OR
        NewTRSE.fill_from_dict(pardict)
    """
    def __init__(self):
        dkeys=['gDirs', 'gMag', 'del1','t1','del2','t2','del3','t3','TE']
        self.ScanPars=dict()
        self.dkeys=dkeys
        for dk in dkeys:
            self.ScanPars[dk]=list()
            
    def fill_from_file(self,fname=None):
        """
        Generates a TRSE object from a Camino scheme file.
        """
        if fname is not None:
            with open(fname,'r') as f:
                f.readline() #first line contains scheme type. Skip.
                for line in f:
                    vals=[float(d) for d in line.split()]
                    self.ScanPars['gDirs'].append(vals[0:3])
                    for dct,dk in enumerate(self.dkeys[1:]):
                        self.ScanPars[dk].append(vals[dct+3])
            for dk in self.dkeys[1:]:
                self.ScanPars[dk]=np.array(self.ScanPars[dk])
        else:
            print('Need to give a filename')
    def fill_from_dict(self,pardict):
        for parnm in self.dkeys:
            self.ScanPars[parnm]=pardict[parnm]
        
    @property
    def gDirs(self):
        return self.ScanPars['gDirs']
    @property
    def gMag(self):
        return self.ScanPars['gMag']
    @property
    def del1(self):
        return self.ScanPars['del1']
    @property
    def del2(self):
        return self.ScanPars['del2']
    @property
    def del3(self):
        return self.ScanPars['del3']
    @property
    def del4(self):
        return self.ScanPars['del1']+self.ScanPars['del2']-self.ScanPars['del3']
    @property
    def t1(self):
        return self.ScanPars['t1']
    @property
    def t2(self):
        return self.ScanPars['t2']
    @property
    def t3(self):
        return self.ScanPars['t3']
    @property
    def TE(self):
        return self.ScanPars['TE']
        
    def GetB(self):
        sp=self.ScanPars
        tFact=sp['del1']**2*(sp['t3']-sp['t1']-sp['del1']/3+sp['del2']-
            sp['del3'])+2*sp['del1']*sp['del2']*(sp['t3']-
            sp['t2'])+sp['del2']**2*(sp['t3']-sp['t2']-sp['del2']/3+
            sp['del3'])+2*sp['del1']*sp['del3']*(sp['t2']-sp['t3']+
            sp['del3'])+2*sp['del2']*sp['del3']*(sp['t2']-
            sp['t3'])+sp['del3']**2*(sp['t3']-sp['t2']+sp['del2']-sp['del3'])
        return (gammap*self.ScanPars['gMag'])**2*tFact 
    def copy(self):
        new_ps=TRSE()
        for k,v in self.ScanPars.items():
            new_ps.ScanPars[k]=v
        return new_ps


class PGSE(object):
    """
    PGSE() Pulsed gradient spin echo object. Generate from either a Camino
    scheme file (fill_from_file), or a dict (fill_from_dict).
    """
    def __init__(self,fname=None):
        dkeys=['gDirs', 'gMag', 'DELTA','delta','TE']
        self.ScanPars=dict()
        self.dkeys=dkeys
        for dk in dkeys:
            self.ScanPars[dk]=list()
            
    def fill_from_file(self,fname=None):
        if fname is not None:
            with open(fname,'r') as f:
                f.readline() #first line contains scheme type. Skip.
                for line in f:
                    vals=[float(d) for d in line.split()]
                    self.ScanPars['gDirs'].append(vals[0:3])
                    for dct,dk in enumerate(self.dkeys[1:]):
                        self.ScanPars[dk].append(vals[dct+3])
            for dk in self.dkeys[1:]:
                self.ScanPars[dk]=np.array(self.ScanPars[dk])
        else:
            print('Need to give a filename')
                
    def fill_from_dict(self,diffdict):
        self.ScanPars['gDirs']=diffdict['gDirs']
        self.ScanPars['gMag']=diffdict['gMag']
        self.ScanPars['DELTA']=diffdict['DELTA']
        self.ScanPars['delta']=diffdict['delta']
        try:
            self.ScanPars['TE']=diffdict['TE']
        except KeyError:
            print('Warning: no TE given. Rest of pulse sequence has filled.')
            self.ScanPars['TE']=np.zeros_like(self.ScanPars['gMag'])
    @property
    def gDirs(self):
        return self.ScanPars['gDirs']
    @property
    def gMag(self):
        return self.ScanPars['gMag']
    @property
    def DELTA(self):
        return self.ScanPars['DELTA']
    @property
    def delta(self):
        return self.ScanPars['delta']
    @property
    def TE(self):
        return self.ScanPars['TE']
        
    def GetB(self):
        sp=self.ScanPars
        return (gammap*sp['gMag']*sp['delta'])**2*(sp['DELTA']-sp['delta']/3)    
    def copy(self):
        new_ps=PGSE()
        for k,v in self.ScanPars.items():
            new_ps.ScanPars[k]=v
        return new_ps
    def __len__(self):
        return len(self.gMag)
    def __add__(self,val):
        new_ps=self.copy()
        new_ps.ScanPars['gMag']=new_ps.ScanPars['gMag']+val
        return new_ps
            
class fromB(object):
    """
    fromB(bVals)
    Generates a PulseSequence object from a list of b-values (whose only method is GetB).
    """
    def __init__(self,bVals=None):
        self.ScanPars=dict()
        self.ScanPars['bVals']=np.array(bVals)
    def GetB(self):
        return self.ScanPars['bVals']
    def copy(self):
        new_ps=fromB()
        for k,v in self.ScanPars.items():
            new_ps.ScanPars[k]=v
        return new_ps
        
def from_b_to_g(bval=1000,smalldel=5e-3,DELTA=30e-3):
    """
    Calculates gradient magnitudes for PGSE from b-values, gradient durations
    (smalldel) and gradient separation (DELTA)
    """
    return np.sqrt(bval/(gammap*smalldel)**2/(DELTA-smalldel/3))

def rep_ps(old_protocol,n):
    """
    rep_ps(old_protocol,n)

    Parameters
    ----------
    old_protocol : PulseSequences object
        The protocol to be repeated.
    n : int
        Number of times to repeat protocol.

    Returns
    -------
    new_protocol : PulseSequences object
        New, replicated protocol.
    """
    new_protocol=old_protocol.copy()
    for k,v in new_protocol.ScanPars.items():
        if k != 'gDirs':
            new_protocol.ScanPars[k]=np.tile(old_protocol.ScanPars[k],n)
        else:
            new_protocol.ScanPars[k]=np.tile(old_protocol.ScanPars[k],[n,1])
    return new_protocol
    
def add_zeros_ps(old_protocol,TEs):
    """
    add_zeros_ps(old_protocol,TEs)
    Adds zeros (gradient magnitude and b-value of zero) to the start of an 
    existing protocol with the provided TEs

    Parameters
    ----------
    old_protocol : PulseSequences object
        The protocol to add zeros to the start of.
    TEs : 1D-array of floats
        TE values for the b=0 points being added to start of protocol.

    Returns
    -------
    new_protocol : PulseSequences object
        New protocol with b=0 points added to front.
    """
    
    new_protocol=old_protocol.copy()
    for k,v in new_protocol.ScanPars.items():
        if k== 'TE':
            new_protocol.ScanPars[k]=np.r_[TEs,old_protocol.ScanPars[k]]
        elif k != 'gDirs':
            new_protocol.ScanPars[k]=np.r_[np.zeros(len(TEs)),old_protocol.ScanPars[k]]
        else:
            new_protocol.ScanPars[k]=[[0,0,0]]*len(TEs)+old_protocol.ScanPars[k]
    return new_protocol
    
def slice_ps(old_protocol,SlicesToKeep):
    """
    slice_ps(old_protocol,SlicesToKeep)
    Keep a particular part of a protocol and remove other measurements
    
    Parameters
    ----------
    old_protocol : PulseSequences object
        The protocol to select the subset from.
    SlicesToKeep : iterable
        list (or array, etc) of indexes to keep from old_protocol

    Returns
    -------
    new_protocol : PulseSequences object
        New protocol with subset of points from old_protocol
    """
    new_protocol=old_protocol.copy()
    for k,v in new_protocol.ScanPars.items():
        new_protocol.ScanPars[k]=list()
        for sv in SlicesToKeep:
            new_protocol.ScanPars[k].append(old_protocol.ScanPars[k][sv])
        if k!='gDirs':
            new_protocol.ScanPars[k]=np.array(new_protocol.ScanPars[k])
    return new_protocol

def get_sample_PGSE():
    """
    Example PGSE seqence with two different gradient separations and b-values
    from 0-3 ms/um^2 that can be used for testing
    """
    pgpars=dict()
    pgpars['delta']=0.003*np.ones([20],)
    pgpars['DELTA']=np.r_[0.01*np.ones([10],),0.05*np.ones([10],)]
    pgpars['gDirs']=[[0,0,1]]*20
    bVals=np.r_[np.linspace(0,3e9,10),np.linspace(0,3e9,10)]
    pgpars['TE']=np.r_[0.015*np.ones([10],),0.055*np.ones([10],)]
    pgpars['gMag']=from_b_to_g(bVals,pgpars['delta'],pgpars['DELTA'])
    NewSeq=PGSE()
    NewSeq.fill_from_dict(pgpars)
    return NewSeq

if __name__ == '__main__':
    """
    Examples and debugging
    """
    Psq1=get_sample_PGSE()
    print(Psq1.GetB()/1e6)
    Psq2=slice_ps(Psq1,np.r_[2:5])
    print(Psq2.GetB()/1e6)