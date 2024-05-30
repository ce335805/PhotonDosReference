import numpy as np
import h5py

import epsilonFunctions as epsFunc

from dosFuncs import dosTEModes as dosTE
from dosFuncs import dosTEEvaModes as dosTEEva
from dosFuncs import dosTEResModes as dosTERes
from dosFuncs import dosTMModes as dosTM
from dosFuncs import dosTMEvaModes as dosTMEva
from dosFuncs import dosTMResModes as dosTMRes
from dosFuncs import dosTMSurfModes as dosTMSurf



def getDosTE(omega, zArr, L, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    if(eps > 0):
        return dosTE.calcDosTE(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return np.zeros(len(zArr))

def getDosTEEva(omega, zArr, L, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    if( eps > 1 ):
        return dosTEEva.calcDosTE(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return np.zeros(len(zArr))

def getDosTERes(omega, zArr, L, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    if(eps < 1.):
        return dosTERes.calcDosTE(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return np.zeros(len(zArr))

def getDosTM(omega, zArr, L, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    if(eps > 0):
        return dosTM.calcDosTM(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return np.zeros(len(zArr))

def getDosTMParaPerp(omega, zArr, L, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    if(eps > 0):
        return dosTM.calcDosTMParaPerp(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return (np.zeros(len(zArr)), np.zeros(len(zArr)))


def getDosTMEva(omega, zArr, L, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    if( eps > 1 ):
        return dosTMEva.calcDosTM(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return np.zeros(len(zArr))


def getDosTMEvaParaPerp(omega, zArr, L, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    if( eps > 1 ):
        return dosTMEva.calcDosTMParaPerp(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return (np.zeros(len(zArr)), np.zeros(len(zArr)))


def getDosTMRes(omega, zArr, L, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    if(eps < 1.):
        return dosTMRes.calcDosTM(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return np.zeros(len(zArr))


def getDosTMResParaPerp(omega, zArr, L, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    if(eps < 1.):
        return dosTMRes.calcDosTMParaPerp(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return (np.zeros(len(zArr)), np.zeros(len(zArr)))

def getDosTMSurf(omega, zArr, L, wLO, wTO, epsInf):
    wInf = np.sqrt(epsInf * wLO**2 + wTO**2) / np.sqrt(epsInf + 1)
    if(omega < wInf and omega > wTO):
        return dosTMSurf.calcDosTM(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return np.zeros(len(zArr))


def produceFreqDataSurf(omegaArr, zArr, L, wLO, wTO, epsInf, filename):

    dosTMSurfVals = getDosTMSurf(zArr, L, omegaArr, wLO, wTO, epsInf)

    dosSurf = dosTMSurfVals
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('dosTM', data=dosSurf)
    h5f.close()


