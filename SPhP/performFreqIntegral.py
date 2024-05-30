import numpy as np
import scipy.constants as consts
import scipy.integrate

import epsilonFunctions as epsFunc
from dosFuncs import dosTMSurfModes as dosTMSurf
import produceFreqData as prod

def freqIntegral(wArrSubdivisions, zArr, wLO, wTO, epsInf, L):

    #evCutoff = 1519.3 * 1e12 # 1eV
    cutoff = 100 * 1e12
    computeEffectiveMass(wArrSubdivisions, zArr, cutoff, wLO, wTO, epsInf, L)

def computeEffectiveMass(wArrSubdivisions, zArr, cutoff, wLO, wTO, epsInf, L):

    dosTETotal, dosTMPara = prod.retrieveDosPara(wArrSubdivisions, zArr, wLO, wTO, epsInf, L)
    wArr = prod.defineFreqArrayOne(wArrSubdivisions)

    dosIntTE = np.zeros(zArr.shape)
    dosIntTM = np.zeros(zArr.shape)

    print("Lambda Low = {} THz".format(wArr[1] * 1e-9))

    for zInd, zVal in enumerate(zArr):
        prefacMass = 3. / 2. * 4. / (3. * np.pi) * consts.fine_structure * consts.hbar / (consts.c**2 * consts.m_e)
        intFuncTE = prefacMass * (dosTETotal[ : , zInd] - dosTETotal[ : , 0])
        dosIntTE[zInd] = np.trapz(intFuncTE, x=wArr, axis = 0)
        intFuncTM = prefacMass * (dosTMPara[ : , zInd] - dosTMPara[ : , 0])
        dosIntTM[zInd] = np.trapz(intFuncTM, x=wArr, axis = 0)

    dosSurf = np.zeros(len(zArr))
    for zInd, zVal in enumerate(zArr):
        dosSurf[zInd] = performSPhPIntegralMass(zVal, wLO, wTO, epsInf)[0]

    dosTot = dosIntTE + dosIntTM + dosSurf
    dosBulk = dosIntTE + dosIntTM

    wLOStr = "wLO" + str(int(wLO * 1e-12))
    wTOStr = "wTO" + str(int(wTO * 1e-12))
    filename = "./savedData/massSPhP" + wLOStr + wTOStr + ".hdf5"
    print("Writing masses to file: " + filename)
    prod.writeMasses(cutoff, zArr, dosTot, dosBulk, filename)

def intFuncMass(omega, zVal, wLO, wTO, epsInf):
    epsilon = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    prefacPara = 1. / (1. + np.abs(epsilon))
    prefacMass = 3. / 2. * 4. / (3. * np.pi) * consts.fine_structure * consts.hbar / (consts.c ** 2 * consts.m_e)
    return prefacPara * prefacMass * dosTMSurf.dosAnalyticalForInt(omega, zVal, wLO, wTO, epsInf)

def performSPhPIntegralMass(zVal, wLO, wTO, epsInf):
    wInf = np.sqrt(epsInf * wLO**2 + wTO**2) / np.sqrt(epsInf + 1)
    return scipy.integrate.quad(intFuncMass, wTO, wInf, args=(zVal, wLO, wTO, epsInf), points=[wInf, wInf - wInf * 1e-5, wInf - wInf * 1e-4, wInf - wInf * 1e-3], limit = 10000)

def performSPhPIntegralAna(zVal, wLO, wTO, epsInf):
    wInf = np.sqrt(wLO**2 + wTO**2) / np.sqrt(2)
    prefacField = consts.hbar * wInf**1 / (2. * consts.epsilon_0 * np.pi**2 * consts.c**3) * 1e24
    rho0Prefac = np.pi ** 2 * consts.c **3 / wInf**2
    rho = 1. / (8. * np.pi) * (wLO**2 - wTO**2) / wLO**2 / zVal**3
    return prefacField * rho0Prefac * rho


def patchDosSurfWithZeros(dosSurf, zArr, arrBelow, arrAboveClose):
    patchBelow = np.zeros((len(arrBelow), len(zArr)))
    patchAbove = np.zeros((len(arrAboveClose), len(zArr)))
    dosSurf = np.append(patchBelow, dosSurf, axis = 0)
    dosSurf = np.append(dosSurf, patchAbove, axis = 0)
    return dosSurf

