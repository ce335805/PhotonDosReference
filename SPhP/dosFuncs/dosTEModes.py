import numpy as np
import matplotlib as mpl
import scipy.constants as consts

import findAllowedKsSPhP as findAllowedKsSPhP
import epsilonFunctions as epsFunc

def NormSqr(kArr, L, omega, wLO, wTO, epsInf):
    kDVal = epsFunc.kDFromK(kArr, omega, wLO, wTO, epsInf)
    normPrefac = epsFunc.normFac(omega, wLO, wTO, epsInf)
    term1 = (1 - np.sin(kArr * L) / (kArr * L)) * np.sin(kDVal * L / 2) ** 2
    term2 = normPrefac * (1 - np.sin(kDVal * L) / (kDVal * L)) * np.sin(kArr * L / 2) ** 2
    return L / 4 * (term1 + term2)

def dosSumPos(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kArr, L, omega, wLO, wTO, epsInf)
    kDArr = epsFunc.kDFromK(kArr, omega, wLO, wTO, epsInf)
    func = np.sin(kArr[None, :] * (L / 2 - zArr[:, None])) * np.sin(kDArr[None, :] * L / 2.)
    diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
    return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

def dosSumNeg(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kArr, L, omega, wLO, wTO, epsInf)
    kDArr = epsFunc.kDFromK(kArr, omega, wLO, wTO, epsInf)
    func = np.sin(kDArr[None, :] * (L / 2. + zArr[:, None])) * np.sin(kArr[None, :] * L / 2.)
    diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
    #diffFac = 1.
    return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

def calcDosTE(zArr, L, omega, wLO, wTO, epsInf):

    kArr = findAllowedKsSPhP.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TE")
    kzArrDel = findAllowedKsSPhP.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TE")

    indNeg = np.where(zArr < 0)
    indPos = np.where(zArr >= 0)
    zPosArr = zArr[indPos]
    zNegArr = zArr[indNeg]

    dosPos = dosSumPos(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    dosNeg = dosSumNeg(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)

    dos = np.pi * consts.c / (2. * omega) * np.append(dosNeg, dosPos)
    return dos



