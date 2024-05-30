import numpy as np
import scipy.constants as consts

import findAllowedKsSPhP as findAllowedKsSPhP
import epsilonFunctions as epsFunc

def NormSqr(kVal, L, omega, wLO, wTO, epsInf):
    kDVal = epsFunc.kDFromKEva(kVal, omega, wLO, wTO, epsInf)
    normPrefac = epsFunc.normFac(omega, wLO, wTO, epsInf)
    brack1 = (1 - np.exp(- 2. * kVal * L)) / (2. * kVal * L) - np.exp(- kVal * L)
    term1 = brack1 * np.sin(kDVal * L / 2) ** 2
    brack2 = 1 - np.sin(kDVal * L) / (kDVal * L)
    term2 = normPrefac * brack2 * 0.25 * (1 + np.exp(-2 * kVal * L) - 2. * np.exp(- kVal * L))
    return L / 4 * (term1 + term2)

def dosSumPos(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kArr, L, omega, wLO, wTO, epsInf)
    #kzArrDel = findAllowedKsSPhP.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TEEva")
    kDArr = epsFunc.kDFromKEva(kArr, omega, wLO, wTO, epsInf)
    func = 0.5 * (np.exp(- kArr[None, :] * zArr[:, None]) - np.exp(kArr * zArr[:, None] - kArr[None, :] * L)) * np.sin(kDArr[None, :] * L / 2.)
    diffFac = (1. + consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
    return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

def dosSumNeg(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kArr, L, omega, wLO, wTO, epsInf)
    #kzArrDel = findAllowedKsSPhP.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TEEva")
    kDArr = epsFunc.kDFromKEva(kArr, omega, wLO, wTO, epsInf)
    func = np.sin(kDArr[None, :] * (L / 2. + zArr[:, None])) * 0.5 * (1 - np.exp(- kArr[None, :] * L))
    diffFac = (1. + consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
    return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

def calcDosTE(zArr, L, omega, wLO, wTO, epsInf):

    kArr = findAllowedKsSPhP.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TEEva")
    kzArrDel = findAllowedKsSPhP.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TEEva")

    indNeg = np.where(zArr < 0)
    indPos = np.where(zArr >= 0)
    zPosArr = zArr[indPos]
    zNegArr = zArr[indNeg]

    dosPos = dosSumPos(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    dosNeg = dosSumNeg(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)

    dos = np.pi * consts.c / (2. * omega) * np.append(dosNeg, dosPos)
    return dos

