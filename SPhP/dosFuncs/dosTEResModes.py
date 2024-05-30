import numpy as np
import scipy.constants as consts

import findAllowedKsSPhP as findAllowedKsSPhP
import epsilonFunctions as epsFunc

def NormSqr(kArr, L, omega, wLO, wTO, epsInf):
    kDArr = epsFunc.kDFromKRes(kArr, omega, wLO, wTO, epsInf)
    normPrefac = epsFunc.normFac(omega, wLO, wTO, epsInf)
    brack1 = 1 - np.sin(kArr * L) / (kArr * L)
    term1 = brack1 * 0.25 * (1 + np.exp(-2 * kDArr * L) - 2. * np.exp(- kDArr * L))
    brack2 = (1 - np.exp(- 2. * kDArr * L)) / (2. * kDArr * L) - np.exp(- kDArr * L)
    term2 = normPrefac * brack2 * np.sin(kArr * L / 2) ** 2
    return L / 4 * (term1 + term2)

def dosSumPos(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kArr, L, omega, wLO, wTO, epsInf)
    #kzArrDel = findAllowedKsSPhP.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TERes")
    kDArr = epsFunc.kDFromKRes(kArr, omega, wLO, wTO, epsInf)
    func = np.sin(kArr[None, :] * (L / 2. - zArr[:, None])) * 0.5 * (1 - np.exp(- kDArr[None, :] * L))
    diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
    return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

def dosSumNeg(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kArr, L, omega, wLO, wTO, epsInf)
    #kzArrDel = findAllowedKsSPhP.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TERes")
    kDArr = epsFunc.kDFromKRes(kArr, omega, wLO, wTO, epsInf)
    func = 0.5 * (np.exp(kDArr[None, :] * zArr[:, None]) - np.exp(-kDArr[None, :] * zArr[:, None] - kDArr[None, :] * L)) * np.sin(kArr[None, :] * L / 2.)
    diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
    return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)


def calcDosTE(zArr, L, omega, wLO, wTO, epsInf):

    kArr = findAllowedKsSPhP.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TERes")
    kzArrDel = findAllowedKsSPhP.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TERes")

    indNeg = np.where(zArr < 0)
    indPos = np.where(zArr >= 0)
    zPosArr = zArr[indPos]
    zNegArr = zArr[indNeg]

    dosPos = dosSumPos(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    dosNeg = dosSumNeg(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)

    dos = np.pi * consts.c / (2. * omega) * np.append(dosNeg, dosPos)
    return dos


