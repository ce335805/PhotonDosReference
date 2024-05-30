import numpy as np
import scipy.constants as consts

import findAllowedKsSPhP as findAllowedKsSPhP
import epsilonFunctions as epsFunc


def NormSqr(kVal, L, omega, wLO, wTO, epsInf):
    kDVal = epsFunc.kDFromK(kVal, omega, wLO, wTO, epsInf)
    normPrefac = epsFunc.normFac(omega, wLO, wTO, epsInf)
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    brack11 = omega**2 / (consts.c**2 * kVal**2)
    brack12 = (omega**2 / (consts.c**2 * kVal**2) - 2) * np.sin(kVal * L) / (kVal * L)
    term1 = (brack11 + brack12) * np.sin(kDVal * L / 2.)**2
    brack21 = eps * omega**2 / (consts.c**2 * kDVal**2)
    brack22 = (eps * omega**2 / (consts.c**2 * kDVal**2) - 2) * np.sin(kDVal * L) / (kDVal * L)
    term2 = normPrefac * (brack21 + brack22) * np.sin(kVal * L / 2.)**2

    return L / 4. * (term1 + term2)

def waveFunctionPosPara(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kArr, L, omega, wLO, wTO, epsInf)
    kDArr = epsFunc.kDFromK(kArr, omega, wLO, wTO, epsInf)
    func = np.sin(kArr[None, :] * (L / 2 - zArr[:, None])) * np.sin(kDArr[None, :] * L / 2.)
    diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
    return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

def waveFunctionNegPara(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kArr, L, omega, wLO, wTO, epsInf)
    kDArr = epsFunc.kDFromK(kArr, omega, wLO, wTO, epsInf)
    func = np.sin(kDArr[None, :] * (L / 2. + zArr[:, None])) * np.sin(kArr[None, :] * L / 2.)
    diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
    return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

def waveFunctionPosPerp(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kArr, L, omega, wLO, wTO, epsInf)
    kDArr = epsFunc.kDFromK(kArr, omega, wLO, wTO, epsInf)
    func = np.sqrt(omega ** 2 / (consts.c ** 2 * kArr[None, :] ** 2) - 1) * np.cos(kArr[None, :] * (L / 2 - zArr[:, None])) * np.sin(kDArr[None, :] * L / 2.)
    diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
    return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

def waveFunctionNegPerp(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
    NSqr = NormSqr(kArr, L, omega, wLO, wTO, epsInf)
    kDArr = epsFunc.kDFromK(kArr, omega, wLO, wTO, epsInf)
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    func = - np.sqrt(eps * omega**2 / (consts.c**2 * kDArr[None, :]**2) - 1) * np.cos(kDArr[None, :] * (L / 2. + zArr[:, None])) * np.sin(kArr[None, :] * L / 2.)
    diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
    return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

def calcDosTM(zArr, L, omega, wLO, wTO, epsInf):

    kArr = findAllowedKsSPhP.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TM")
    kzArrDel = findAllowedKsSPhP.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TM")

    if(len(kArr) == 0):
        return 0

    indNeg = np.where(zArr < 0)
    indPos = np.where(zArr >= 0)
    zPosArr = zArr[indPos]
    zNegArr = zArr[indNeg]

    dosPos = waveFunctionPosPara(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    dosNeg = waveFunctionNegPara(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    dosPos += waveFunctionPosPerp(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    dosNeg += waveFunctionNegPerp(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    dos = np.pi * consts.c / (2. * omega) * np.append(dosNeg, dosPos)
    return dos

def calcDosTMParaPerp(zArr, L, omega, wLO, wTO, epsInf):

    kArr = findAllowedKsSPhP.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TM")
    kzArrDel = findAllowedKsSPhP.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TM")

    if(len(kArr) == 0):
        return 0

    indNeg = np.where(zArr < 0)
    indPos = np.where(zArr >= 0)
    zPosArr = zArr[indPos]
    zNegArr = zArr[indNeg]

    dosPosPara = waveFunctionPosPara(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    dosNegPara = waveFunctionNegPara(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    dosPosPerp = waveFunctionPosPerp(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    dosNegPerp = waveFunctionNegPerp(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    dosPara = np.pi * consts.c / (2. * omega) * np.append(dosNegPara, dosPosPara)
    dosPerp = np.pi * consts.c / (2. * omega) * np.append(dosNegPerp, dosPosPerp)
    return (dosPara, dosPerp)

