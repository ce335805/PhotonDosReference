import numpy as np
import scipy.constants as consts

import findAllowedKsSPhP as findAllowedKsSPhP
import epsilonFunctions as epsFunc

def NormSqr(kVal, L, omega, wLO, wTO, epsInf):
    kDArr = epsFunc.kDFromKSurf(kVal, omega, wLO, wTO, epsInf)
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)

    brack11 = np.exp(- kVal * L) * omega ** 2 / (consts.c ** 2 * kVal ** 2)
    brack12 = (omega ** 2 / (consts.c ** 2 * kVal ** 2) + 2) * (1 - np.exp(- 2 * kVal * L)) / (2 * kVal * L)
    term1 = (brack11 + brack12) * 0.25 * (1 + np.exp(-2. * kDArr * L) - 2. * np.exp(-kDArr * L))

    normPrefac = epsFunc.normFac(omega, wLO, wTO, epsInf)
    brack21 = np.exp(- kDArr * L) * eps * omega ** 2 / (consts.c ** 2 * kDArr ** 2)
    brack22 = (eps * omega ** 2 / (consts.c ** 2 * kDArr ** 2) + 2) * (1 - np.exp(- 2 * kDArr * L)) / (2 * kDArr * L)
    term2 = normPrefac * (brack21 + brack22) * 0.25 * (1 + np.exp(-2. * kVal * L) - 2. * np.exp(-kVal * L))

    return L / 4. * (term1 + term2)

def waveFunctionPosPara(zArr, kVal, L, omega, wLO, wTO, epsInf):
    epsAbs = np.abs(epsFunc.epsilon(omega, wLO, wTO, epsInf))
    NSqr = NormSqr(kVal, L, omega, wLO, wTO, epsInf)
    kDVal = epsFunc.kDFromKSurf(kVal, omega, wLO, wTO, epsInf)
    kzDel = findAllowedKsSPhP.findKsDerivativeWSurf(L, omega, wLO, wTO, epsInf)
    kzDelAna =  1. / (consts.c * np.sqrt(epsAbs - 1)) * (1 + epsInf * omega**2 / (epsAbs - 1) * (wLO**2 - wTO**2) / (wTO**2 - omega**2)**2)
    func = 0.25 * (np.exp(- kVal * zArr) - np.exp(kVal * (zArr - L))) * (1 - np.exp(-kDVal * L))
    diffFac = (1. + consts.c ** 2 * kVal / omega * kzDel)
    return 1. / NSqr * func ** 2 * diffFac

def waveFunctionNegPara(zArr, kVal, L, omega, wLO, wTO, epsInf):
    epsAbs = np.abs(epsFunc.epsilon(omega, wLO, wTO, epsInf))
    NSqr = NormSqr(kVal, L, omega, wLO, wTO, epsInf)
    kDVal = epsFunc.kDFromKSurf(kVal, omega, wLO, wTO, epsInf)
    kzDel = findAllowedKsSPhP.findKsDerivativeWSurf(L, omega, wLO, wTO, epsInf)
    func = 0.25 * (np.exp(kDVal * zArr) - np.exp(- kDVal * (L + zArr))) * (1 - np.exp(-kVal * L))
    diffFac = (1. + consts.c ** 2 * kVal / omega * kzDel)
    return 1. / NSqr * func ** 2 * diffFac

def waveFunctionPosPerp(zArr, kVal, L, omega, wLO, wTO, epsInf):
    epsAbs = np.abs(epsFunc.epsilon(omega, wLO, wTO, epsInf))
    NSqr = NormSqr(kVal, L, omega, wLO, wTO, epsInf)
    kDVal = epsFunc.kDFromKSurf(kVal, omega, wLO, wTO, epsInf)
    kzDel = findAllowedKsSPhP.findKsDerivativeWSurf(L, omega, wLO, wTO, epsInf)
    func = 0.25 * np.sqrt(omega ** 2 / (consts.c ** 2 * kVal ** 2) + 1) * (np.exp(- kVal * zArr) + np.exp(kVal * (zArr - L))) * (1 - np.exp(-kDVal * L))
    diffFac = (1. + consts.c ** 2 * kVal / omega * kzDel)
    return 1. / NSqr * func ** 2 * diffFac

def waveFunctionNegPerp(zArr, kVal, L, omega, wLO, wTO, epsInf):
    epsAbs = np.abs(epsFunc.epsilon(omega, wLO, wTO, epsInf))
    NSqr = NormSqr(kVal, L, omega, wLO, wTO, epsInf)
    kDVal = epsFunc.kDFromKSurf(kVal, omega, wLO, wTO, epsInf)
    kzDel = findAllowedKsSPhP.findKsDerivativeWSurf(L, omega, wLO, wTO, epsInf)
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    func = -0.25 * np.sqrt(eps * omega**2 / (consts.c**2 * kDVal**2) + 1) * ( np.exp(kDVal * zArr) +  np.exp(-kDVal * (L + zArr))) * (1 - np.exp(-kVal * L))
    diffFac = (1. + consts.c ** 2 * kVal / omega * kzDel)
    return 1. / NSqr * func ** 2 * diffFac

def calcDosTM(zArr, L, omega, wLO, wTO, epsInf):

    kVal = findAllowedKsSPhP.findKsSurf(L, omega, wLO, wTO, epsInf)

    indNeg = np.where(zArr < 0)
    indPos = np.where(zArr >= 0)
    zPosArr = zArr[indPos]
    zNegArr = zArr[indNeg]

    dosPos = waveFunctionPosPara(zPosArr, kVal, L, omega, wLO, wTO, epsInf)
    dosNeg = waveFunctionNegPara(zNegArr, kVal, L, omega, wLO, wTO, epsInf)
    dosPos += waveFunctionPosPerp(zPosArr, kVal, L, omega, wLO, wTO, epsInf)
    dosNeg += waveFunctionNegPerp(zNegArr, kVal, L, omega, wLO, wTO, epsInf)

    dos = np.pi * consts.c / (2. * omega) * np.append(dosNeg, dosPos)
    return dos

def dosAnalyticalPos(zArr, omega, wLO, wTO, epsInf):
    epsAbs = np.abs(epsFunc.epsilon(omega, wLO, wTO, epsInf))
    normPrefac = epsFunc.normFac(omega, wLO, wTO, epsInf)
    fac1 = np.pi / np.sqrt(1 - 1. / epsAbs) ** 3
    fac2 = np.sqrt(epsAbs) / (epsAbs + normPrefac / epsAbs)
    expFac = np.exp(- 2. * omega * zArr / (consts.c * np.sqrt(epsAbs - 1)))
    diffExtraFac = 1 + 1. / epsInf * omega**2 / (1 - 1. / epsAbs) * (wLO**2 - wTO**2) / (wLO**2 - omega**2)**2
    return fac1 * fac2 * expFac * diffExtraFac

def dosAnalyticalNeg(zArr, omega, wLO, wTO, epsInf):
    epsAbs = np.abs(epsFunc.epsilon(omega, wLO, wTO, epsInf))
    normPrefac = epsFunc.normFac(omega, wLO, wTO, epsInf)
    fac1 = np.pi / np.sqrt(1 - 1. / epsAbs) ** 3
    fac2 = 1. / np.sqrt(epsAbs) / (epsAbs + normPrefac / epsAbs)
    expFac = np.exp(2. * omega * zArr * epsAbs / (consts.c * np.sqrt(epsAbs - 1)))
    diffExtraFac = 1 + 1. / epsInf * omega**2 / (1 - 1. / epsAbs) * (wLO**2 - wTO**2) / (wLO**2 - omega**2)**2
    return fac1 * fac2 * expFac * diffExtraFac

def dosAnalyticalForInt(omega, zVal, wLO, wTO, epsInf):
    epsAbs = np.abs(epsFunc.epsilon(omega, wLO, wTO, epsInf))
    normPrefac = epsFunc.normFac(omega, wLO, wTO, epsInf)
    fac1 = np.pi / np.sqrt(1 - 1. / epsAbs) ** 3
    fac2 = np.sqrt(epsAbs) / (epsAbs + normPrefac / epsAbs)
    expFac = np.exp(- 2. * omega * zVal / (consts.c * np.sqrt(epsAbs - 1)))
    diffExtraFac = 1 + 1. / epsInf * omega**2 / (1 - 1. / epsAbs) * (wLO**2 - wTO**2) / (wLO**2 - omega**2)**2
    return fac1 * fac2 * expFac * diffExtraFac

def calcDosAna(zArr, omega, wLO, wTO, epsInf):

    indNeg = np.where(zArr < 0)
    indPos = np.where(zArr >= 0)
    zPosArr = zArr[indPos]
    zNegArr = zArr[indNeg]

    dosPos = dosAnalyticalPos(zPosArr, omega, wLO, wTO, epsInf)
    dosNeg = dosAnalyticalNeg(zNegArr, omega, wLO, wTO, epsInf)

    dos = np.append(dosNeg, dosPos)
    return dos


def dosSurfAnalyticalPosArr(zArr, omegaArr, wLO, wTO, epsInf):
    epsAbs = np.abs(epsFunc.epsilon(omegaArr, wLO, wTO, epsInf))
    normPrefac = epsFunc.normFac(omegaArr, wLO, wTO, epsInf)
    fac1 = np.pi / np.sqrt(1 - 1. / epsAbs[:, None]) ** 3
    fac2 = np.sqrt(epsAbs[:, None]) / (epsAbs[:, None] + normPrefac[:, None] / epsAbs[:, None])
    expFac = np.exp(- 2. * omegaArr[:, None] * zArr[None, :] / (consts.c * np.sqrt(epsAbs[:, None] - 1)))
    diffExtraFac = 1 + 1. / epsInf * omegaArr[:, None] ** 2 / (1 - 1. / epsAbs[:, None]) * (wLO ** 2 - wTO ** 2) / (
            wLO ** 2 - omegaArr[:, None] ** 2) ** 2
    return fac1 * fac2 * expFac * diffExtraFac
