from itertools import repeat
from multiprocessing import Pool
from time import perf_counter

import dosAsOfFreq as dosAsOfFreq
import h5py
import numpy as np

def defineFreqArray(wArrSubdivisons):
    wBot = 50 * 1e9 # Lower Cutoff
    wTop = 100. * 1e12 # Frequencies are in Hz
    nW = 200
    fullArr = np.linspace(wBot, wTop, nW)

    part_sizes = [len(fullArr) // wArrSubdivisons] * wArrSubdivisons
    remaining = len(fullArr) - sum(part_sizes)
    part_sizes[0] += remaining
    split_arrays = [fullArr[sum(part_sizes[:i]):sum(part_sizes[:i+1])] for i in range(wArrSubdivisons)]

    return np.array(split_arrays)

def defineFreqArrayOne(wArrSubdivisions):
    wArrs = defineFreqArray(wArrSubdivisions)
    wArrOne = np.zeros(0, dtype=float)
    for wArrInd, wArr in enumerate(wArrs):
        wArrOne = np.append(wArrOne, wArr)
    return wArrOne

def parameterName(wLO, wTO, wBot, wTop, nW, L, eps):
    wLOStr = "wLO" + str(int(wLO * 1e-12))
    wTOStr = "wTO" + str(int(wTO * 1e-12))
    wBotStr = "wBot" + str(int(wBot * 1e-12))
    wTopStr = "wTop" + str(int(wTop * 1e-12))
    nWStr = "nW" + str(int(nW))
    ellStr = "L" + str(int(10 * L))
    epsStr = "Eps" + str(int(10 * eps))
    return ellStr + epsStr + wLOStr + wTOStr + wBotStr + wTopStr + nWStr

def produceFreqData(wSubArrInd, wArrSubdivisions, zArr, wLO, wTO, epsInf, L):
    wArrs = defineFreqArray(wArrSubdivisions)
    t_start = perf_counter()
    computeFreqData(wArrs[wSubArrInd], zArr, wLO, wTO, epsInf, L)
    t_stop = perf_counter()
    print("Time to produce frequency data: {}".format(t_stop - t_start))

def computeFreqData(wArr, zArr, wLO, wTO, epsInf, L):
    wBot = wArr[0]
    wTop = wArr[-1]
    nW = len(wArr)
    parameterStr = parameterName(wLO, wTO, wBot, wTop, nW, L, epsInf)

    filenameTE = 'savedData/dosTE' + parameterStr + '.h5'
    produceFreqDataTE(wArr, zArr, L, wLO, wTO, epsInf, filenameTE)
    filenameTM = 'savedData/dosTM' + parameterStr + '.h5'
    produceFreqDataTM(wArr, zArr, L, wLO, wTO, epsInf, filenameTM)

def produceFreqDataTE(omegaArr, zArr, L, wLO, wTO, epsInf, filename):

    #with Pool() as pool:
    #    dosTEVals = np.array(pool.starmap(dosAsOfFreq.getDosTE, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
    #    dosTEEvaVals = np.array(pool.starmap(dosAsOfFreq.getDosTEEva, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
    #    dosTEResVals = np.array(pool.starmap(dosAsOfFreq.getDosTERes, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))

    dosTEVals = np.zeros((len(omegaArr), len(zArr)))
    dosTEEvaVals = np.zeros((len(omegaArr), len(zArr)))
    dosTEResVals = np.zeros((len(omegaArr), len(zArr)))
    for omegaInd, omegaVal in enumerate(omegaArr):
       dosTEVals[omegaInd, :] = dosAsOfFreq.getDosTE(omegaVal, zArr, L, wLO, wTO, epsInf)
       dosTEEvaVals[omegaInd, :] = dosAsOfFreq.getDosTEEva(omegaVal, zArr, L, wLO, wTO, epsInf)
       dosTEResVals[omegaInd, :] = dosAsOfFreq.getDosTERes(omegaVal, zArr, L, wLO, wTO, epsInf)

    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('dosTE', data=dosTEVals)
    h5f.create_dataset('dosTEEva', data=dosTEEvaVals)
    h5f.create_dataset('dosTERes', data=dosTEResVals)
    h5f.close()

def produceFreqDataTM(omegaArr, zArr, L, wLO, wTO, epsInf, filename):

    #with Pool() as pool:

    #    retTM = np.array(pool.starmap(dosAsOfFreq.getDosTMParaPerp, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
    #    retTMEva = np.array(pool.starmap(dosAsOfFreq.getDosTMEvaParaPerp, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))
    #    retTMRes = np.array(pool.starmap(dosAsOfFreq.getDosTMResParaPerp, zip(omegaArr, repeat(zArr), repeat(L), repeat(wLO), repeat(wTO), repeat(epsInf))))

    #dosTMValsPara = retTM[:, 0, :]
    #dosTMValsPerp = retTM[:, 1, :]
    #dosTMEvaValsPara = retTMEva[:, 0, :]
    #dosTMEvaValsPerp = retTMEva[:, 1, :]
    #dosTMResValsPara = retTMRes[:, 0, :]
    #dosTMResValsPerp = retTMRes[:, 1, :]

    dosTMValsPara = np.zeros((len(omegaArr), len(zArr)))
    dosTMValsPerp = np.zeros((len(omegaArr), len(zArr)))
    dosTMEvaValsPara = np.zeros((len(omegaArr), len(zArr)))
    dosTMEvaValsPerp = np.zeros((len(omegaArr), len(zArr)))
    dosTMResValsPara = np.zeros((len(omegaArr), len(zArr)))
    dosTMResValsPerp = np.zeros((len(omegaArr), len(zArr)))
    for omegaInd, omegaVal in enumerate(omegaArr):
       (dosTMValsPara[omegaInd, :], dosTMValsPerp[omegaInd, :]) = dosAsOfFreq.getDosTMParaPerp(omegaVal, zArr, L, wLO, wTO, epsInf)
       (dosTMEvaValsPara[omegaInd, :], dosTMEvaValsPerp[omegaInd, :]) = dosAsOfFreq.getDosTMEvaParaPerp(omegaVal, zArr, L, wLO, wTO, epsInf)
       (dosTMResValsPara[omegaInd, :], dosTMResValsPerp[omegaInd, :]) = dosAsOfFreq.getDosTMResParaPerp(omegaVal, zArr, L, wLO, wTO, epsInf)

    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('dosTMPara', data=dosTMValsPara)
    h5f.create_dataset('dosTMEvaPara', data=dosTMEvaValsPara)
    h5f.create_dataset('dosTMResPara', data=dosTMResValsPara)
    h5f.create_dataset('dosTMPerp', data=dosTMValsPerp)
    h5f.create_dataset('dosTMEvaPerp', data=dosTMEvaValsPerp)
    h5f.create_dataset('dosTMResPerp', data=dosTMResValsPerp)
    h5f.close()

def retrieveDosPara(wArrSubdivisions, zArr, wLO, wTO, epsInf, L):
    wArrs = defineFreqArray(wArrSubdivisions)

    dosTETotal = np.zeros((0, len(zArr)), dtype=float)
    dosTMPara = np.zeros((0, len(zArr)), dtype=float)
    for wArrInd, wArr in enumerate(wArrs):
        dosTETotal = np.append(dosTETotal, retrieveDosTE(wArr, L, wLO, wTO, epsInf), axis = 0)
        dosTMPara = np.append(dosTMPara, retrieveDosTMPara(wArr, L, wLO, wTO, epsInf), axis = 0)

    return (dosTETotal, dosTMPara)

def retrieveDosTE(wArr, L, wLO, wTO, epsInf):
    wBot = wArr[0]
    wTop = wArr[-1]
    nW = len(wArr)
    parameterStr = parameterName(wLO, wTO, wBot, wTop, nW, L, epsInf)

    dir = "savedData/"

    filenameTE = dir + 'dosTE' + parameterStr + '.h5'
    print("Read file: " + filenameTE)
    h5f = h5py.File(filenameTE, 'r')
    dosTETotal = h5f['dosTE'][:] + h5f['dosTEEva'][:] + h5f['dosTERes'][:]
    h5f.close()

    return dosTETotal

def retrieveDosTMTotal(wArr, L, wLO, wTO, epsInf):
    wBot = wArr[0]
    wTop = wArr[-1]
    nW = len(wArr)
    parameterStr = parameterName(wLO, wTO, wBot, wTop, nW, L, epsInf)

    dir = "savedData/"

    filenameTM = dir + 'dosTM' + parameterStr + '.h5'
    h5f = h5py.File(filenameTM, 'r')
    dosTMTotal = h5f['dosTMPara'][:] + h5f['dosTMEvaPara'][:] + h5f['dosTMResPara'][:] + h5f['dosTMPerp'][:] + h5f['dosTMEvaPerp'][:] + h5f['dosTMResPerp'][:]
    h5f.close()

    return dosTMTotal


def retrieveDosTMPara(wArr, L, wLO, wTO, epsInf):
    wBot = wArr[0]
    wTop = wArr[-1]
    nW = len(wArr)
    parameterStr = parameterName(wLO, wTO, wBot, wTop, nW, L, epsInf)

    dir = "savedData/"

    filenameTM = dir + 'dosTM' + parameterStr + '.h5'
    h5f = h5py.File(filenameTM, 'r')
    dosTMPara = h5f['dosTMPara'][:] + h5f['dosTMEvaPara'][:] + h5f['dosTMResPara'][:]
    h5f.close()

    return dosTMPara

def writeMasses(cutoff, zArr, mArrTot, mArrBulk, filename):
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('cutoff', data=np.array([cutoff]))
    h5f.create_dataset('zArr', data=zArr)
    h5f.create_dataset('delMTot', data=mArrTot)
    h5f.create_dataset('delMBulk', data=mArrBulk)
    h5f.close()

def readMasses(filename):
    h5f = h5py.File(filename, 'r')
    cutoff = h5f['cutoff'][:]
    mArrTot = h5f['delMTot'][:]
    mArrBulk = h5f['delMBulk'][:]
    zArr = h5f['zArr'][:]
    h5f.close()
    return (cutoff, zArr, mArrTot, mArrBulk)
