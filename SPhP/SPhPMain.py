import numpy as np
import sys
import produceFreqData
import performFreqIntegral
import scipy.constants as consts

def main():
    print("Compute full Dos and all modes")


    wSubArrInd = 0
    if len(sys.argv) > 1:
        try:
            wSubArrInd = int(sys.argv[1])
            print("Integer value passed:", wSubArrInd)
        except ValueError:
            print("Error -- no valid integer was passed.")


    #Dividing the frequency array in smaller subarrays to mitigate memory peaks
    wArrSubdivisions = 5

    epsInf = 1.
    wLO = 32.04 * 1e12 #STO
    wTO = 7.92 * 1e12 #STO
    #wTO = 1e6 #Something that was used for the Drude metal instead of 0 (doesn't affect due to lower cutoff / finite losses)
    L = 0.01#length of the outer box in meter
    wInf = np.sqrt(epsInf * wLO ** 2 + wTO ** 2) / np.sqrt(epsInf + 1)
    lambda0 = 2. * np.pi * consts.c / wInf
    zArr = np.logspace(np.log10(1e1 * lambda0), np.log10(1e-5 * lambda0), 50, endpoint=True, base = 10)
    zArr = np.append([L / 4.], zArr) #append on value that is far away as reference

    #produceFreqData.produceFreqData(wSubArrInd, wArrSubdivisions, zArr, wLO, wTO, epsInf, L)
    performFreqIntegral.freqIntegral(wArrSubdivisions, zArr, wLO, wTO, epsInf, L)

if __name__ == "__main__":
    main()