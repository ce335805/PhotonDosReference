import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import quad

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors
import h5py
from matplotlib import gridspec
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches

import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import epsilonFunctions as epsFunc
import dosAsOfFreq

import produceFreqData as prodV2

import scipy.constants as consts

fontsize = 8

mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['font.size'] = 8  # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['xtick.major.width'] = .5
mpl.rcParams['ytick.major.width'] = .5
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['figure.titlesize'] = 8
mpl.rc('text', usetex=True)

mpl.rcParams['text.latex.preamble'] = [
    #    r'\renewcommand{\familydefault}{\sfdefault}',
    #    r'\usepackage[scaled=1]{helvet}',
    r'\usepackage[helvet]{sfmath}',
    #    r'\everymath={\sf}'
]

def readMassesSPhP(filename):

    dir = "./savedData/"
    fileName = dir + filename + ".hdf5"

    h5f = h5py.File(fileName, 'r')
    cutoff = h5f['cutoff'][:]
    mArrTot = h5f['delMTot'][:]
    mArrBulk = h5f['delMBulk'][:]
    zArr = h5f['zArr'][:]
    h5f.close()
    return (cutoff, zArr, mArrTot, mArrBulk)


def Fig1():

    wLO = 32.04 * 1e12
    wTO = 7.92 * 1e12

    wLOStr = "wLO" + str(int(wLO * 1e-12))
    wTOStr = "wTO" + str(int(wTO * 1e-12))
    filenameSPhP = "massSPhP" + wLOStr + wTOStr
    cutoff, zArr, massSPhPArr, _ = readMassesSPhP(filenameSPhP)

    fig = plt.figure(figsize=(3., 1.8), dpi=400)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.0, hspace=0.3, top=0.95, bottom=0.14, left=0.19, right=0.98)
    axSPhP = plt.subplot(gs[0, 0])
    axSPhPUpperAx = axSPhP.twiny()

    wInf = np.sqrt(wLO ** 2 + wTO ** 2) / np.sqrt(2)
    lambda0 = 2. * np.pi * consts.c / wInf

    cmapPink = cm.get_cmap('pink')

    axSPhP.plot(zArr / lambda0, massSPhPArr, color=cmapPink(.43), lw=1.2)
    axSPhPUpperAx.plot(zArr, massSPhPArr, color=cmapPink(.43), lw=1.2)
    axSPhP.axhline(0.1, color = "black", lw = 0.3)
    axSPhP.set_xscale("log")
    axSPhPUpperAx.set_xscale("log")
    zmin = 0.7 * 1e-4
    zmax = 2.5 * 1e-2
    ymin = -0.0025
    ymax = 0.3
    axSPhP.set_xlim(zmin, zmax)
    axSPhP.set_ylim(ymin, ymax)
    axSPhPUpperAx.set_ylim(ymin, ymax)
    axSPhPUpperAx.set_xlim(zmin * lambda0, zmax * lambda0)

    cmapCustom = LinearSegmentedColormap.from_list('', ['white', '#bcd4e3'])
    imN = 50
    X = np.zeros((imN, imN), dtype=float)
    X[:, 0] = 1.
    X[:, 1] = .7
    X[:, 2] = .5
    X[:, 3] = .3
    X[:, 4] = .2
    X[:, 5] = .1
    X[:, 6] = .08
    X[:, 7] = .05
    X[:, 8] = .04
    X[:, 9] = .03
    X[:, 10] = .02
    X[:, 11] = .01
    X[:, 12] = .005
    X[:, 13] = .0025
    X[:, 14] = .001
    X[:, 15] = .0005
    X[:, 16] = .00025
    X[:, 17] = .0001

    axSPhP.imshow(X, interpolation='bicubic', cmap=cmapCustom,
                  extent=(zmin, 2 * 1e-3, ymin, ymax), alpha=1.)

    axSPhP.set_xlabel("$z [\lambda_0]$", fontsize=8)
    axSPhP.set_ylabel(r"$\Delta m \, [m_{\rm e}]$", labelpad=0., fontsize=8)

    axSPhP.set_xticks([1e-4, 1e-2])
    axSPhP.set_xticklabels(["$10^{-4}$", "$10^{-2}$"], fontsize=8)

    axSPhPUpperAx.set_xticks([1e-8, 1e-6])
    axSPhPUpperAx.set_xticklabels(["$10 \mathrm{nm}$", "$1 \mu \mathrm{m}$"], fontsize=8)

    axSPhP.set_yticks([0., 0.1, 0.3])
    axSPhP.set_yticklabels(["$0$", "$0.1$", "$0.3$"], fontsize=8)

    axSPhP.text(.5, .37, r"$\mathrm{Ultra{-}Strong \, Coupling}$", fontsize=8, transform=axSPhP.transAxes,
                zorder=666)

    axComic = plt.subplot(gs[0, 0])

    axComic.text(-.23, 1., r"$\mathrm{(a)}$", transform=axComic.transAxes, fontsize=8)
    axSPhP.text(-.23, 1.12, r"$\mathrm{(b)}$", transform=axSPhP.transAxes, fontsize=8)


    for axis in ['top', 'bottom', 'left', 'right']:
        axSPhP.spines[axis].set_linewidth(.5)
        axSPhPUpperAx.spines[axis].set_linewidth(.5)
        axComic.spines[axis].set_linewidth(.0)


    plt.savefig("./savedPlots/Fig1.png")


def createFig2Plot():

    wArrSubdivisions = 100

    epsInf = 1.
    #numbers for sto
    wLO = 32.04 * 1e12
    wTO = 7.92 * 1e12
    L = 1.
    wInf = np.sqrt(epsInf * wLO ** 2 + wTO ** 2) / np.sqrt(epsInf + 1)
    lambda0 = 2. * np.pi * consts.c / wInf

    zArr = np.logspace(np.log10(1e2 * lambda0), np.log10(1e-5 * lambda0), 200, endpoint=True, base = 10)
    zArr = np.append([L / 4.], zArr)

    wArrSPhP, dosSPhPBulk, dosSPhPSurf = getDosDataSPhP(wArrSubdivisions, zArr, wLO, wTO, epsInf, L)

    zArrM, mArr, mBulkArr = getMass(wLO, wTO)

    NewFig2(zArr, wArrSPhP, dosSPhPBulk + dosSPhPSurf, zArrM, mArr, mBulkArr, wLO, wTO)


def getDosDataSPhP(wArrSubdivisions, zArr, wLO, wTO, epsInf, L):

    dosTETotal, dosTMPara = prodV2.retrieveDosPara(
        wArrSubdivisions,
        zArr,
        wLO,
        wTO,
        epsInf,
        L)

    wArrOne = prodV2.defineFreqArrayOne(wArrSubdivisions)
    dosSurf = np.zeros((len(wArrOne), len(zArr)))
    for wInd, wVal in enumerate(wArrOne):
        epsilon = epsFunc.epsilon(wVal, wLO, wTO, epsInf)
        dosSurf[wInd, :] = 1. / (1. + np.abs(epsilon)) * dosAsOfFreq.getDosTMSurf(wVal, zArr, L, wLO, wTO, epsInf)

    return (wArrOne, dosTETotal + dosTMPara, dosSurf)

def getMass(wLO, wTO):

    wLOStr = "wLO" + str(int(wLO * 1e-12))
    wTOStr = "wTO" + str(int(wTO * 1e-12))
    #filename = "./savedData/PaperData/massForPaperNew" + wLOStr + wTOStr + ".hdf5"
    filename = "./savedData/PaperData/massForPaperLowerCutoff50GHzSPhP" + wLOStr + wTOStr + ".hdf5"
    h5f = h5py.File(filename, 'r')
    mArr = h5f['delM'][:]
    mBulkArr = h5f['delMBulk'][:]
    zArr = h5f['zArr'][:]
    h5f.close()

    return (zArr, mArr, mBulkArr)

def NewFig2(zArr, wArrSPhP, dosSPhP, zArrM, mArr, mBulkArr, wLO, wTO):
    wInf = np.sqrt(wLO ** 2 + wTO ** 2) / np.sqrt(1 + 1)
    lambda0 = 2. * np.pi * consts.c / wInf

    fig = plt.figure(figsize=(7., 1.5), dpi=400)
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 3],
                           wspace=0.27, hspace=0., top=0.9, bottom=0.225, left=0.05, right=0.99)

    axSPhP = plt.subplot(gs[0, 0])
    axM = plt.subplot(gs[0, 1])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    indArr = np.array([1, 30, 60, 92], dtype=int)
    #indArr = np.array([1, 30, 60, 80, 180], dtype=int)

    # axSPhP.plot(wArr, dos[:, indArr[0]], color=cmapBone(0.1), lw=.7, label="$z = $" + "{:1.0f}".format(zArr[indArr[0]] / lambda0) + r"$\lambda_0$")
    axSPhP.plot(wArrSPhP[0:], dosSPhP[0:, indArr[1]], color=cmapPink(0.1), lw=1.2,
            label="$z = $" + "{:1.0f}".format(zArr[indArr[1]] / lambda0) + r"$\lambda_0$")
    axSPhP.plot(wArrSPhP[0:], dosSPhP[0:, indArr[2]], color=cmapPink(0.4), lw=1.2,
            label="$z = $" + "{:1.0f}".format(zArr[indArr[2]] / lambda0) + r"$\lambda_0$")
    axSPhP.plot(wArrSPhP[0:], dosSPhP[0:, indArr[3]], color=cmapPink(0.6), lw=1.2,
            label="$z = $" + "{:1.1f}".format(zArr[indArr[3]] / lambda0) + r"$\lambda_0$")
    #
    axSPhP.axhline(2. / 3., lw = 0.5, color = 'gray', zorder = -666)

    axSPhP.set_xlim(0, 40 * 1e12)
    axSPhP.set_ylim(0., 5.)


    axSPhP.set_xticks([0., wTO, wInf, wLO])
    axSPhP.set_xticklabels([r"$0$", r"$\omega_{\rm TO}$", r"$\omega_{\infty}$", r"$\omega_{\rm LO}$"], fontsize=8)
    axSPhP.set_yticks([0., 2. / 3., 2])
    axSPhP.set_yticklabels([r"$0$", r"$\frac{2}{3}$", r"$2$"], fontsize=8)

    axSPhP.set_xlabel(r"$\omega$", labelpad = 1)
    axSPhP.set_ylabel(r"$\rho_{||} / \rho_0$")

    axSPhP.text(0.05, 0.8, r"$\mathrm{Dielectric}$", fontsize=8, transform=axSPhP.transAxes)
    axSPhP.text(0.05, 0.68, r"$\mathrm{Behavior}$", fontsize=8, transform=axSPhP.transAxes)
    axSPhP.text(0.65, 0.4, r"$\mathrm{Surface \, Mode}$", fontsize=8, transform=axSPhP.transAxes)

    arrowSPhP = mpatches.FancyArrowPatch((0.75, 0.525), (0.55, 0.7),
                                     mutation_scale=0.1,
                                     facecolor = 'black',
                                     lw = .5,
                                     edgecolor="black",
                                     transform=axSPhP.transAxes)

    axSPhP.add_patch(arrowSPhP)


    axSPhP.text(-0.1, 1.05, r"$(\mathrm{a})$", fontsize=8, transform=axSPhP.transAxes)

    legend = axSPhP.legend(fontsize=8, loc='lower left', bbox_to_anchor=(-0.02, 0.92), edgecolor='black', ncol=4)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    wInf = np.sqrt(wLO**2) / np.sqrt(2)
    lambda0 = 2. * np.pi * consts.c / wInf
    unitFac = consts.m_e / (consts.hbar * wLO / consts.c**2)

    axM.plot(zArrM / lambda0, (mArr) * unitFac, color = 'red', lw = .8, linestyle = "--", label = r"$\mathrm{Total}$", zorder = 666)
    #axM.plot(zArrM / lambda0, mArr[-1] * zArrM[-1] ** 3 / zArrM ** 3 * unitFac, color="red",
    #            lw=.4, linestyle = "-", label = r"$\sim z^{-3}$")
    axM.plot(zArrM / lambda0, mBulkArr * unitFac, color = cmapPink(0.2), lw = 1.5, label = r"$\mathrm{Only \, Bulk \, Modes}$")
    axM.plot(zArrM / lambda0, (mArr - mBulkArr) * unitFac, color = cmapPink(0.525), lw = 1.5, label = r"$\mathrm{Only \, Surface \, Modes}$")

    axM.set_xscale("log")
    axM.set_yscale("log")
    axM.set_xlim(1e-4, 5. * 1e0)
    axM.set_ylim(1. * 1e-5, 7 * 1e1)

    axM.set_xlabel(r"$z [\lambda_0]$", fontsize = 8)
    axM.set_ylabel(r"$\Delta m \left[\frac{\hbar \omega_{\infty}}{c^2} \right]$", fontsize = 8, labelpad = 2)

    axM.minorticks_off()
    axM.set_xticks([1e-4, 1e-2, 1e-0])
    axM.set_yticks([1e-4, 1e-2, 1e0, 1e2])

    axM.text(-0.2, 1.05, r"$(\mathrm{b})$", fontsize=8, transform=axM.transAxes)

    legend = axM.legend(fontsize=7, loc='upper left', bbox_to_anchor=(0., .9), edgecolor='black', ncol=1)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        axSPhP.spines[axis].set_linewidth(.5)
        axM.spines[axis].set_linewidth(.5)

    plt.savefig("./PaperPlots/fig2Broad.png")

def createFig3Plot():

    wArrSubdivisions = 200

    epsInf = 1.
    #numbers for sto
    wLO = 32.04 * 1e12
    wTO = 1e6
    L = 1.
    wInf = np.sqrt(epsInf * wLO ** 2 + wTO ** 2) / np.sqrt(epsInf + 1)
    lambda0 = 2. * np.pi * consts.c / wInf

    zArr, massArr = getMassDataSPP(wLO, wTO)

    wArr, dosTot = getDosDataSPP(wArrSubdivisions, zArr, wLO, wTO, epsInf, L)

    filenameFP = "delMassFP50GHzCutoff"
    cutoff, dArr, FPMassArr = retrieveMassData(filenameFP)

    NewFig3(zArr, wArr, dosTot, massArr, dArr, FPMassArr, wLO, wTO)

def getMassDataSPP(wLO, wTO):

    wLOStr = "wLO" + str(int(wLO * 1e-12))
    wTOStr = "wTO" + str(int(wTO * 1e-12))
    filenameSPP = "./savedData/PaperData/massForPaperLowerCutoff10GHz" + wLOStr + wTOStr + ".hdf5"
    cutoff, zArrSPP, massSPPArr = prodV2.readMasses(filenameSPP)

    return (zArrSPP, massSPPArr)

def getDosDataSPP(wArrSubdivisions, zArr, wLO, wTO, epsInf, L):

    dosTETotal, dosTMPara = prodV2.retrieveDosPara(
        wArrSubdivisions,
        zArr,
        wLO,
        wTO,
        epsInf,
        L)

    wArrOne = prodV2.defineFreqArrayOne(wArrSubdivisions)
    dosSurf = np.zeros((len(wArrOne), len(zArr)))
    for wInd, wVal in enumerate(wArrOne):
        epsilon = epsFunc.epsilon(wVal, wLO, wTO, epsInf)
        dosSurf[wInd, :] = 1. / (1. + np.abs(epsilon)) * dosAsOfFreq.getDosTMSurf(wVal, zArr, L, wLO, wTO, epsInf)

    return (wArrOne, dosTETotal + dosTMPara + dosSurf)

def NewFig3(zArrSPP, wArr, dosTot, massSPPArr, dArr, FPMassArr, wLO, wTO):

    wInf = np.sqrt(wLO**2) / np.sqrt(2)
    lambda0 = 2. * np.pi * consts.c / wInf
    unitFac = consts.m_e / (consts.hbar * wLO / consts.c**2)

    fig = plt.figure(figsize=(7.05, 1.5), dpi=400)
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 1],
                           wspace=0.4, hspace=0, top=0.89, bottom=0.2, left=0.05, right=0.96)

    axDos = plt.subplot(gs[0, 0])
    axMass = plt.subplot(gs[0, 1])
    axFPMass = plt.subplot(gs[0, 2])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')


    indArr = np.array([1, 1, 167, 333], dtype=int)
    #indArr = np.array([1, 30, 60, 80, 180], dtype=int)

    print(np.where(dosTot[0:, indArr[3]] < 1e-10))
    #dosTot[6407, indArr[3]] = dosTot[6406, indArr[3]]
    zeroInds = [0, 6407]


    wArrPlot = np.delete(wArr, zeroInds)
    dosPlotArr1 = np.delete(dosTot[:, indArr[1]], zeroInds)
    dosPlotArr2 = np.delete(dosTot[:, indArr[2]], zeroInds)
    dosPlotArr3 = np.delete(dosTot[:, indArr[3]], zeroInds)


    # axSPhP.plot(wArr, dos[:, indArr[0]], color=cmapBone(0.1), lw=.7, label="$z = $" + "{:1.0f}".format(zArrSPP[indArr[0]] / lambda0) + r"$\lambda_0$")
    axDos.plot(wArrPlot, dosPlotArr1, color=cmapPink(0.1), lw=1.2,
            label="$z = $" + "{:1.0f}".format(zArrSPP[indArr[1]] / lambda0) + r"$\lambda_0$")
    axDos.plot(wArrPlot, dosPlotArr2, color=cmapPink(0.4), lw=1.2,
            label="$z = $" + "{:1.0f}".format(zArrSPP[indArr[2]] / lambda0) + r"$\lambda_0$")
    axDos.plot(wArrPlot, dosPlotArr3, color=cmapPink(0.6), lw=1.2,
            label="$z = $" + "{:1.1f}".format(zArrSPP[indArr[3]] / lambda0) + r"$\lambda_0$")
    #
    axDos.axhline(2. / 3., lw = 0.5, color = 'gray', zorder = -666)

    axDos.set_xlim(0, 40 * 1e12)
    axDos.set_ylim(0., 5.)
    
    axDos.set_xticks([0., wInf, wLO])
    axDos.set_xticklabels([r"$0$", r"$\omega_{\infty}$", r"$\omega_{\rm P}$"], fontsize=8)
    axDos.set_yticks([0., 2. / 3., 2])
    axDos.set_yticklabels([r"$0$", r"$\frac{2}{3}$", r"$2$"], fontsize=8)

    axDos.set_xlabel(r"$\omega$", labelpad = 1)
    axDos.set_ylabel(r"$\rho_{||} / \rho_0$")

    axDos.text(0.05, 0.5, r"$\mathrm{Metallic}$", fontsize=8, transform=axDos.transAxes)
    axDos.text(0.05, 0.38, r"$\mathrm{Behavior}$", fontsize=8, transform=axDos.transAxes)
    axDos.text(0.62, 0.4, r"$\mathrm{Surface \, Mode}$", fontsize=8, transform=axDos.transAxes)

    arrowSPhP = mpatches.FancyArrowPatch((0.72, 0.525), (0.52, 0.7),
                                     mutation_scale=0.1,
                                     facecolor = 'black',
                                     lw = .5,
                                     edgecolor="black",
                                     transform=axDos.transAxes)

    axDos.add_patch(arrowSPhP)


    legend = axDos.legend(fontsize=8, loc='lower left', bbox_to_anchor=(-0.1, 0.92), edgecolor='black', ncol=3, columnspacing=0.8)
    legend.get_frame().set_alpha(0.)
    legend.get_frame().set_boxstyle('Square', pad=0.1)
    legend.get_frame().set_linewidth(0.0)
    ####################################
    #plot mass
    ##################################

    axIn = axMass.inset_axes([0.45, 0.48, 0.5, 0.5])

    axIn.plot(zArrSPP / lambda0, np.abs(massSPPArr) * unitFac, color = cmapBone(.25), lw = 1.2, label = r"$\omega_{\rm p} = 32 \mathrm{THz}$")
    axIn.plot(zArrSPP / lambda0, np.abs(massSPPArr[-1]) * zArrSPP[-1]**3 / zArrSPP**3 * unitFac, color = "red", lw = .4)
    fitInd = 150
    axIn.plot(zArrSPP / lambda0, np.abs(massSPPArr[fitInd]) * zArrSPP[fitInd]**1 / zArrSPP**1 * unitFac, color = "red", lw = .4)
    axIn.set_xscale("log")
    axIn.set_yscale("log")
    axIn.set_xlim(1e-2, 1e1)
    axIn.set_ylim(1e-5, 2. * 1e1)
    axIn.set_xticks([1e-2, 1e1])
    axIn.set_xticklabels([r"$10^{-2}$", r"$10^{1}$"], fontsize = 6)
    axIn.set_yticks([1e-4, 1e0])
    axIn.set_yticklabels([r"$10^{-4}$", r"$10^{0}$"], fontsize = 6)
    axIn.set_xlabel("$z [\lambda_0]$", fontsize = 6, labelpad = -6)
    axIn.set_ylabel(r"$|\Delta m| \, \left[\frac{h \hspace{-1mm} \rule[1.05ex]{0.25em}{0.3pt} \omega_{\rm p}}{c^2} \right]$", labelpad = 4., fontsize = 6)
    axIn.text(0.1, 0.8, r"$\sim z^{-3}$", color = 'red', transform = axIn.transAxes, fontsize = 6)
    axIn.text(0.7, 0.3, r"$\sim z^{-1}$", color = 'red', transform = axIn.transAxes, fontsize = 6)

    #axMass.text(0.1, 1.03, r"$\mathrm{Near{-}Field}$", transform = axMass.transAxes, fontsize = 8)
    #axMass.text(0.65, 1.03, r"$\mathrm{Far{-}Field}$", transform = axMass.transAxes, fontsize = 8)

##################################
    # Inset
##################################

    axMass.plot(zArrSPP / lambda0, massSPPArr * unitFac * 1e4, color = cmapBone(.25), lw = 1.2, label = r"$\omega_{\rm p} = 32 \mathrm{THz}$")
    axMass.axhline(0., lw = 0.5, color = 'black')
    axMass.set_xlim(1e-2, 6)
    axMass.set_ylim(- 2.8, 6.5)
    axMass.set_xlabel("$z [\lambda_0]$", fontsize = 8, labelpad = 0)
    axMass.set_ylabel(r"$\Delta m \, \left[\frac{h \hspace{-1mm} \rule[1.05ex]{0.25em}{0.3pt} \omega_{\rm p}}{c^2} \right]$", labelpad = 0., fontsize = 8)
    axMass.set_yticks([-2., 0, 2., 4.])
    axMass.set_yticklabels([r"$-2$", r"$0$", r"$2$", r"$4$"], fontsize = 8)
    axMass.set_xticks([0, 1, 5])
    axMass.set_xticklabels([r"$0$", r"$1$", r"$5$"], fontsize = 8)
    axMass.text(-0.18, 0.9, r"$\times 10^{-4}$", transform = axMass.transAxes, fontsize = 8)
#########################################################
    #Plot FabryPerotMass
#########################################################

    axFPMass.plot(dArr, FPMassArr * 1e9, color=cmapBone(.45), lw=1.2)

    axFPMass.set_xscale("log")
    axFPMass.set_xlim(np.amin(dArr), np.amax(dArr))

    axFPMass.set_ylim(-2., 0.)
    axFPMass.set_xlabel("$d [\mathrm{m}]$", fontsize=8, labelpad = 0)
    axFPMass.set_ylabel(r"$\Delta m \, [m_{\rm e}]$", labelpad=0., fontsize=8)

    axFPMass.set_xticks([1e-6, 1e-3])
    axFPMass.set_xticklabels(["$10^{-6}$", "$10^{-3}$"], fontsize=8)

    axFPMass.set_yticks([0., -1.])
    axFPMass.set_yticklabels(["$0$", "$-1$"], fontsize=8)

    axFPMass.text(0., 1.05, r"$\times 10^{-9}$", fontsize=7, transform = axFPMass.transAxes)
    axFPMass.text(0.3, 0.6, r"$\mathrm{Fabry{-}Perot}$", fontsize=7, transform = axFPMass.transAxes)
    axFPMass.text(0.3, 0.45, r"$\mathrm{only \; far{-}field}$", fontsize=7, transform = axFPMass.transAxes)




    axDos.text(-0.15, 1.04, r"$\mathrm{(a)}$", fontsize = 8, transform = axDos.transAxes)
    axMass.text(-0.22, 1.04, r"$\mathrm{(b)}$", fontsize = 8, transform = axMass.transAxes)
    axFPMass.text(-0.4, 1.04, r"$\mathrm{(c)}$", fontsize = 8, transform = axFPMass.transAxes)

    for axis in ['top', 'bottom', 'left', 'right']:
        axDos.spines[axis].set_linewidth(.5)
        axMass.spines[axis].set_linewidth(.5)
        axIn.spines[axis].set_linewidth(.5)
        axFPMass.spines[axis].set_linewidth(0.5)


    plt.savefig("./PaperPlots/Fig3Broad.png")

def getDosFP():

    dir = "./savedData/PaperData/"

    fileName = dir + "DosFP.h5"
    h5f = h5py.File(fileName, 'r')
    d = h5f['d'][:]
    wArr = h5f['wArr'][:]
    dos = h5f['dos'][:]
    h5f.close()

    return (d, wArr, dos)

def FabryPerotPlot():
    #filenameFP = "delMassFPPaperNew"
    filenameFP = "delMassFP50GHzCutoff"
    cutoff, dArr, massFPArr = retrieveMassData(filenameFP)

    dFP, wArrFP, dosFP = getDosFP()

    fig = plt.figure(figsize=(3.3, 1.8), dpi=400)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.0, hspace=0.5, top=0.95, bottom=0.22, left=0.14, right=0.95)
    axDos = plt.subplot(gs[0, 0])
    #axMass = plt.subplot(gs[1, 0])

    cmapPink = cm.get_cmap('pink')
    cmapBone = cm.get_cmap('bone')

    axDos.plot(wArrFP, dosFP, color=cmapPink(0.45), lw=1.2)
    axDos.axhline(2. / 3., lw=0.5, color='gray', zorder=-666)

    axDos.set_xlim(0., 1e14)
    axDos.set_ylim(0., 2.1)
    w0 = np.pi * consts.c / dFP[0]
    axDos.set_xticks([0., w0, 3. * w0, 5. * w0, 7. * w0, 9. * w0])
    axDos.set_xticklabels([r"$0$", r"$\omega_0$", r"$3\omega_0$", r"$5\omega_0$", r"$7\omega_0$", r"$9\omega_0$"])
    axDos.set_yticks([0., 2. / 3., 2])
    axDos.set_yticklabels([r"$0$", r"$\frac{2}{3}$", r"$2$"], fontsize=8)

    axDos.set_xlabel(r"$\omega$")
    axDos.set_ylabel(r"$\rho_{||} / \rho_0$")

    #axDos.text(-0.15, 1.05, r"$(\mathrm{a})$", fontsize=8, transform=axDos.transAxes)

    #axMass.plot(dArr, massFPArr * 1e9, color=cmapBone(.45), lw=1.2)

    #axMass.set_xscale("log")
    #axMass.set_xlim(np.amin(dArr), np.amax(dArr))

    #axMass.set_ylim(-2., 0.)
    #axMass.set_xlabel("$d [\mathrm{m}]$", fontsize=8)
    #axMass.set_ylabel(r"$\Delta m \, [m_{\rm e}]$", labelpad=0., fontsize=8)

    #axMass.set_xticks([1e-6, 1e-5, 1e-4, 1e-3])
    #axMass.set_xticklabels(["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$"], fontsize=8)

    #axMass.set_yticks([0., -1.])
    #axMass.set_yticklabels(["$0$", "$-1$"], fontsize=8)

    #axMass.text(1.1 * 1e-6, -.3, r"$\times 10^{-9}$", fontsize=8)

    ####### Insets
    #axinFP = axMass.inset_axes([0.4, 0.25, 0.5, 0.55])
    #axinFP.plot(dArr, np.abs(massFPArr), color=cmapBone(.45), lw=1.6)
    #axinFP.plot(dArr, dArr[-1] / dArr * np.abs(massFPArr[-1]), color=cmapPink(.55), linestyle='--', dashes=(4, 4),
    #            lw=1.2, label=r"$\sim \hspace{-0.5mm} d^{-1}$")

    #axinFP.set_xscale("log")
    #axinFP.set_yscale("log")
    #axinFP.set_xlim(1e-6, 1e-3)
    #axinFP.set_ylim(1e-12, 1e-8)
    #axinFP.set_ylabel(r"$ \log \left(|\Delta m|\right)$", labelpad=2, fontsize=7)
    #axinFP.set_xlabel("$ \log(d)$", fontsize=7, labelpad=2)

    #axinFP.set_xticks([])
    #axinFP.set_yticks([])
    #axinFP.tick_params(axis='y', which='major', pad=0)
    #axinFP.tick_params(axis='x', which='major', pad=1)

    #axMass.text(-.15, 1.05, r"$\mathrm{(b)}$", transform=axMass.transAxes, fontsize=8)

    #legend = axinFP.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1., 1.1), edgecolor='black', ncol=1,
    #                       handlelength=2, handletextpad=0.5)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    for axis in ['top', 'bottom', 'left', 'right']:
        axDos.spines[axis].set_linewidth(.5)
        #axMass.spines[axis].set_linewidth(.5)
        #axinFP.spines[axis].set_linewidth(.5)

    plt.savefig("../SphPStuff/PaperPlots/SuppFPPlot.png")


def main():

    Fig1()
    #createFig2Plot()
    #createFig3Plot()
    #FabryPerotPlot()


main()