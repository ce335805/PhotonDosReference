import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors
from matplotlib import gridspec

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

def normFac(omega, wLO, wTO, epsInf):
    #wTO = 0.
    #tau = 1e8
    #frac = (wLO**2 - wTO**2) / (wTO**2 - omega**2 - tau**2)**2
    frac = (wLO**2 - wTO**2) / (wTO**2 - omega**2)**2
    return epsInf * (1 + wTO**2 * frac)
    #return epsInf

def epsilon(omega, wLO, wTO, epsInf):
    #wTO = 0
    #tau = 1e8
    #return epsInf * (wLO ** 2 - omega ** 2) / (wTO ** 2 - omega ** 2 - tau**2)
    return epsInf * (wLO ** 2 - omega ** 2) / (wTO ** 2 - omega ** 2)
    #return epsInf

def kDFromK(kVal, omega, wLO, wTO, epsInf):
    return np.sqrt((epsilon(omega, wLO, wTO, epsInf) - 1) * omega ** 2 / consts.c ** 2 + kVal ** 2)

def kDFromKEva(kVal, omega, wLO, wTO, epsInf):
    return np.sqrt((epsilon(omega, wLO, wTO, epsInf) - 1) * omega ** 2 / consts.c ** 2 - kVal ** 2)

def kDFromKRes(kVal, omega, wLO, wTO, epsInf):
    return np.sqrt((1 - epsilon(omega, wLO, wTO, epsInf)) * omega ** 2 / consts.c ** 2 - kVal ** 2)

def kDFromKSurf(kVal, omega, wLO, wTO, epsInf):
    return np.sqrt((1 - epsilon(omega, wLO, wTO, epsInf)) * omega ** 2 / consts.c ** 2 + kVal ** 2)


def plotEpsilon(omegaArr, epsilonArr, epsInf, wLO, wTO):

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    ax.plot(omegaArr * 1e-12, epsilonArr, color='peru', lw=1.)

    ax.axhline(epsInf, lw = 0.5, color = 'gray')
    ax.axhline(0, lw = 0.5, color = 'gray')
    ax.axvline(wTO * 1e-12, lw = 0.5, color = 'gray')
    ax.axvline(wLO * 1e-12, lw = 0.5, color = 'gray')

    ax.set_xlim(np.amin(omegaArr) * 1e-12, np.amax(omegaArr) * 1e-12)
    ax.set_ylim(- 10., 40.)

    ax.set_xlabel(r"$\omega \, [\mathrm{THz}]$")
    ax.set_ylabel(r"$\varepsilon(\omega)$")

    plt.savefig("./SPhPPlotsSaved/epsilon.png")