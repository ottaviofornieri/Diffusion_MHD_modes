from __future__ import division

import os
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import rc, rcParams
import matplotlib.ticker as mticker

import scipy.integrate as integrate
from scipy import interpolate
from scipy.interpolate import griddata
import scipy.special as bessel
import scipy.optimize as opt
from scipy.optimize import curve_fit

import sys
import pylab
import time
from tqdm import tqdm



###############
# LaTeX block #
###############
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Palatino']})
rc('xtick', labelsize=18)
rc('ytick', labelsize=18)
rcParams['legend.numpoints'] = 1



def plot_cosmetics_single():
    ax = plt.gca()
    ax.tick_params(direction='in', axis='both', which='major', length=6.5, width=1.2, labelsize=18)
    ax.tick_params(direction='in', axis='both', which='minor', length=3., width=1.2, labelsize=18)
    ax.xaxis.set_tick_params(pad=7)
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 15
    
    
def plot_cosmetics_multi():
    ax = plt.gca()
    ax.tick_params(direction='in', axis='both', which='major', length=6.5, width=1.2, labelsize=20)
    ax.tick_params(direction='in', axis='both', which='minor', length=3., width=1.2, labelsize=20)
    ax.xaxis.set_tick_params(pad=10)
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 10



########################################################
# CONSTANTS: physical constants and conversion factors #
########################################################
pc_mt = 3.0857e+16              # parsec in meters
pc_cm = 3.0857e+18              # parsec in centimeters
m_to_pc = pc_mt**(-1)           # to pass from meters to parsec
cm_to_pc = 3.24e-19             # to pass from centimeters to parsec
conv_factor_GeV_to_erg = 0.00160218   # the value in [GeV] has to be multipled by this factor
conv_factor_erg_to_GeV = 624.151
conv_factor_yr_to_sec = 31.56e+6
conv_GeV_g = 1.78e-24

m_p = 0.939                     # proton mass in GeV/c^2
m_e = 0.000510998918            # electron mass in GeV/c^2
c = 2.99e+10                    # speed of light in [cm sec^(-1)]
c_pcSec = c * cm_to_pc          # same, in [pc sec^{-1}]



################################
### Environmental parameters ###
################################
gamma_ratio = m_p/m_e
WIM_size = 900.               # in [pc]
Halo_size_minus_WIM = 5100.   # in [pc]


B_field = 10.         # magnetic field, in [muGauss]
n_ISM = 0.1           # density of the environment, in [cm^{-3}]
T_ISM = 5.e3          # in [K]

L_inj = 10.           # in [pc]
M_A = 1.              # Alfvenic Mach Number


####
v_A = 6.27e5 * (B_field / 3.) * (1. / n_ISM)**(1/2)
v_A_pcSec = v_A * cm_to_pc                                                # in [pc sec^{-1}]
rho = n_ISM * (m_p * conv_GeV_g)
Debye_length = 0.95e5 * (T_ISM / 1.e6)**(1/2) * (1.e-3 / n_ISM)**(1/2)    # in [cm]

b_min = 8.63e-8 * (1.e4 / T_ISM)                                          # in [cm]
ln_Lambda = np.log(Debye_length / b_min)
eta_0 = 6.e3 * (37 / ln_Lambda) * (T_ISM / 1.e8)**(5/2)  # in [g cm^{-1} s^{-1}]


x_c = ( (6 * rho * M_A**2 * (L_inj*pc_cm) * v_A) / (eta_0) )**(2/3)
beta_plasma = 3.3 * (3. / B_field)**2 * (n_ISM / 1.) * (T_ISM / 1.e4)
####


print('rho =', rho, '[g]')
print('v_A =', v_A / 1.e5, '[km s^{-1}]')
print('ln_Lambda =', ln_Lambda)
print('x_c =', x_c)
print('plasma beta =', beta_plasma)
print('')



path_plots = '/Users/ottaviofornieri/PHYSICS_projects/Diffusion_Fast_Modes/Plots_GSSI/'
p_CR = np.logspace(-1, 5, num=60)

p_CR_chosen = [1.e2, 1.e4]  # in [GeV]
indx_p_CR = [np.argmin( abs(p_CR - p_CR_chosen[i]) ) for i in range(len(p_CR_chosen))]


def Larmor_radius(p_, B_ISM_):
    # result in [cm] and [pc]
    return 3.31e+12 * (p_ / 1.) * (1. / B_ISM_), 3.31e+12 * (p_ / 1.) * (1. / B_ISM_) * cm_to_pc


def LarmorToMomentum(r_L_, B_):
    # with r_L in [cm] and B in [muG], the result is in [GeV]
    return 3.021e-13 * (r_L_ / 1.) * (B_ / 1.)


ell_inverse = np.logspace(start=1., stop=10., num=700)
kL_list = [L_inj / ell_inverse[ik] for ik in range(len(ell_inverse))]
indx_result_100GeV = np.argmin( abs(Larmor_radius(p_CR[indx_p_CR[0]], B_field)[1] - kL_list) )
indx_result_10TeV = np.argmin( abs(Larmor_radius(p_CR[indx_p_CR[1]], B_field)[1] - kL_list) )


print(f'indx 100GeV: {indx_result_100GeV}, indx 10TeV: {indx_result_10TeV}')
print(f'kL (100GeV) = {ell_inverse[indx_result_100GeV]}')
print('')



##################################################
### Calculate the truncation scale for damping ###
##################################################
import warnings
warnings.filterwarnings('ignore')

max_kL = L_inj / Larmor_radius(1.e-2, B_field)[1]    # maximum kL, resonating with 10^{-2}GeV CRs


# collisionless
def kmaxL_coll( alpha_ ):
    numerator = 4. * M_A**4 * gamma_ratio * ( np.cos( np.radians(alpha_) ) )**2
    denominator = np.pi * beta_plasma * ( np.sin( np.radians(alpha_) ) )**4
    exp = np.exp( 2. / ( beta_plasma * gamma_ratio * ( np.cos( np.radians(alpha_) ) )**2 ) )
    return numerator / denominator * exp

def kmaxL_coll_xi( cosAlpha_ ):
    numerator = 4. * M_A**4 * gamma_ratio * ( cosAlpha_ )**2
    denominator = np.pi * beta_plasma * ( 1 - cosAlpha_**2 )**2
    exp = np.exp( 2. / ( beta_plasma * gamma_ratio * ( cosAlpha_ )**2 ) )
    return min(max_kL, numerator / denominator * exp)


# collisional (viscous)
def kmaxL_visc( alpha_ ):
    return x_c * ( np.sin( np.radians(alpha_) ) )**(-4/3)

def kmaxL_visc_xi( cosAlpha_ ):
    return min(max_kL, x_c * ( 1 - cosAlpha_**2 )**(-2/3))



num_alpha = 700
alpha = np.linspace(start = 0., stop = 90., num = num_alpha)    # wave pitch-angle in degrees
cosAlpha = np.linspace(start = 0.1, stop = 1., num = num_alpha)


'''plt.figure(figsize=(10.5, 4.5))
plt.subplot(1, 2, 1)
plot_cosmetics_multi()

plt.plot(alpha, kmaxL_coll(alpha), lw=2.5, color='blue', label='collisionless')
plt.plot(alpha, kmaxL_visc(alpha), lw=2.5, color='orange', label='viscous')
plt.axis([0.,90., 1.e2,1.e8])
plt.axhline(y=ell_inverse[indx_result_100GeV], ls='--', lw=1.5, color='red')
plt.axhline(y=ell_inverse[indx_result_10TeV], ls='--', lw=1.5, color='green')
plt.text(0.4, 0.703, '$k_{\mathrm{max}}L = r_L \\big|_{100 \, \mathrm{GeV}}$', fontsize=15, transform = plt.gca().transAxes)
plt.text(0.02, 0.366, '$k_{\mathrm{max}}L = r_L \\big|_{10 \, \mathrm{TeV}}$', fontsize=15, transform = plt.gca().transAxes)
plt.title('Truncation scale, $M_{\mathrm{A}} = \,$' + str(M_A), fontsize=16, loc='center', pad=None)
plt.xlabel('$\\alpha_w$',fontsize=20)
plt.ylabel('$k_\mathrm{max}L$',fontsize=20)
plt.yscale('log')
plt.legend(fontsize=16, frameon=False, loc='best')


plt.subplot(1, 2, 2)
plot_cosmetics_multi()

plt.plot(cosAlpha, [kmaxL_coll_xi(i) for i in cosAlpha], lw=2.5, color='blue', label='collisionless')
plt.plot(cosAlpha, [kmaxL_visc_xi(i) for i in cosAlpha], lw=2.5, color='orange', label='viscous')
plt.title('Truncation scale, $M_{\mathrm{A}} = \,$' + str(M_A), fontsize=16, loc='center', pad=None)
plt.xlabel('$\\xi = \mathrm{cos} \; \\alpha_w$',fontsize=20)
plt.yscale('log')
plt.legend(fontsize=16, frameon=False, loc='best')
plt.tight_layout()'''


###############################
### Define useful functions ###
###############################
# Resonance functions

def resonance_fast( x_, xi_, mu_, R_, n_ ):
    factor = L_inj * np.sqrt(np.pi) / ( np.sqrt(2) * x_ * np.abs(xi_) * c_pcSec * np.sqrt( (1 - mu_**2) * M_A ) )
    exp_n0 = np.exp( - ( mu_ - v_A_pcSec / (xi_ * c_pcSec) )**2 / ( 2 * (1 - mu_**2) * M_A ) )
    exp_nPlus1 = np.exp( - ( mu_ + 1. / (x_ * xi_ * R_) )**2 / ( 2 * (1 - mu_**2) * M_A ) )
    exp_nMinus1 = np.exp( - ( mu_ - 1. / (x_ * xi_ * R_) )**2 / ( 2 * (1 - mu_**2) * M_A ) )
    if n_ == 0:
        return factor * exp_n0
    elif n_ == 1:
        return factor * exp_nPlus1
    elif n_ == -1:
        return factor * exp_nMinus1


def resonance_alf( x_para_, mu_, R_, n_ ):
    numerator = L_inj * np.sqrt(np.pi)
    denominator = np.sqrt(2) * x_para_ * c_pcSec * np.sqrt( (1 - mu_**2) * M_A )
    exp_nPlus1 = np.exp( - ( mu_ + 1. / (x_para_ * R_) )**2 / ( 2 * (1 - mu_**2) * M_A ) )
    exp_nMinus1 = np.exp( - ( mu_ - 1. / (x_para_ * R_) )**2 / ( 2 * (1 - mu_**2) * M_A ) )
    if n_ == 1:
        return numerator / denominator * exp_nPlus1
    elif n_ == -1:
        return numerator / denominator * exp_nMinus1
    
    
    
# Calculation of the D_mumu
    
def fast_spectrum_norm( x_ ):
    return M_A**2 * L_inj**3 / ( 8 * np.pi ) * x_**(-7/2)

def factor_FastModes( mu_, R_ ):
    return ( 4 * np.pi * c_pcSec * (1 - mu_**2) ) / ( L_inj**4 * R_**2 )

def Bessel_arg( R_, x_, xi_, mu_ ):
    return R_ * x_ * np.sqrt(1 - xi_**2) * np.sqrt(1 - mu_**2)



'''print( f'nan? {resonance_fast(1.e8, 0.000001, 0.5, 2.14488e-8, 0)}' )
cosAlpha_symm = np.linspace(start = -1., stop = 1., num = num_alpha)
plt.plot(cosAlpha_symm, resonance_fast(1.e8, cosAlpha_symm, 0.9, 2.14488e-8, 0), lw=2., color='blue', label='current')
plt.show()'''


#############################
### Calculate the D(R)  #####
#############################

case_region = 'Halo'
#case_region = 'Disk'

    
# Define the integration variables #
length_energy_array = 60
length_mu_array = 50    

p_CR = np.logspace(start = -1., stop = 5., num = length_energy_array)     # CR momentum, in [GeV/c]
R = Larmor_radius(p_CR, B_field)[1] / L_inj
mu_array = np.linspace(start = 0., stop = 0.95, num = length_mu_array)    # problems with 0° pitch-angle scattering
pointsPerDecade = 10                                                      # for the integral over x


integralOverX_TTD = np.zeros( len(cosAlpha) )
integralOverX_Gyro = np.zeros( len(cosAlpha) )
integralOverCsi = np.zeros( (len(R), len(mu_array)) )
integralOverMu = np.zeros( len(R) )


print(f'The D(E) is computed in the {case_region}.')
print('')
print(f'energy points: {len(R)}')
print(f'number of points/decade in x: {pointsPerDecade}')
print(f'N_xi = {len(cosAlpha)}, xi array: [{cosAlpha[0]} -> {cosAlpha[-1]}]')
print(f'N_mu = {len(mu_array)}, mu array: [{mu_array[0]} -> {mu_array[-1]}]')
print('')


for indx_R, r in enumerate(tqdm(R)):
    
    for indx_mu, mu in enumerate(mu_array):
        
        for indx_csi, csi in enumerate(cosAlpha):
            

            if case_region == 'Halo':
                n_decades = round( np.log10( kmaxL_coll_xi(csi) ) )
                x_grid = np.logspace(start=0., stop=np.log10( kmaxL_coll_xi(csi) ), num=n_decades*pointsPerDecade)
            
            elif case_region == 'Disk':
                n_decades = round( np.log10( min( kmaxL_visc_xi(csi), kmaxL_coll_xi(csi) ) ) )
                x_grid = np.logspace(start=0., stop=np.log10( min( kmaxL_visc_xi(csi), kmaxL_coll_xi(csi) ) ), num=n_decades*pointsPerDecade)
            

            ## TTD (n=0)
            n_TTD = 0
            besselFunc_squared = bessel.jv( 1, Bessel_arg(r, x_grid, csi, mu) )**2
            resonanceFunc_TTD = resonance_fast( x_grid, csi, mu, r, n_TTD )
            turbSpectrum = fast_spectrum_norm( x_grid )
            
            integralOverX_TTD[indx_csi] = np.trapz( y = csi**2 * x_grid**2 * besselFunc_squared * resonanceFunc_TTD * turbSpectrum, x = x_grid, axis=-1 )
            
            
            ## Gyro (n≠0)
            n_Gyro_plus1 = 1
            n_Gyro_minus1 = -1
            resonanceFunc_Gyro = resonance_fast( x_grid, csi, mu, r, n_Gyro_plus1 ) + resonance_fast( x_grid, csi, mu, r, n_Gyro_minus1 )
            
            integralOverX_Gyro[indx_csi] = np.trapz( y = csi**2 * x_grid**2 * besselFunc_squared * resonanceFunc_Gyro * turbSpectrum, x = x_grid, axis=-1 )
            
            
        integralOverCsi[indx_R, indx_mu] = c_pcSec / L_inj * factor_FastModes(mu, r) * np.trapz( integralOverX_TTD + integralOverX_Gyro, x = cosAlpha, axis=-1 )
        #integralOverCsi[indx_R, indx_mu] = c_pcSec / L_inj * factor_FastModes(mu, r) * np.trapz( y = integralOverX_TTD, x = cosAlpha, axis=-1 )


for ir in range( len(R) ):
    integralOverMu[ir] = np.trapz( y = c**2 * (1 - mu_array**2)**2 / integralOverCsi[ir,:], x = mu_array, axis=-1 )


## Plot the resulting D(E)
plt.figure(figsize=(13, 5.))

plt.subplot(1, 2, 1)
plot_cosmetics_multi()


plt.loglog(R, integralOverMu, lw=2.5, color='blue')
plt.xlabel('$ R \equiv r_L \\big/L$',fontsize=20)
plt.ylabel('$D(E) \, [\mathrm{cm^2 \cdot s^{-1}}]$',fontsize=20)
plt.text(0.85, 0.05, str(case_region), fontsize=17, transform = plt.gca().transAxes)


plt.subplot(1, 2, 2)
plot_cosmetics_multi()

plt.loglog( LarmorToMomentum(R * (L_inj*pc_cm), B_field), integralOverMu, lw=2.5, color='blue')
plt.xlabel('$ E \, [\mathrm{GeV}]$',fontsize=20)

plt.text(0.05, 0.9, '$B = \,$' + str("{:.1f}".format(B_field)) + '$\, \mu \mathrm{G}$', fontsize=17, transform = plt.gca().transAxes)
plt.text(0.05, 0.8, '$M_\mathrm{A} = \,$' + str("{:.0f}".format(M_A)), fontsize=17, transform = plt.gca().transAxes)
plt.text(0.05, 0.7, '$L_\mathrm{inj} = \,$' + str("{:.0f}".format(L_inj)) + '$\, \mathrm{pc}$', fontsize=17, transform = plt.gca().transAxes)
plt.text(0.05, 0.6, '$n_\mathrm{ISM} = \,$' + str("{:.1f}".format(n_ISM)) + '$\, \mathrm{cm^{-3}}$', fontsize=17, transform = plt.gca().transAxes)
plt.text(0.05, 0.5, '$T_\mathrm{ISM} = \,$' + str("{:.0f}".format(T_ISM)) + '$\, \mathrm{K}$', fontsize=17, transform = plt.gca().transAxes)
plt.text(0.05, 0.4, '$\\beta_{\mathrm{plasma}} = \,$' + str("{:.3f}".format(beta_plasma)), fontsize=17, transform = plt.gca().transAxes)
plt.text(0.85, 0.05, str(case_region), fontsize=17, transform = plt.gca().transAxes)
plt.tight_layout()
plt.show()