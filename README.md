# Diffusion_MHD_modes
We compute the diffusion coefficient D(E) generated by the scattering of CR particles off the Magnetohydrodynamic (MHD) modes, within parametrized turbulent spectra.
The spectra of the turbulent fluctuations are modelled as in Yan&Lazarian (2002) - Phys. Rev. Lett. 89, 281102. 

- The fast magneto-sonic modes cascade with an isotropic spectrum W(k)~k^{5/3}. They suffer collisionless and collisional damping.
- The slow magneto-sonic modes are found to be negligible and are ignored in the calculation. They suffer collisionless and collisional damping.
- The Alfvénic turbulence is, on the other hand, anisotropic -- as found in Goldreich&Sridhar (1995) -- which is the reason why these modes are very inefficient in confining particles, or, in other words, why they are subdominant in the calculation of the spatial D(E). They are here considered free of damping.


"Github_DiffusionCoefficient_MHD-modes_Halo.ipynb" is a Jupyter Notebook that calculates the D(E) in the Halo region (L_{Halo} = 6 kpc), where fast magnetosonic modes are damped via collisionless processes.

"Github_DiffusionCoefficient_MHD-modes_Disk.ipynb" is a Jupyter Notebook that calculates the D(E) in the Warm Ionized Medium (WIM) region (L_{WIM} = 900 pc), where fast magnetosonic modes are damped via both collisionless and collisional (viscous) processes.


