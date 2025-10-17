#!/usr/bin/env python3
# ============================================================
# 3D parabolic-band conductivity:
# τ on coarse grid, f on homogeneous fine grid around each coarse point
# Integration domain: (-1,-1,-1) → (1,1,1)
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

outfile = 'DoubleGrid_Metal.pdf'

# ------------------------------------------------------------
# Physical parameters
# ------------------------------------------------------------
hbar = 1.0
m_eff = 1.0
mu = 0.50
T = 0.025

# ------------------------------------------------------------
# 3D parabolic DOS and Fermi–Dirac distribution
# ------------------------------------------------------------
def DOS(E):
    return (np.sqrt(2.0) / (np.pi**2)) * np.sqrt(np.maximum(E, 0.0))

def fermi(E, mu, T):
    return 1.0 / (np.exp((E - mu) / T) + 1.0)

# ------------------------------------------------------------
# σ integral with τ on coarse grid, f on homogeneous fine grid
# ------------------------------------------------------------
def sigma_tau_coarse_f_fine_homogeneous(nk_coarse, nk_fine):
    # Global homogeneous fine spacing
    df = 2.0 / (nk_coarse * nk_fine)
    dk = df * nk_fine

    # 1D homogeneous fine grid points and coarse grid points
    k_fine_1d = -1.0 + (np.arange(nk_coarse * nk_fine) + 0.5) * df
    k_coarse_1d = -1.0 + (np.arange(nk_coarse) + 0.5) * dk

    # 3D grids
    kx_c, ky_c, kz_c = np.meshgrid(k_coarse_1d, k_coarse_1d, k_coarse_1d, indexing="ij")
    E_c = (hbar**2 * (kx_c**2 + ky_c**2 + kz_c**2)) / (2.0 * m_eff)
    tau_c = 1.0 / DOS(E_c)

    # fine offsets relative to coarse centers
    fine_offsets = (np.arange(nk_fine) - nk_fine/2 + 0.5) * df
    kx_f, ky_f, kz_f = np.meshgrid(fine_offsets, fine_offsets, fine_offsets, indexing="ij")

    sigma_total = 0.0

    for i in range(nk_coarse):
        for j in range(nk_coarse):
            for k in range(nk_coarse):
                kx = kx_c[i, j, k] + kx_f
                ky = ky_c[i, j, k] + ky_f
                kz = kz_c[i, j, k] + kz_f
                E = (hbar**2 * (kx**2 + ky**2 + kz**2)) / (2.0 * m_eff)
                f = fermi(E, mu, T)
                vx = (hbar * kx) / m_eff
                fine_integral = np.sum(vx**2 * f * (1.0 - f))
                sigma_total += tau_c[i, j, k] * fine_integral

    return sigma_total * (df**3)

# ------------------------------------------------------------
# Run convergence
# ------------------------------------------------------------
nks = np.arange(10, 101, 10)
nks2 = np.arange(10, 61, 10)
nks_conv = np.arange(200, 201, 10)

sigmas_conv = [sigma_tau_coarse_f_fine_homogeneous(nk, 1) for nk in nks_conv]
sigmas_f1 = [sigma_tau_coarse_f_fine_homogeneous(nk, 1) for nk in nks]
sigmas_f4 = [sigma_tau_coarse_f_fine_homogeneous(nk, 4) for nk in nks2]
# sigmas_f8 = [sigma_tau_coarse_f_fine_homogeneous(nk, 8) for nk in nks2]

sigma_ref = sigmas_conv[-1]
errors_f1 = np.abs((np.array(sigmas_f1) - sigma_ref) / sigma_ref)
errors_f4 = np.abs((np.array(sigmas_f4) - sigma_ref) / sigma_ref)
# errors_f8 = np.abs((np.array(sigmas_f8) - sigma_ref) / sigma_ref)

# ------------------------------------------------------------
# Plot results
# ------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

fig.suptitle(r'Convergence of the conductivity of a Metal',
             fontsize=13, y=0.98)

# Left panel: σ vs nk
ax[0].semilogy(nks, sigmas_f1, 'o--', lw=1.2, label=r'$n_{\rm fine}=1$')
ax[0].semilogy(nks2, sigmas_f4, 's-', lw=1.2, label=r'$n_{\rm fine}=4$')
# ax[0].semilogy(nks2, sigmas_f8, 'd-', lw=1.2, label=r'$n_{\rm fine}=8$')
ax[0].set_xlabel(r'$n_k$ in each direction')
ax[0].set_ylabel(r'$\sigma_{xx}$')
ax[0].grid(True, which='both')
ax[0].legend()

# Right panel: relative error
ax[1].semilogy(nks, errors_f1, 'o--', lw=1.2, label=r'$n_{\rm fine}=1$')
ax[1].semilogy(nks2, errors_f4, 's-', lw=1.2, label=r'$n_{\rm fine}=4$')
# ax[1].semilogy(nks2, errors_f8, 'd-', lw=1.2, label=r'$n_{\rm fine}=8$')
ax[1].set_xlabel(r'$n_k$ in each direction')
ax[1].set_ylabel(r'Relative error')
ax[1].grid(True, which='both')
ax[1].legend()

plt.tight_layout()
plt.savefig(outfile, bbox_inches='tight')
plt.show()