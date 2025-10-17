#!/usr/bin/env python3
# ============================================================
# 3D parabolic-band: E(k) and DOS with two Fermi levels
# Shared vertical axis (energy) and no k-ticks on band plot
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Physical parameters
# ------------------------------------------------------------
hbar = 1.0
m_eff = 1.0
k_vals = np.linspace(-1, 1, 400)
E_band = (hbar**2 * (k_vals**2 * 3)) / (2.0 * m_eff)
fermi_levels = [-0.2, 0.5]

# ------------------------------------------------------------
# 3D DOS
# ------------------------------------------------------------
def DOS(E):
    return (np.sqrt(2.0) / (np.pi**2)) * np.sqrt(np.maximum(E, 0.0))

E_vals = np.linspace(-0.5, np.max(E_band), 400)
DOS_vals = DOS(E_vals)
DOS_vals /= np.max(DOS_vals)
DOS_vals *= np.max(E_band)

# ------------------------------------------------------------
# Plot setup (shared y-axis for energy)
# ------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7, 4),
                       gridspec_kw={'width_ratios':[2,1]}, sharey=True)

fig.suptitle(r'3D Parabolic Band and DOS', fontsize=13, y=1.03)

# --- Left: Band structure ---
ax[0].plot(k_vals, E_band, color='tab:blue', lw=2)
for Ef in fermi_levels:
    ax[0].axhline(Ef, color='gray', ls='--', lw=1)
    ax[0].text(1.02, Ef, f'$E_F={Ef}$', va='center', fontsize=10, color='gray')

ax[0].set_xlabel(r'$k$')
ax[0].set_ylabel(r'$E$')
ax[0].set_xlim(-1, 1)
ax[0].grid(True, ls=':', lw=0.5)
ax[0].set_xticks([])
ax[0].set_xticklabels([])

# --- Right: DOS ---
ax[1].plot(DOS_vals, E_vals, color='tab:red', lw=2)
for Ef in fermi_levels:
    ax[1].axhline(Ef, color='gray', ls='--', lw=1)

ax[1].set_xlabel(r'$D(E)$ (arb. units)')
ax[1].grid(True, ls=':', lw=0.5)

plt.tight_layout()
plt.subplots_adjust(wspace=0.05)

# ✅ Save figure to file
plt.savefig("Band_DOS.png", dpi=300, bbox_inches='tight')
# plt.savefig("Band_DOS.pdf", bbox_inches='tight')  # alternative for vector output
plt.close()

print("✅ Figure saved as Band_DOS.png")
