import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# -------------------------
# Hexagonal reciprocal basis
# -------------------------
b1 = np.array([1.0, 0.0])
b2 = np.array([0.5, np.sqrt(3)/2])
B = np.column_stack([b1, b2])
B_inv = np.linalg.inv(B)

# -------------------------
# Coarse / fine grids
# -------------------------
n_coarse = 5
n_fine = 12
frac_coarse = np.linspace(0, 1, n_coarse)
frac_fine = np.linspace(0, 1, n_fine + 1)

Uc, Vc = np.meshgrid(frac_coarse, frac_coarse, indexing='ij')
Uf, Vf = np.meshgrid(frac_fine, frac_fine, indexing='ij')

coarse_frac = np.stack([Uc.ravel(), Vc.ravel()], axis=1)
coarse_pts  = (B @ coarse_frac.T).T
fine_frac   = np.stack([Uf.ravel(), Vf.ravel()], axis=1)
fine_pts    = (B @ fine_frac.T).T

# -------------------------
# Red coarse points (A_r)
# -------------------------
def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],[s, c]])

p22_frac = np.array([frac_coarse[1], frac_coarse[1]])  # (0.25, 0.25)
p22 = B @ p22_frac
Ar_mask = np.zeros(len(coarse_pts), dtype=bool)
for k in range(6):
    pr = rot(k*np.pi/3) @ p22
    frac = B_inv @ pr
    frac_wrapped = frac - np.floor(frac)
    snap = np.round(frac_wrapped * 4)/4.0
    snap[np.isclose(snap, 0.0, atol=1e-12)] = 0.0
    snap[np.isclose(snap, 1.0, atol=1e-12)] = 1.0
    idx = np.argmin(np.linalg.norm(coarse_frac - snap, axis=1))
    Ar_mask[idx] = False

# -------------------------
# FBZ outline
# -------------------------
corners_frac = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])
corners = (B @ corners_frac.T).T

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(6,6))


# Coarse grid (black) and A_r (highlighted red)
ax.scatter(coarse_pts[~Ar_mask,0], coarse_pts[~Ar_mask,1], s=90, c='k', marker='o', zorder=1)
ax.scatter(coarse_pts[Ar_mask,0],  coarse_pts[Ar_mask,1],  s=140, c='red', edgecolors='k', linewidths=1.5, marker='o', zorder=3)
# Fine grid (orange)
ax.scatter(fine_pts[:,0], fine_pts[:,1], s=12, c='orange', marker='o', zorder=5)

# FBZ boundary
ax.plot(corners[:,0], corners[:,1], linestyle='--', linewidth=2, color=(0,0,0,0.5), zorder=4)

ax.set_aspect('equal', adjustable='box')
ax.set_axis_off()

pad = 0.05
x_min, x_max = fine_pts[:,0].min()-pad, fine_pts[:,0].max()+pad
y_min, y_max = fine_pts[:,1].min()-pad, fine_pts[:,1].max()+pad
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.tight_layout(pad=0)
plt.savefig("Doublegrid.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()


