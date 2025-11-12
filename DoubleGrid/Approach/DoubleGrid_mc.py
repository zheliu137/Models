# Add legend (bottom-right corner) for all point styles except WS hexagon
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import matplotlib.pyplot as plt

# Reuse the previously computed arrays and masks (short version, same logic as last cell)
nkc = 4
b1 = np.array([1.0, 0.0])
b2 = np.array([0.5, np.sqrt(3)/2])
B = np.column_stack([b1, b2])
B_inv = np.linalg.inv(B)
def frac_to_cart(frac): return (B @ frac.T).T
def cart_to_frac(cart): return (B_inv @ cart.T).T
def wrap01(fr):
    f = np.mod(fr, 1.0)
    f[np.isclose(f, 1.0, atol=1e-10)] = 0.0
    return f
def rot(theta): c, s = np.cos(theta), np.sin(theta); return np.array([[c, -s],[s, c]])
def mirror(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],[s, c]]); M = np.array([[1,0],[0,-1]]); return R @ M @ R.T

u = np.arange(0, 1, 1/nkc); v = np.arange(0, 1, 1/nkc)
U, V = np.meshgrid(u, v, indexing='ij')
coarse_frac = np.stack([U.ravel(), V.ravel()], axis=1); coarse_cart = frac_to_cart(coarse_frac)

rots = [rot(k*np.pi/3) for k in range(6)]; mirs = [mirror(k*np.pi/6) for k in range(6)]
ops = rots + mirs
def canonical_rep(fr_uv):
    cart = frac_to_cart(np.array([fr_uv]))[0]
    reps = [wrap01(cart_to_frac(Op @ cart)) for Op in ops]
    reps = np.array(reps); reps_rounded = np.round(reps, 8)
    min_idx = np.lexsort((reps_rounded[:,1], reps_rounded[:,0]))[0]
    return tuple(reps_rounded[min_idx])
groups = {}
for fr in coarse_frac:
    rep = canonical_rep(fr)
    fr_key = tuple(np.round(fr, 8)); rep_key = tuple(np.round(rep, 8))
    if rep_key not in groups: groups[rep_key] = []
    groups[rep_key].append(fr_key)
IBZ = np.array(list(groups.keys()))
ibz_keys = [tuple(rep) for rep in IBZ]
site_is_ibz = np.array([any(np.allclose(fr, rep, atol=1e-8) for rep in ibz_keys)
                        for fr in coarse_frac])

# Γ-cell WS fine grid
nt = 1; nref = 3
uc = np.linspace(-nt/nkc, nt/nkc, 2*nt + 1)
vc = np.linspace(-nt/nkc, nt/nkc, 2*nt + 1)
Uc, Vc = np.meshgrid(uc, vc, indexing='ij')
coarse_frac_local = np.stack([Uc.ravel(), Vc.ravel()], axis=1)
coarse_cart_local = frac_to_cart(coarse_frac_local)
fine_range = np.linspace(-nt/nkc, nt/nkc, 2*nt*nref + 1)
Uf, Vf = np.meshgrid(fine_range, fine_range, indexing='ij')
fine_frac = np.stack([Uf.ravel(), Vf.ravel()], axis=1)
fine_cart = frac_to_cart(fine_frac)

tol = 1e-15; colors = []; center_idx = np.argmin(np.linalg.norm(coarse_cart_local, axis=1))
for fc in fine_cart:
    dists = np.linalg.norm(coarse_cart_local - fc, axis=1)
    min_dist = np.min(dists)
    nearest = np.isclose(dists, min_dist, atol=tol)
    n_nearest = np.sum(nearest)
    if nearest[center_idx] and n_nearest == 1: colors.append('blue')
    elif nearest[center_idx] and n_nearest > 1: colors.append('green')
    else: colors.append('orange')
is_green = np.array([c == 'green' for c in colors])
is_blue  = np.array([c == 'blue'  for c in colors])
in_central = (np.abs(fine_frac[:,0]) <= 1/nkc + 1e-12) & (np.abs(fine_frac[:,1]) <= 1/nkc + 1e-12)
green_pts = fine_cart[is_green & in_central]
blue_pts  = fine_cart[is_blue  & in_central]
angles = np.arctan2(green_pts[:,1], green_pts[:,0]); order = np.argsort(angles)
ws_poly = green_pts[order]; ws_poly_closed = np.vstack([ws_poly, ws_poly[0]])

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(coarse_cart[:,0], coarse_cart[:,1], s=120, facecolors='none',
           edgecolors='black', linewidths=1.5, marker='o', zorder=3)
ibz_cart = frac_to_cart(IBZ)
ax.scatter(ibz_cart[:,0], ibz_cart[:,1], s=80, c='black', marker='o', zorder=4)
for rc in coarse_cart:
    poly = ws_poly_closed + rc
    ax.plot(poly[:,0], poly[:,1], 'r--', linewidth=1.2, zorder=2)

alpha_val = 0.95; size_val = 30; lw_val = 1.7
for rc in frac_to_cart(coarse_frac[site_is_ibz]):
    b = blue_pts + rc; g = green_pts + rc
    ax.scatter(b[:,0], b[:,1], s=size_val, facecolors='blue', edgecolors='none',
               alpha=alpha_val, marker='o', zorder=5)
    ax.scatter(g[:,0], g[:,1], s=size_val, facecolors='green', edgecolors='none',
               alpha=alpha_val, marker='o', zorder=6)
for rc in frac_to_cart(coarse_frac[~site_is_ibz]):
    b = blue_pts + rc; g = green_pts + rc
    ax.scatter(b[:,0], b[:,1], s=size_val, facecolors='none', edgecolors='blue',
               linewidths=lw_val, alpha=alpha_val, marker='o', zorder=5)
    ax.scatter(g[:,0], g[:,1], s=size_val, facecolors='none', edgecolors='green',
               linewidths=lw_val, alpha=alpha_val, marker='o', zorder=6)
corners_frac = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])
corners = frac_to_cart(corners_frac)
ax.plot(corners[:,0], corners[:,1], '--', color=(0,0,0,0.5), linewidth=1.5, zorder=1)

# Legend (bottom-right)
black_solid = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='IBZ coarse')
black_hollow = mlines.Line2D([], [], color='black', marker='o', markerfacecolor='none', linestyle='None', markersize=10, label='Non-IBZ coarse')
blue_solid = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=7, label='IBZ microcell')
green_solid = mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=7, label='IBZ microcell')
blue_hollow = mlines.Line2D([], [], color='blue', marker='o', markerfacecolor='none', linestyle='None', markersize=7, label='non-IBZ microcell')
green_hollow = mlines.Line2D([], [], color='green', marker='o', markerfacecolor='none', linestyle='None', markersize=7, label='non-IBZ microcell')
ax.legend(handles=[black_solid, black_hollow, blue_solid, green_solid, blue_hollow, green_hollow],
          loc='lower right', fontsize=15, frameon=False)

ax.set_aspect('equal'); ax.set_axis_off()
plt.tight_layout(pad=0)

out_path = "fullBZ_WSmc.png"
plt.savefig(out_path, dpi=220, bbox_inches="tight")
plt.show()