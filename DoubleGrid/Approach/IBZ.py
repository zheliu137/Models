import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Parameters and Reciprocal Basis
# ============================================================
nkc = 4
b2factor = 1.0
nref2 = 3
nref1 = nref2 * int(b2factor)
nt = 1

b1 = np.array([1.0, 0.0])
b2 = np.array([0.5, np.sqrt(3)/2])*b2factor
B = np.column_stack([b1, b2])
B_inv = np.linalg.inv(B)

def frac_to_cart(frac):
    return (B @ frac.T).T

def cart_to_frac(cart):
    return (B_inv @ cart.T).T

def wrap01(fr):
    f = np.mod(fr, 1.0)
    f[np.isclose(f, 1.0, atol=1e-10)] = 0.0
    return f

def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],[s, c]])

def mirror(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],[s, c]])
    M = np.array([[1,0],[0,-1]])
    return R @ M @ R.T

# ============================================================
# 2. Generate coarse grid
# ============================================================
u = np.arange(0, 1, 1/nkc)
v = np.arange(0, 1, 1/nkc)
U, V = np.meshgrid(u, v, indexing='ij')
coarse_frac = np.stack([U.ravel(), V.ravel()], axis=1)
coarse_cart = frac_to_cart(coarse_frac)

# ============================================================
# 3. Build D6 symmetry operations (rotations + mirrors) and IBZ
# ============================================================
rots = [rot(k*np.pi/3) for k in range(6)]
mirs = [mirror(k*np.pi/6) for k in range(6)]
ops = rots + mirs

def canonical_rep(fr_uv):
    cart = frac_to_cart(np.array([fr_uv]))[0]
    reps = []
    for Op in ops:
        f_trans = wrap01(cart_to_frac(Op @ cart))
        reps.append(f_trans)
    reps = np.array(reps)
    reps_rounded = np.round(reps, 8)
    min_idx = np.lexsort((reps_rounded[:,1], reps_rounded[:,0]))[0]
    return tuple(reps_rounded[min_idx])

groups = {}
for fr in coarse_frac:
    rep = canonical_rep(fr)
    for key in groups.keys():
        if np.allclose(key, rep, atol=1e-8):
            rep = key
            break
    groups.setdefault(rep, []).append(tuple(fr))
IBZ = np.array(list(groups.keys()))

# ============================================================
# 4. Compute WS microcell boundary at Γ automatically
# ============================================================
uc = np.linspace(-nt/nkc, nt/nkc, 2*nt + 1)
vc = np.linspace(-nt/nkc, nt/nkc, 2*nt + 1)
Uc, Vc = np.meshgrid(uc, vc, indexing='ij')
coarse_frac_local = np.stack([Uc.ravel(), Vc.ravel()], axis=1)
coarse_cart_local = frac_to_cart(coarse_frac_local)

fine_range1 = np.linspace(-nt/nkc, nt/nkc, 2*nt*nref1 + 1)
fine_range2 = np.linspace(-nt/nkc, nt/nkc, 2*nt*nref2 + 1)
Uf, Vf = np.meshgrid(fine_range1, fine_range2, indexing='ij')
fine_frac = np.stack([Uf.ravel(), Vf.ravel()], axis=1)
fine_cart = frac_to_cart(fine_frac)

tol = 1e-15
subweights, colors = [], []
center_idx = np.argmin(np.linalg.norm(coarse_cart_local, axis=1))

for fc in fine_cart:
    dists = np.linalg.norm(coarse_cart_local - fc, axis=1)
    min_dist = np.min(dists)
    nearest = np.isclose(dists, min_dist, atol=tol)
    n_nearest = np.sum(nearest)
    if nearest[center_idx] and n_nearest == 1:
        subweights.append(1); colors.append('blue')
    elif nearest[center_idx] and n_nearest > 1:
        subweights.append(n_nearest); colors.append('green')
    else:
        subweights.append(0); colors.append('orange')

is_green = np.array([c == 'green' for c in colors])
in_central = (np.abs(fine_frac[:,0]) <= 1/nkc + 1e-12) & (np.abs(fine_frac[:,1]) <= 1/nkc + 1e-12)
green_idx = np.where(is_green & in_central)[0]
green_pts_cart = fine_cart[green_idx]

# Order the green points (WS polygon) by polar angle
angles = np.arctan2(green_pts_cart[:,1], green_pts_cart[:,0])
order = np.argsort(angles)
ws_poly = green_pts_cart[order]
ws_poly_closed = np.vstack([ws_poly, ws_poly[0]])


# Plot dashed FBZ parallelogram spanning fractional [0,0]→[1,1]

# Parallelogram corners in fractional coords
para_frac = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [0.0, 0.0]
])
para_cart = frac_to_cart(para_frac)

fig, ax = plt.subplots(figsize=(7,7))

# Hollow coarse grid
ax.scatter(coarse_cart[:,0], coarse_cart[:,1], s=80, facecolors='none',
           edgecolors='black', linewidths=1.2, marker='o', zorder=3)

# Filled IBZ representatives
cart_ibz = frac_to_cart(IBZ)
ax.scatter(cart_ibz[:,0], cart_ibz[:,1], s=80, c='black', marker='o', zorder=4)

# Draw FBZ parallelogram (black dashed)
ax.plot(para_cart[:,0], para_cart[:,1], 'k--', linewidth=1.5, zorder=2)

# Emphasize Γ center
ax.scatter([0],[0], s=40, c='black', zorder=6)

ax.set_aspect('equal')
ax.set_axis_off()
plt.tight_layout(pad=0)

# out_path_para = "/mnt/data/ibz_ws_microcells_gamma_parallelogram.png"
# plt.savefig(out_path_para, dpi=200, bbox_inches="tight")
plt.show()