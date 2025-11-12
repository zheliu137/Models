# Execute the updated D6-symmetry script and display the IBZ plot inline.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# ============================================================
# 1. User Parameters
# ============================================================
nkc = 4       # coarse grid divisions per direction
nkf = 12      # fine grid divisions per direction

# Check commensurability
if nkf % nkc != 0:
    raise SystemExit(f"Error: nkf ({nkf}) must be a multiple of nkc ({nkc}) for commensurate grids.")

Nc = nkc * nkc
Nf = nkf * nkf

# ============================================================
# 2. Reciprocal Lattice Basis
# ============================================================
b1 = np.array([1.0, 0.0])
b2 = np.array([0.5, np.sqrt(3)/2])
B = np.column_stack([b1, b2])
B_inv = np.linalg.inv(B)

def frac_to_cart(frac):
    return (B @ frac.T).T

def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],[s, c]])

def mirror(theta):
    """Mirror across line rotated by θ from x-axis."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],[s, c]])
    M = np.array([[1,0],[0,-1]])
    return R @ M @ R.T

def wrap01(fr):
    """Wrap fractional coords to [0,1) and round to fine-grid resolution."""
    f = fr - np.floor(fr)
    f = np.round(f*nkf)/nkf
    f = np.mod(f, 1.0)
    f[np.isclose(f, 0.0, atol=1e-12)] = 0.0
    return f

# ============================================================
# 3. Generate Coarse and Fine Grids
# ============================================================
uc = np.arange(0, 1, 1/nkc)
vc = np.arange(0, 1, 1/nkc)
Uc, Vc = np.meshgrid(uc, vc, indexing='ij')
coarse_frac = np.stack([Uc.ravel(), Vc.ravel()], axis=1)
coarse_cart = frac_to_cart(coarse_frac)

uf = np.linspace(0, 1, nkf+1)
vf = np.linspace(0, 1, nkf+1)
Uf, Vf = np.meshgrid(uf, vf, indexing='ij')
fine_frac_all = np.stack([Uf.ravel(), Vf.ravel()], axis=1)
fine_cart_all = frac_to_cart(fine_frac_all)

# ============================================================
# 4. Define red coarse points (A_r) by 6 rotations of (1/nkc, 1/nkc)
# ============================================================
p_frac = np.array([1/nkc, 1/nkc])
p_cart = B @ p_frac
Ar_frac = set()
for k in range(6):
    f = B_inv @ (rot(k*np.pi/3) @ p_cart)
    f = wrap01(np.round(f*nkc)/nkc)
    Ar_frac.add((float(f[0]), float(f[1])))
Ar_frac = np.array(sorted(list(Ar_frac)))

def find_idx(fr):
    d = np.linalg.norm(coarse_frac - fr, axis=1)
    return int(np.argmin(d))

Ar_indices = np.array([find_idx(fr) for fr in Ar_frac])
red_set = set(Ar_indices.tolist())

# ============================================================
# 5. Determine nearest coarse points for each fine point
# ============================================================
def nearest_sets(query_cart, base_cart, base_ids, rng=range(-2,3)):
    shifts = np.array([(i,j) for i in rng for j in rng])
    shift_cart = (B @ shifts.T).T
    copies = (base_cart[:,None,:] + shift_cart[None,:,:]).reshape(-1,2)
    copy_ids = np.repeat(base_ids, len(shifts))
    d = np.linalg.norm(query_cart[:,None,:] - copies[None,:,:], axis=2)
    minv = d.min(axis=1)
    near = np.isclose(d, minv[:,None], atol=1e-10)
    groups = []
    for i in range(query_cart.shape[0]):
        groups.append(np.unique(copy_ids[near[i]]))
    return groups

all_ids = np.arange(len(coarse_cart))
near_all = nearest_sets(fine_cart_all, coarse_cart, all_ids)

Nr = np.array([len(set(g) & red_set) for g in near_all])
Ns = np.array([len(set(g)) for g in near_all])
is_near_red = np.array([len(set(g) & red_set) > 0 for g in near_all])
is_blue = (Nr == 1) & (Ns == 1) & is_near_red
is_green = is_near_red & ~is_blue

# ============================================================
# 6. Construct Adaptive Set A_adp
# ============================================================
rows = []

# Coarse black (exclude red)
for idx, fr in enumerate(coarse_frac):
    if idx in red_set:
        continue
    rows.append({"type":"coarse_black","u":fr[0],"v":fr[1],"weight":1.0/Nc})

# Fine near-red (blue & green)
for fr, blue, nr, ns in zip(fine_frac_all, is_blue, Nr, Ns):
    if not is_near_red[np.where((fine_frac_all==fr).all(axis=1))[0][0]]:
        continue
    if blue:
        Nw = (Nf / Nc)
        t = "fine_blue"
    else:
        Nw = (Nf / Nc) * (ns / max(nr,1))
        t = "fine_green"
    w = 1.0 / (Nc * Nw)
    rows.append({"type":t,"u":fr[0],"v":fr[1],"weight":w})

A_adp = pd.DataFrame(rows).sort_values(by=["type","u","v"]).reset_index(drop=True)

# ============================================================
# 7. Reduce to IBZ using full D6 symmetries (6 rotations + 6 mirrors)
# ============================================================
rots = [rot(k*np.pi/3) for k in range(6)]
mirs = [mirror(k*np.pi/6) for k in range(6)]
ops = rots + mirs  # 12 operations

def canonical_rep(u, v):
    cart = B @ np.array([u, v])
    reps = []
    for Op in ops:
        fr = wrap01(B_inv @ (Op @ cart))
        reps.append(tuple(fr))
    return min(reps)

groups = {}
for _, row in A_adp.iterrows():
    rep = canonical_rep(row.u, row.v)
    groups.setdefault(rep, 0.0)
    groups[rep] += row.weight

IBZ = pd.DataFrame(
    [{"u":r[0], "v":r[1], "weight":w} for r,w in groups.items()]
).sort_values(by=["u","v"]).reset_index(drop=True)

# ============================================================
# 8. Plot IBZ (black / blue / green) with dashed FBZ outline
# ============================================================
def nearest_type(fr):
    d = np.linalg.norm(A_adp[["u","v"]].values - fr, axis=1)
    idx = np.argmin(d)
    return A_adp.iloc[idx]["type"]

IBZ["type"] = [nearest_type(fr) for fr in IBZ[["u","v"]].values]

fig, ax = plt.subplots(figsize=(6,6))
for typ, color, size in [("coarse_black","k",90),("fine_blue","blue",40),("fine_green","green",40)]:
    sub = IBZ[IBZ["type"]==typ]
    if len(sub):
        cart = frac_to_cart(sub[["u","v"]].values)
        ax.scatter(cart[:,0], cart[:,1], s=size, c=color, marker='o')

# Dark yellow dashed lines
p1_frac = np.array([[0,0],[1/3,1/3]])
p2_frac = np.array([[1/3,1/3],[0,0.5]])
p1 = frac_to_cart(p1_frac)
p2 = frac_to_cart(p2_frac)

ax.plot(p1[:,0], p1[:,1], linestyle='--', linewidth=2, color='goldenrod')
ax.plot(p2[:,0], p2[:,1], linestyle='--', linewidth=2, color='goldenrod')

corners_frac = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])
corners = frac_to_cart(corners_frac)
ax.plot(corners[:,0], corners[:,1], linestyle='--', linewidth=2, color=(0,0,0,0.5))
ax.set_aspect('equal')
ax.set_axis_off()
plt.tight_layout(pad=0)
plt.savefig("Adp_IBZ.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
