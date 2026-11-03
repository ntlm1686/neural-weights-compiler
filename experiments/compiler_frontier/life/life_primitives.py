"""Two weight-writing primitives used by the Game-of-Life solution (explicit formulas, no fitting).

TORUS_BOX_MEAN_ATTN : one attention head that, at every cell token, averages a value register over the
                      3x3 torus neighbourhood (self included).
INTEGER_LUT_MLP     : a ReLU MLP that evaluates an arbitrary function on a small set of integer inputs
                      z = w . x (linear in the residual) as a piecewise-linear interpolant through the
                      points (z_k, f(z_k)); exact on integers.
"""
import numpy as np


def torus_box_mean_attn(model, layer, head, row_reg, col_reg, val_reg, out_reg, n_rows, n_cols, margin=40.0):
    """Score(i, j) = margin * ([row_j in {r_i-1, r_i, r_i+1}] + [col_j in {c_i-1, c_i, c_i+1}])  (mod size).

    The 9 cells of the 3x3 box score 2*margin, every other cell <= margin, so softmax gives each box cell
    weight 1/9 up to a relative error <= (N-9)/9 * exp(-margin).  Q = (row band matrix, col band matrix) applied
    to the query's position one-hots, K = the key's position one-hots, V = val_reg.
    """
    W, rs = model.weights, model.register_slices
    sq = np.sqrt(model.d_head)
    WQ, WK = W[f"L{layer}.H{head}.W_Q"], W[f"L{layer}.H{head}.W_K"]
    WV, WO = W[f"L{layer}.H{head}.W_V"], W[f"L{layer}.H{head}.W_O"]
    r0, c0 = rs[row_reg].start, rs[col_reg].start
    dim = 0
    for reg0, n in ((r0, n_rows), (c0, n_cols)):
        for j in range(n):            # head dim `dim` is "key has coordinate j"
            WK[reg0 + j, dim] = 1.0
            for r in range(n):        # query at coordinate r wants j in r-1, r, r+1 (mod n)
                if (j - r) % n in (0, 1, n - 1):
                    WQ[reg0 + r, dim] = margin * sq
            dim += 1
    WV[rs[val_reg].start, dim] = 1.0
    WO[dim, rs[out_reg].start] = 1.0


def integer_lut_mlp(model, layer, in_coeffs, points, out_reg, cursor=0):
    """Adds f(z) to out_reg where z = sum_reg coeff * reg (+ coeff on const_one) is an integer,
    `points` = sorted list of (z_k, f(z_k)) covering every reachable z (consecutive z_k differ by 1).
    Uses the ReLU expansion f(z) = f(z_0) + sum_k (slope_k - slope_{k-1}) * ReLU(z - z_k)."""
    W, rs = model.weights, model.register_slices
    Wfc, bfc, Wp = W[f"L{layer}.W_fc"], W[f"L{layer}.b_fc"], W[f"L{layer}.W_proj"]
    o = rs[out_reg].start
    zs = [p[0] for p in points]
    fs = [p[1] for p in points]
    # constant term f(z_0): a unit that is 1 everywhere (ReLU(const_one))
    units = [(None, fs[0])]
    prev_slope = 0.0
    for k in range(len(zs) - 1):
        slope = (fs[k + 1] - fs[k]) / (zs[k + 1] - zs[k])
        if slope != prev_slope:
            units.append((zs[k], slope - prev_slope))
        prev_slope = slope
    for knot, coef in units:
        if coef == 0:
            continue
        if knot is None:
            Wfc[rs["const_one"].start, cursor] = 1.0
        else:
            for reg, c in in_coeffs.items():
                Wfc[rs[reg].start, cursor] += c
            bfc[cursor] = -knot
        Wp[cursor, o] = coef
        cursor += 1
    return cursor
