import numpy as np
from neural_compiler import ExecutableMLP

def build_policy():
    W1 = np.zeros((600, 32), dtype=np.float32)
    b1 = np.zeros(32, dtype=np.float32)
    W2 = np.zeros((32, 32), dtype=np.float32)
    b2 = np.zeros(32, dtype=np.float32)
    W3 = np.zeros((32, 6), dtype=np.float32)
    b3 = np.zeros(6, dtype=np.float32)
    def idx(y, x, c):
        return (y * 10 + x) * 6 + c
    # Detect the submarine at the surface. Channel 0 includes both halves.
    for x in range(10):
        W1[idx(0, x, 0), 0] = 3.0
    b1[0] = -1.5
    W2[0, 0] = 3.0
    # Descend from the surface; otherwise keep firing.
    W3[0, 4] = 5.0
    b3[4] = 1.0
    b3[5] = 4.0
    return ExecutableMLP(W1, b1, W2, b2, W3, b3)
