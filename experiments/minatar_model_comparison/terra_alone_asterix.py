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
    # Detect the player's row and nearby enemies and gold in each row.
    for row in range(8):
        y = row + 1
        for x in range(10):
            W1[idx(y, x, 0), row] = 3.0
        b1[row] = -1.5
        for x in range(3, 8):
            W1[idx(y, x, 1), 8 + row] = 3.0
            W1[idx(y, x, 2), 16 + row] = 3.0
        b1[8 + row] = -1.5
        b1[16 + row] = -1.5
    # Approximate the player's horizontal displacement from the center.
    for y in range(1, 9):
        for x in range(10):
            W1[idx(y, x, 0), 24] = (x - 4.5) * 0.5
    # AND gates for player row and a row's enemy or gold detection.
    for row in range(8):
        W2[row, row] = 3.0
        W2[8 + row, row] = 3.0
        b2[row] = -4.5
        W2[row, 22 + row] = 3.0
        W2[16 + row, 22 + row] = 3.0
        b2[22 + row] = -4.5
    for row in range(1, 8):
        unit = 8 + row - 1
        W2[row, unit] = 3.0
        W2[8 + row - 1, unit] = 3.0
        b2[unit] = -4.5
    for row in range(7):
        unit = 15 + row
        W2[row, unit] = 3.0
        W2[8 + row + 1, unit] = 3.0
        b2[unit] = -4.5
    W2[24, 30] = 2.0
    # Convert each tanh gate into an approximately 0/1 contribution.
    def contribute(unit, action, amount):
        W3[unit, action] += amount / 2.0
        b3[action] += amount / 2.0
    b3[0] = 2.5
    b3[1] = -1.0
    b3[2] = -0.5
    b3[3] = -1.0
    b3[4] = -0.5
    b3[5] = -4.0
    for row in range(8):
        if row > 0:
            contribute(row, 2, 5.0)
        if row < 7:
            contribute(row, 4, 5.0)
        contribute(22 + row, 0, 1.5)
    for unit in range(8, 15):
        contribute(unit, 2, -7.0)
    for unit in range(15, 22):
        contribute(unit, 4, -7.0)
    W3[30, 1] = 2.0
    W3[30, 3] = -2.0
    return ExecutableMLP(W1, b1, W2, b2, W3, b3)
