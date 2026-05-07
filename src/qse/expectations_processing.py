import numpy as np
import itertools as it

from sympy import Matrix


def assemble_S_matrix(S_expval, results, krylov_dim, prefactors):
    # Assemble S, the overlap matrix of dimension D:
    S_first_row = np.zeros(krylov_dim, dtype=complex)
    S_first_row[0] = S_expval

    # Add in ancilla-only measurements:
    for i in range(krylov_dim - 1):
        # Get expectation values from experiment
        expval_real = results.data.evs[0][0][i]  # automatic extrapolated evs if ZNE is used
        expval_imag = results.data.evs[1][0][i]  # automatic extrapolated evs if ZNE is used

        # Get expectation values
        expval = expval_real + 1j * expval_imag
        S_first_row[i + 1] += prefactors[i] * expval

    S_circ = np.zeros((krylov_dim, krylov_dim), dtype=complex)

    # Distribute entries from first row across matrix:
    for i, j in it.product(range(krylov_dim), repeat=2):
        if i >= j:
            S_circ[j, i] = S_first_row[i - j]
        else:
            S_circ[j, i] = np.conj(S_first_row[j - i])

    return S_circ


def assemble_H_matrix(H_expval, H_op, results, krylov_dim, prefactors):
    # Assemble H
    H_first_row = np.zeros(krylov_dim, dtype=complex)
    H_first_row[0] = H_expval

    for obs_idx, (pauli, coeff) in enumerate(zip(H_op.paulis, H_op.coeffs)):
        # Add in ancilla-only measurements:
        for i in range(krylov_dim - 1):
            # Get expectation values from experiment
            expval_real = results.data.evs[2 + 2 * obs_idx][0][
                i
            ]  # automatic extrapolated evs if ZNE is used
            expval_imag = results.data.evs[2 + 2 * obs_idx + 1][0][
                i
            ]  # automatic extrapolated evs if ZNE is used

            # Get expectation values
            expval = expval_real + 1j * expval_imag
            H_first_row[i + 1] += prefactors[i] * coeff * expval

    H_eff_circ = np.zeros((krylov_dim, krylov_dim), dtype=complex)

    # Distribute entries from first row across matrix:
    for i, j in it.product(range(krylov_dim), repeat=2):
        if i >= j:
            H_eff_circ[j, i] = H_first_row[i - j]
        else:
            H_eff_circ[j, i] = np.conj(H_first_row[j - i])

    return H_eff_circ


def flat(I, n, s):
    return I * (s + 1) + n


def assemble_H_and_S_multiref(d_refs, s, meta_mr, results_mr, H_op):
    N = d_refs * (s + 1)

    S_mr = np.zeros((N, N), dtype=complex)
    H_mr = np.zeros((N, N), dtype=complex)

    for idx, (I, J, m, n) in enumerate(meta_mr):
        res = results_mr[idx]

        S_re = float(np.asarray(res.data.evs[0]).item())
        S_im = float(np.asarray(res.data.evs[1]).item())
        S_val = S_re + 1j * S_im

        H_val = 0.0 + 0.0j
        for obs_idx, coeff in enumerate(H_op.coeffs):
            re = float(np.asarray(res.data.evs[2 + 2 * obs_idx]).item())
            im = float(np.asarray(res.data.evs[2 + 2 * obs_idx + 1]).item())
            H_val += coeff * (re + 1j * im)

        a = flat(I, m, s)
        b = flat(J, n, s)
        S_mr[a, b] = S_val
        H_mr[a, b] = H_val

    # Hermitize (good practice)
    S_mr = (S_mr + S_mr.conj().T) / 2
    H_mr = (H_mr + H_mr.conj().T) / 2

    return S_mr, H_mr

