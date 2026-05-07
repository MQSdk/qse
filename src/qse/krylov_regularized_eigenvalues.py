import numpy as np
import scipy as sp
import itertools as it

from typing import Union, List
from qiskit.quantum_info import SparsePauliOp, Pauli
from common.expectations_processing import flat


def solve_generalized_eig(H_eff_circ, S_circ, krylov_dim, threshold):
    gnd_en_circ_est_list = []
    for d in range(1, krylov_dim + 1):
        # Solve generalized eigenvalue problem
        gnd_en_circ_est = solve_regularized_gen_eig(
            H_eff_circ[:d, :d], S_circ[:d, :d], threshold=threshold
        )
        gnd_en_circ_est_list.append(gnd_en_circ_est)
        print(f"Estimated ground state energy for d={d}: ", gnd_en_circ_est)

    return gnd_en_circ_est_list


def solve_generalized_eig_multi(S_mr, H_mr, krylov_dim, s, d_refs, threshold=1e-6):
    gnd_mr_list = []
    for k in range(1, krylov_dim + 1):
        keep = []
        for I in range(d_refs):
            for n in range(k):
                keep.append(flat(I, n, s))
        keep = np.array(keep, dtype=int)

        S_k = S_mr[np.ix_(keep, keep)]
        H_k = H_mr[np.ix_(keep, keep)]

        gnd = solve_regularized_gen_eig(H_k, S_k, threshold=1e-6)
        gnd_mr_list.append(gnd)
        print(f"Estimated ground energy (k={k}, dim={len(keep)}):", gnd)
    return gnd_mr_list


def solve_regularized_gen_eig(
    h: np.ndarray,
    s: np.ndarray,
    threshold: float,
    k: int = 1,
    return_dimn: bool = False,
) -> Union[float, List[float]]:
    """
    Method for solving the generalized eigenvalue problem with regularization

    Args:
        h (numpy.ndarray):
            The effective representation of the matrix in our Krylov subspace
        s (numpy.ndarray):
            The matrix of overlaps between vectors of our Krylov subspace
        threshold (float):
            Cut-off value for the eigenvalue of s
        k (int):
            Number of eigenvalues to return
        return_dimn (bool):
            Whether to return the size of the regularized subspace

    Returns:
        lowest k-eigenvalue(s) that are the solution of the regularized generalized eigenvalue problem


    """
    s_vals, s_vecs = sp.linalg.eigh(s)
    s_vecs = s_vecs.T
    good_vecs = np.array([vec for val, vec in zip(s_vals, s_vecs) if val > threshold])
    h_reg = good_vecs.conj() @ h @ good_vecs.T
    s_reg = good_vecs.conj() @ s @ good_vecs.T
    if k == 1:
        if return_dimn:
            return sp.linalg.eigh(h_reg, s_reg)[0][0], len(good_vecs)
        else:
            return sp.linalg.eigh(h_reg, s_reg)[0][0]
    else:
        if return_dimn:
            return sp.linalg.eigh(h_reg, s_reg)[0][:k], len(good_vecs)
        else:
            return sp.linalg.eigh(h_reg, s_reg)[0][:k]


def single_particle_gs(H_op, n_qubits):
    """
    Find the ground state of the single particle(excitation) sector
    """
    H_x = []
    for p, coeff in H_op.to_list():
        H_x.append(set([i for i, v in enumerate(Pauli(p).x) if v]))

    H_z = []
    for p, coeff in H_op.to_list():
        H_z.append(set([i for i, v in enumerate(Pauli(p).z) if v]))

    H_c = H_op.coeffs

    print("n_sys_qubits", n_qubits)

    n_exc = 1
    sub_dimn = int(sp.special.comb(n_qubits + 1, n_exc))
    print("n_exc", n_exc, ", subspace dimension", sub_dimn)

    few_particle_H = np.zeros((sub_dimn, sub_dimn), dtype=complex)

    sparse_vecs = [
        set(vec) for vec in it.combinations(range(n_qubits + 1), r=n_exc)
    ]  # list all of the possible sets of n_exc indices of 1s in n_exc-particle states

    m = 0
    for i, i_set in enumerate(sparse_vecs):
        for j, j_set in enumerate(sparse_vecs):
            m += 1

            if len(i_set.symmetric_difference(j_set)) <= 2:
                for p_x, p_z, coeff in zip(H_x, H_z, H_c):
                    if i_set.symmetric_difference(j_set) == p_x:
                        sgn = ((-1j) ** len(p_x.intersection(p_z))) * (
                            (-1) ** len(i_set.intersection(p_z))
                        )
                    else:
                        sgn = 0

                    few_particle_H[i, j] += sgn * coeff

    gs_en = min(np.linalg.eigvalsh(few_particle_H))
    print("single particle ground state energy: ", gs_en)
    return gs_en

