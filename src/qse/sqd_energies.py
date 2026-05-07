import numpy as np
import scipy as sp
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.qubit import solve_qubit

from common.measurement_processing import postselect_counts

scipy_kwargs = {"k": 2, "which": "SA"}


def sqd_ground_energy_safe(bitstring_matrix, H_op):
    """
    Safe wrapper around SQD diagonalization:
    - If subspace dimension N is small, fall back to dense eig.
    - Otherwise use solve_qubit (which uses scipy.sparse.linalg.eigsh).
    """
    N = bitstring_matrix.shape[0]

    # If the selected subspace is tiny, ARPACK constraints often break.
    if N <= 2:
        # Build the projected Hamiltonian explicitly as dense via SQD internals
        from qiskit_addon_sqd.qubit import project_operator_to_subspace
        ham_proj = project_operator_to_subspace(bitstring_matrix, H_op)
        ham_dense = ham_proj.toarray() if sp.sparse.issparse(ham_proj) else np.asarray(ham_proj)
        return float(np.min(np.linalg.eigvalsh(ham_dense)))

    # Safe k for eigsh: must satisfy k < N-1
    desired_k = int(scipy_kwargs.get("k", 2))
    safe_k = min(desired_k, N - 2)

    if safe_k < 1:
        # Shouldn't happen for N>=3, but just in case:
        from qiskit_addon_sqd.qubit import project_operator_to_subspace
        ham_proj = project_operator_to_subspace(bitstring_matrix, H_op)
        ham_dense = ham_proj.toarray() if sp.sparse.issparse(ham_proj) else np.asarray(ham_proj)
        return float(np.min(np.linalg.eigvalsh(ham_dense)))

    # Use solve_qubit with a safe k
    scipy_kwargs_safe = dict(scipy_kwargs)
    scipy_kwargs_safe["k"] = safe_k

    eigenvals, _ = solve_qubit(bitstring_matrix, H_op, verbose=False, **scipy_kwargs_safe)
    return float(np.min(eigenvals))


def sqd_energies_from_cumulative(counts_cum, H_op, num_spins, min_unique=1, num_ones=None):
    if num_ones is None: #chain hamiltonian
        num_ones = num_spins // 2
    # print(f"num_ones={num_ones}")
    # print(f"scipy_kwargs: {scipy_kwargs}")

    energies = []
    for step, counts in enumerate(counts_cum, start=1):

        # Filters out bitstrings that do not have specified number (`num_ones`) of `1` bits.
        counts = postselect_counts(counts, num_ones=num_ones)

        # If postselection leaves too few states, we can't form a meaningful subspace.
        if len(counts) < min_unique:
            energies.append(np.nan)
            print(f"[step {step}] postselected unique bitstrings={len(counts)} < {min_unique} -> energy=NaN")
            continue

        bitstring_matrix, probs = counts_to_arrays(counts=counts)
        E0 = sqd_ground_energy_safe(bitstring_matrix, H_op)
        energies.append(E0)
        print(f"[step {step}] subspace N={bitstring_matrix.shape[0]} -> E0={E0:.6f}")

    return energies
