import numpy as np

from qiskit.quantum_info import SparsePauliOp, Pauli
from pyscf import ao2mo, gto, mcscf, scf

from qiskit.transpiler import CouplingMap
from qiskit_addon_utils.problem_generators import generate_xyz_hamiltonian


def cholesky(V, eps):
    # see https://arxiv.org/pdf/1711.02242.pdf section B2
    # see https://arxiv.org/abs/1808.02625
    # see https://arxiv.org/abs/2104.08957
    no = V.shape[0]
    chmax, ng = 20 * no, 0
    W = V.reshape(no**2, no**2)
    L = np.zeros((no**2, chmax))
    Dmax = np.diagonal(W).copy()
    nu_max = np.argmax(Dmax)
    vmax = Dmax[nu_max]
    while vmax > eps:
        L[:, ng] = W[:, nu_max]
        if ng > 0:
            L[:, ng] -= np.dot(L[:, 0:ng], (L.T)[0:ng, nu_max])
        L[:, ng] /= np.sqrt(vmax)
        Dmax[: no**2] -= L[: no**2, ng] ** 2
        ng += 1
        nu_max = np.argmax(Dmax)
        vmax = Dmax[nu_max]
    L = L[:, :ng].reshape((no, no, ng))
    print(
        "accuracy of Cholesky decomposition ",
        np.abs(np.einsum("prg,qsg->prqs", L, L) - V).max(),
    )
    return L, ng


def identity(n):
    return SparsePauliOp.from_list([("I" * n, 1)])


def creators_destructors(n, mapping="jordan_wigner"):
    c_list = []
    if mapping == "jordan_wigner":
        for p in range(n):
            if p == 0:
                ell, r = "I" * (n - 1), ""
            elif p == n - 1:
                ell, r = "", "Z" * (n - 1)
            else:
                ell, r = "I" * (n - p - 1), "Z" * p
            cp = SparsePauliOp.from_list([(ell + "X" + r, 0.5), (ell + "Y" + r, -0.5j)])
            c_list.append(cp)
    else:
        raise ValueError("Unsupported mapping.")
    d_list = [cp.adjoint() for cp in c_list]
    return c_list, d_list


def build_hamiltonian(ecore: float, h1e: np.ndarray, h2e: np.ndarray) -> SparsePauliOp:
    ncas, _ = h1e.shape

    C, D = creators_destructors(2 * ncas, mapping="jordan_wigner")
    Exc = []
    for p in range(ncas):
        Excp = [C[p] @ D[p] + C[ncas + p] @ D[ncas + p]]
        for r in range(p + 1, ncas):
            Excp.append(
                C[p] @ D[r]
                + C[ncas + p] @ D[ncas + r]
                + C[r] @ D[p]
                + C[ncas + r] @ D[ncas + p]
            )
        Exc.append(Excp)

    # low-rank decomposition of the Hamiltonian
    Lop, ng = cholesky(h2e, 1e-6)
    t1e = h1e - 0.5 * np.einsum("pxxr->pr", h2e)

    H = ecore * identity(2 * ncas)
    # one-body term
    for p in range(ncas):
        for r in range(p, ncas):
            H += t1e[p, r] * Exc[p][r - p]
    # two-body term
    for g in range(ng):
        Lg = 0 * identity(2 * ncas)
        for p in range(ncas):
            for r in range(p, ncas):
                Lg += Lop[p, r, g] * Exc[p][r - p]
        H += 0.5 * Lg @ Lg

    return H.chop().simplify()


####################################################################
# Hamiltonians for molecules: H2 and LiH

def build_H2_hamiltonian():
    # 1. Define the molecule
    distance = 0.735
    a = distance / 2
    mol = gto.Mole()
    mol.build(
        verbose=0,
        atom=[
            ["H", (0, 0, -a)],
            ["H", (0, 0, a)],
        ],
        basis="sto-6g",
        spin=0,
        charge=0,
        symmetry="Dooh",
    )

    mf = scf.RHF(mol)
    mf.scf()

    print(
        mf.energy_nuc(),
        mf.energy_elec()[0],
        mf.energy_tot(),
        mf.energy_tot() - mol.energy_nuc(),
    )

    active_space = range(mol.nelectron // 2 - 1, mol.nelectron // 2 + 1)

    # 2. Generate fermionic Hamiltonian
    E1 = mf.kernel()
    mx = mcscf.CASCI(mf, ncas=2, nelecas=(1, 1))
    mo = mx.sort_mo(active_space, base=0)
    E2 = mx.kernel(mo)[:2]

    h1e, ecore = mx.get_h1eff()
    h2e = ao2mo.restore(1, mx.get_h2eff(), mx.ncas)

    H_H2 = build_hamiltonian(ecore, h1e, h2e)
    return H_H2


def build_LiH_hamiltonian():
    # 1. Define the molecule
    distance = 1.56
    mol = gto.Mole()
    mol.build(
        verbose=0,
        atom=[["Li", (0, 0, 0)], ["H", (0, 0, distance)]],
        basis="sto-6g",
        spin=0,
        charge=0,
        symmetry="Coov",
    )
    mf = scf.RHF(mol)

    # 2. Generate fermionic Hamiltonian
    E1 = mf.kernel()

    mx = mcscf.CASCI(mf, ncas=5, nelecas=(1, 1))
    cas_space_symmetry = {"A1": 3, "E1x": 1, "E1y": 1}
    mo = mcscf.sort_mo_by_irrep(mx, mf.mo_coeff, cas_space_symmetry)
    E2 = mx.kernel(mo)[:2]
    h1e, ecore = mx.get_h1eff()
    h2e = ao2mo.restore(1, mx.get_h2eff(), mx.ncas)

    H_LiH = build_hamiltonian(ecore, h1e, h2e)
    return H_LiH


def build_N2_strongly_correlated_hamiltonian(distance=2.0, basis="sto-3g", ncas=6, nelecas=(3,3)):
    """
    Strongly correlated model: stretched N2 with a modest active space.
    NOTE: This is heavier than H2. Adjust ncas/nelecas to match your budget.
    """
    a = distance / 2

    mol = gto.Mole()
    mol.build(
        verbose=0,
        atom=[["N", (0, 0, -a)], ["N", (0, 0, a)]],
        basis=basis,
        spin=0,
        charge=0,
        symmetry="Dooh",
    )

    mf = scf.RHF(mol)
    mf.kernel()

    mx = mcscf.CASCI(mf, ncas=ncas, nelecas=nelecas)

    # crude active-space pick around HOMO/LUMO region
    nocc = mol.nelectron // 2
    start = max(0, nocc - ncas//2)
    active_space = range(start, start + ncas)
    mo = mx.sort_mo(active_space, base=0)

    mx.kernel(mo)

    h1e, ecore = mx.get_h1eff()
    h2e = ao2mo.restore(1, mx.get_h2eff(), mx.ncas)

    H = build_hamiltonian(ecore, h1e, h2e)
    return H


####################################################################
# Hamiltonian for the antiferromagnetic XX-Z spin-1/2 chain
# originally used for:
# Sample-based Krylov Quantum Diagonalization (SKQD)

def build_antiferromagnetic_XX_Z_spin_1_2_chain_hamiltonian(num_spins: int):
    coupling_map = CouplingMap.from_ring(num_spins)
    H_op = generate_xyz_hamiltonian(coupling_map, coupling_constants=(0.3, 0.3, 1.0))

    return H_op


####################################################################


def build_heisenberg_chain_hamiltonian(n_qubits: int):
    # coupling strength for XX, YY, and ZZ interactions
    JX = 1
    JY = 3
    JZ = 2

    # Define the Hamiltonian:
    H_int = [["I"] * n_qubits for _ in range(3 * (n_qubits - 1))]
    for i in range(n_qubits - 1):
        H_int[i][i] = "Z"
        H_int[i][i + 1] = "Z"
    for i in range(n_qubits - 1):
        H_int[n_qubits - 1 + i][i] = "X"
        H_int[n_qubits - 1 + i][i + 1] = "X"
    for i in range(n_qubits - 1):
        H_int[2 * (n_qubits - 1) + i][i] = "Y"
        H_int[2 * (n_qubits - 1) + i][i + 1] = "Y"
    H_int = ["".join(term) for term in H_int]
    H_tot = [
        (term, JZ)
        if term.count("Z") == 2
        else (term, JY)
        if term.count("Y") == 2
        else (term, JX)
        for term in H_int
    ]

    # Get operator
    H_op = SparsePauliOp.from_list(H_tot)

    return H_op


def get_heisenberg_hamiltonian_restricted_to_single_particle_states(n_qubits, H_op):
    single_particle_H = np.zeros((n_qubits, n_qubits))
    for i in range(n_qubits):
        for j in range(i + 1):
            for p, coeff in H_op.to_list():
                p_x = Pauli(p).x
                p_z = Pauli(p).z
                if all(p_x[k] == ((i == k) + (j == k)) % 2 for k in range(n_qubits)):
                    sgn = ((-1j) ** sum(p_z[k] and p_x[k] for k in range(n_qubits))) * (
                            (-1) ** p_z[i]
                    )
                else:
                    sgn = 0
                single_particle_H[i, j] += sgn * coeff
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            single_particle_H[i, j] = np.conj(single_particle_H[j, i])

    return single_particle_H
