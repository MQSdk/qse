from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit.circuit import Parameter


def initial_circuit(H_op, dt, num_trotter_steps, num_spins) -> QuantumCircuit:
    evol_gate = PauliEvolutionGate(
        H_op,
        time=(dt / num_trotter_steps),
        synthesis=LieTrotter(reps=num_trotter_steps)
    )  # `U` operator

    qr = QuantumRegister(num_spins)
    qc_evol = QuantumCircuit(qr)
    qc_evol.append(evol_gate, qargs=qr)
    return qc_evol


def build_krylov_measurement_circuits(ref_preps, krylov_dim, qc_evol):
    circuits = []
    labels = []  # keep track which (I, rep) each circuit corresponds to
    for I, qc_ref in enumerate(ref_preps):
        for rep in range(krylov_dim):
            circ = qc_ref.copy()

            for _ in range(rep):
                circ.compose(other=qc_evol, inplace=True)

            circ.measure_all()
            circuits.append(circ)
            labels.append((I, rep))
            
    return circuits, labels


def build_krylov_measurement_circuits_one_ref_state(qc_state_prep, krylov_dim, qc_evol):
    circuits = []
    for rep in range(krylov_dim):
        circ = qc_state_prep.copy()

        # Repeating the `U` operator to implement U^0, U^1, U^2, and so on, for power Krylov space
        for _ in range(rep):
            circ.compose(other=qc_evol, inplace=True)

        circ.measure_all()
        circuits.append(circ)

    return circuits


def build_efficient_time_evolution_circuit(n_qubits, num_trotter_steps, t):
    # Create instruction for rotation about XX+YY-ZZ:
    Rxyz_circ = QuantumCircuit(2)
    Rxyz_circ.rxx(2 * t, 0, 1)
    Rxyz_circ.ryy(2 * t, 0, 1)
    Rxyz_circ.rzz(-2 * t, 0, 1)
    Rxyz_instr = Rxyz_circ.to_instruction(label="RXX+YY-ZZ")

    interaction_list = [
        [[i, i + 1] for i in range(0, n_qubits - 1, 2)],
        [[i, i + 1] for i in range(1, n_qubits - 1, 2)],
    ]  # linear chain

    qr = QuantumRegister(n_qubits)
    trotter_step_circ = QuantumCircuit(qr)
    for i, color in enumerate(interaction_list):
        for interaction in color:
            trotter_step_circ.append(Rxyz_instr, interaction)
        if i < len(interaction_list) - 1:
            trotter_step_circ.barrier()

    reverse_trotter_step_circ = trotter_step_circ.reverse_ops()

    qc_evol = QuantumCircuit(qr)
    for step in range(num_trotter_steps):
        if step % 2 == 0:
            qc_evol = qc_evol.compose(trotter_step_circ)
        else:
            qc_evol = qc_evol.compose(reverse_trotter_step_circ)

    return qc_evol


def build_efficient_time_evolution_circuit_multi(n_qubits, num_trotter_steps, t):
    # MR-only: build a gate-friendly qc_evol_mr with the SAME structure/time parameter t

    # Create instruction for rotation about XX+YY-ZZ:
    Rxyz_gate = QuantumCircuit(2)
    Rxyz_gate.rxx(2 * t, 0, 1)
    Rxyz_gate.ryy(2 * t, 0, 1)
    Rxyz_gate.rzz(-2 * t, 0, 1)
    Rxyz_gate = Rxyz_gate.to_gate(label="RXX+YY-ZZ")  # <-- gate (control-able)

    interaction_list = [
        [[i, i + 1] for i in range(0, n_qubits - 1, 2)],
        [[i, i + 1] for i in range(1, n_qubits - 1, 2)],
    ]  # linear chain

    qr_sys = QuantumRegister(n_qubits)
    trotter_step_mr = QuantumCircuit(qr_sys)
    for i, color in enumerate(interaction_list):
        for (a, b) in color:
            trotter_step_mr.append(Rxyz_gate, [a, b])
        if i < len(interaction_list) - 1:
            trotter_step_mr.barrier()

    reverse_step_mr = trotter_step_mr.reverse_ops()

    qc_evol_mr = QuantumCircuit(qr_sys)
    for step in range(num_trotter_steps):
        qc_evol_mr = qc_evol_mr.compose(trotter_step_mr if step % 2 == 0 else reverse_step_mr)

    return qc_evol_mr


###################################################################
# multireference

def strip_barriers(qc: QuantumCircuit) -> QuantumCircuit:
    qc2 = QuantumCircuit(*qc.qregs)
    for inst, qargs, cargs in qc.data:
        if inst.name != "barrier":
            qc2.append(inst, qargs, cargs)
    return qc2


def evol_gate_at_k(k: int, qc_evol_mr_nb, t, dt_circ):
    """
    Returns a Gate implementing U(k*dt) approx.
    NOTE: we bind t = k*dt_circ (same convention as SR template).
    """
    qc_bound = qc_evol_mr_nb.assign_parameters({t: float(k * dt_circ)}, inplace=False)
    return qc_bound.to_gate(label=f"U(k={k})")


def controlled_prepare_ref(qc: QuantumCircuit, anc: int, sys: list, bitstring: str):
    """Controlled preparation of computational-basis |Phi> (only X gates -> CX when controlled)."""
    for q, b in enumerate(bitstring):
        if b == "1":
            qc.cx(anc, sys[q])


def build_mr_circuit(I: int, m: int, J: int, n: int, n_qubits, ref_bitstrings, qc_evol_mr_nb, t, dt_circ) -> QuantumCircuit:
    """
    Paper Sec.2.2 / Fig.2:
      prepare (|0>|psi_{I,m}> + |1>|psi_{J,n}>)/sqrt2
      where |psi_{I,m}> = U_m |Phi_I>, |psi_{J,n}> = U_n |Phi_J>
    then ancilla X+iY gives <psi_{I,m}|psi_{J,n}>.

    Hamiltonian elements are obtained by measuring (P ⊗ X_anc) and (P ⊗ Y_anc)
    on this SAME state (as your observable list already does).
    """
    qr = QuantumRegister(n_qubits + 1)
    qc = QuantumCircuit(qr, name=f"MR_I{I}m{m}_J{J}n{n}")

    anc = 0
    sys = list(range(1, n_qubits + 1))

    qc.h(anc)

    # --- Branch |1>: prepare |Phi_J>, then apply U_n ---
    controlled_prepare_ref(qc, anc, sys, ref_bitstrings[J])
    Un = evol_gate_at_k(n, qc_evol_mr_nb, t, dt_circ)
    qc.append(Un.control(1), [anc] + sys)

    # --- Branch |0>: prepare |Phi_I>, then apply U_m ---
    # Control-on-|0> implemented by X anc / controlled / X anc
    qc.x(anc)
    controlled_prepare_ref(qc, anc, sys, ref_bitstrings[I])
    Um = evol_gate_at_k(m, qc_evol_mr_nb, t, dt_circ)
    qc.append(Um.control(1), [anc] + sys)
    qc.x(anc)

    return qc


def build_multi_ref_circuits(d_refs, s, n_qubits, ref_bitstrings, qc_evol_mr_nb, t, dt_circ):
    circuits_mr = []
    meta_mr = []

    for I in range(d_refs):
        for J in range(d_refs):
            for m in range(s + 1):
                for n in range(s + 1):
                    circuits_mr.append(
                        build_mr_circuit(I, m, J, n, n_qubits, ref_bitstrings, qc_evol_mr_nb, t, dt_circ))
                    meta_mr.append((I, J, m, n))

    return circuits_mr, meta_mr
