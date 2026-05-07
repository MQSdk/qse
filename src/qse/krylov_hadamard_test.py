from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import StabilizerState, Pauli


def get_parameters_for_template_circuit(krylov_dim, dt_circ):
    parameters = []
    for idx in range(1, krylov_dim):
        parameters.append(dt_circ * (idx))

    return parameters


def build_modified_hadamard_test_circuit(n_qubits, controlled_state_prep, qc_evol):
    qr = QuantumRegister(n_qubits + 1)
    qc = QuantumCircuit(qr)
    qc.h(0)
    qc.compose(controlled_state_prep, list(range(n_qubits + 1)), inplace=True)
    qc.barrier()
    qc.compose(qc_evol, list(range(1, n_qubits + 1)), inplace=True)
    qc.barrier()
    qc.x(0)
    qc.compose(controlled_state_prep.inverse(), list(range(n_qubits + 1)), inplace=True)
    qc.x(0)

    return qc


def get_observables_S(n_qubits, qc_trans):
    observable_S_real = "I" * (n_qubits) + "X"
    observable_S_imag = "I" * (n_qubits) + "Y"

    observable_op_real = SparsePauliOp(observable_S_real)
    observable_op_imag = SparsePauliOp(observable_S_imag)

    layout = qc_trans.layout
    observable_op_real = observable_op_real.apply_layout(layout)
    observable_op_imag = observable_op_imag.apply_layout(layout)

    observable_S_real = (observable_op_real.paulis.to_labels())
    observable_S_imag = observable_op_imag.paulis.to_labels()

    observables_S = [[observable_S_real], [observable_S_imag]]

    return observables_S


def get_observables_H(H_op, qc_trans):
    # Hamiltonian terms to measure
    observable_list = []
    for pauli, coeff in zip(H_op.paulis, H_op.coeffs):
        # print(pauli)
        observable_H_real = pauli[::-1].to_label() + "X"
        observable_H_imag = pauli[::-1].to_label() + "Y"
        observable_list.append([observable_H_real])
        observable_list.append([observable_H_imag])

    layout = qc_trans.layout

    observable_trans_list = []
    for observable in observable_list:
        observable_op = SparsePauliOp(observable)
        observable_op = observable_op.apply_layout(layout)
        observable_trans_list.append([observable_op.paulis.to_labels()])

    return observable_trans_list


def get_expectation_values_S(n_qubits, qc_cliff):
    # Get expectation values from experiment
    S_expval_real = StabilizerState(qc_cliff).expectation_value(
        Pauli("I" * (n_qubits) + "X")
    )
    S_expval_imag = StabilizerState(qc_cliff).expectation_value(
        Pauli("I" * (n_qubits) + "Y")
    )

    # Get expectation values
    S_expval = S_expval_real + 1j * S_expval_imag
    return S_expval


def get_expectation_values_H(H_op, qc_cliff):
    H_expval = 0
    for obs_idx, (pauli, coeff) in enumerate(zip(H_op.paulis, H_op.coeffs)):
        # Get expectation values from experiment
        expval_real = StabilizerState(qc_cliff).expectation_value(
            Pauli(pauli[::-1].to_label() + "X")
        )
        expval_imag = StabilizerState(qc_cliff).expectation_value(
            Pauli(pauli[::-1].to_label() + "Y")
        )
        expval = expval_real + 1j * expval_imag

        # Fill-in matrix elements
        H_expval += coeff * expval

    return H_expval