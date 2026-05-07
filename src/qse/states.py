from qiskit import QuantumCircuit


def neel_state(num_spins: int, shift: int = 0) -> QuantumCircuit:
    """
    Product-state reference (a single determinant in JW sense).
    shift=0 -> 1010...
    shift=1 -> 0101...
    """
    qc = QuantumCircuit(num_spins)
    for i in range(num_spins):
        if (i + shift) % 2 == 0:
            qc.x(i)
    return qc


def get_krylov_initial_state(n_qubits) -> QuantumCircuit:
    control = 0
    excitation = int(n_qubits / 2) + 1
    controlled_state_prep = QuantumCircuit(n_qubits + 1)
    controlled_state_prep.cx(control, excitation)

    return controlled_state_prep


def get_ref_bitstring(n, n_qubits):
    excitation_sys = int(n_qubits / 2)

    if n < 0 or n >= n_qubits:
        raise ValueError("n must be between 0 and n_qubits")

    pos = (excitation_sys - n) % n_qubits
    return "0" * pos + "1" + "0" * (n_qubits - pos - 1)


def get_ref_bitstring_chem(ncas, occ_alpha, occ_beta):
    """
    Returns a computational-basis bitstring for a Slater determinant.

    Bit order assumed: alpha block then beta block
    bitstring[0:ncas]   -> alpha occupations
    bitstring[ncas:2ncas] -> beta occupations
    """
    n_qubits = 2 * ncas
    bits = ["0"] * n_qubits
    for p in occ_alpha:
        bits[p] = "1"
    for p in occ_beta:
        bits[ncas + p] = "1"
    return "".join(bits)
