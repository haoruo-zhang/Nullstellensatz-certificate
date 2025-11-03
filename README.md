# Nullstellensatz Certificate Computation

This repository provides an experimental framework for constructing and verifying **Nullstellensatz certificates** for polynomial systems arising from **birank (2,1) critical point conditions** in low-rank moment optimization.  
The implementation uses **SymPy**, **NumPy**, and optional **GPU acceleration** via **CuPy**.

The goal is to determine whether a polynomial system
\[
p_1 = p_2 = \cdots = p_k = 0
\]
admits a **Nullstellensatz certificate** of the form
\[
\sum_{i=1}^{k} q_i \cdot p_i \equiv 1,
\]
where \( q_i \) are polynomials with bounded degree.  
If such a certificate exists, the system has **no real solution**.  
If not, the degree bound may need to be increased.

---

## Project Structure

├── .vscode/
│ └── settings.json
├── examples/
│ ├── birank(2,1)_example.py # Example using the birank(2,1) system
│ ├── solvable_example.py # System with a real solution
│ └── unsolvable_example.py # System with a Nullstellensatz certificate
├── unit_test/
│ ├── failed_escape_samples_10_14.pkl
│ └── unit_test.py # Unit tests for validation
├── utils/
│ └── utils.py # Core functions (matrix build, equations, GPU solver)
└── README.md
