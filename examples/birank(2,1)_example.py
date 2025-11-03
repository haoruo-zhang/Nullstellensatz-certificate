import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.utils import (
    nullstellensatz_certificate_fast,
    build_birank21_equations
)
import numpy as np

def main():
    B = np.random.randn(5,5)
    B = B / np.linalg.norm(B)   

    # total degree of q
    deg = 20
    p_list_sym, vars_sym = build_birank21_equations(B, dX=4, dY=4)
    r = nullstellensatz_certificate_fast(
        p_list_sym,
        vars_sym,
        q_degree=deg,
        basis="total",
        force_constant_row=True
    )

if __name__ == "__main__":
    main()