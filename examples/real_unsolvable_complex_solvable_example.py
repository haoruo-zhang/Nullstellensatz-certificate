import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.utils import nullstellensatz_certificate_fast
import sympy as sp

def main():
    x, y = sp.symbols('x y')
    p1 = x**2 + y**2 - 1
    p2 = x-y
    p_list = [p1, p2]
    vars = (x, y)

    print("\n===== Case 1: No solution system =====")
    for q in range(0, 5):
        print(f"\nq = {q}")
        r = nullstellensatz_certificate_fast(
            p_list,
            vars,
            q_degree=q,
            basis="total",
            force_constant_row=True
        )

if __name__ == "__main__":
    main()