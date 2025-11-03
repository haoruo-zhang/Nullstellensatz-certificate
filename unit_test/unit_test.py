import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.utils import (
    build_A_numpy,
    sympy_poly_to_exp_coeff_list,
    f_from_B, build_birank21_equations
)
import sympy as sp
import numpy as np
import pickle

def test_build_A_numpy():
    """
    This is the test for the function test_build_A_numpy
    Use the Example 2.3 from Tim's note page 22
    One can verify the A in the order of [1,y,y^2,x,xy,x^2]
    P.S. The calculation on the note is wrong
    """
    # Variables
    x, y = sp.symbols('x y')
    
    # Polynomials from Example 2.3
    p1 = x + y
    p2 = 2*x + 3*y - 1
    p3 = 3*x - y - 2
    p_list_sym = [p1, p2, p3]
    vars_sym = (x,y)

    p_list_terms = [sympy_poly_to_exp_coeff_list(p, vars_sym) for p in p_list_sym]

    # {x,y,1}
    monomial_exps = [(1,0), (0,1), (0,0)]
    # Call your function
    A, b, row_exp,_ = build_A_numpy(p_list_terms, monomial_exps, force_constant_row=True)
    A = A.toarray().astype(float)
    # # Expected monomial ordering: [1,y,y^2,x,xy,x^2]

    # # Example 2.3 gives the coefficient matrix:
    A_expected = np.array([
        [0, 0, 0, 0, 0, -1, 0, 0, -2],   # coeff constant
        [0, 0, 1, 0, -1, 3, 0, -2, -1],  # coeff of y
        [0, 1, 0, 0, 3, 0, 0, -1, 0], # coeff of y^2
        [0, 0, 1, -1, 0, 2, -2, 0, 3],   # coeff of x
        [1, 1, 0, 3, 2, 0, -1, 3, 0],  # coeff of xy
        [1, 0, 0, 2, 0, 0, 3, 0, 0], # coeff of x^2
    ], dtype=float)

    # b expected = [1,0,0,0,0,0]^T
    b_expected = np.array([1, 0, 0, 0, 0, 0], dtype=float)

    # Compare shapes
    assert A.shape == A_expected.shape
    assert b.shape == b_expected.shape
    
    # Numerical compare
    np.testing.assert_allclose(A, A_expected)
    np.testing.assert_allclose(b, b_expected)

    print("✅ test_build_A_numpy passed")


def test_f_from_B():
    """
    Test for f_from_B
    """
    print("=== Testing f_from_B ===")
    B = np.array([
        [1, 2],   # 1 + 2y
        [3, 4],   # 3x + 4xy
    ], dtype=int)

    x, y = sp.symbols('x y')
    f = f_from_B(B, x, y, dX=1, dY=1)

    f_expected = 1 + 2*y + 3*x + 4*x*y

    if sp.expand(f - f_expected) == 0:
        print("✅ f_from_B PASSED")
    else:
        print("❌ f_from_B FAILED")
        print("Computed:", f)
        print("Expected:", f_expected)

def test_build_birank21_equations():
    """
    Test for build_birank21_equations
    """
    print("\n=== Testing build_birank21_equations ===")

    B = np.random.randn(5, 5)

    p_list, vars5 = build_birank21_equations(B)

    if len(p_list) == 6:
        print("✅ p_list length correct (6)")
    else:
        print("❌ p_list length incorrect:", len(p_list))

    if len(vars5) == 5:
        print("✅ vars length correct (5)")
    else:
        print("❌ vars length incorrect:", len(vars5))

    allowed = set(vars5)
    for p in p_list:
        extra = set(p.free_symbols) - allowed
        if extra:
            print("❌ Unexpected variables in equation:", extra)
            return

    print("✅ No unexpected variables in polynomials")

   
    x1, x2, y1, lam, R = vars5
    p2 = p_list[0]
    p3 = p_list[1]

    if sp.simplify(p2.subs({x2:x1}) - p3.subs({x2:x1})) == 0:
        print("✅ Consistency check passed (p2=p3 when x1=x2)")
    else:
        print("❌ Consistency check failed (p2 != p3 when x1=x2)")



def test_failed_sample(pkl_path="failed_escape_samples_10_14.pkl", index=0, tol=1e-6, verbose=True):
    """
    Test sample[index] from failed_escape_samples pkl file against the birank(2,1) equations.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(base_dir, "failed_escape_samples_10_14.pkl")
    # Load sample
    with open(pkl_path, "rb") as f:
        failed_escape_samples = pickle.load(f)

    sample = failed_escape_samples[index]

    # Extract values
    x1_val = float(sample['x1'])
    x2_val = float(sample['x2'])
    y1_val = float(sample['y1'])
    lam_val = float(sample['lam'])
    R_val  = float(sample['R'])
    B_val  = sample['B']

    # Build equations
    p_list, (x1, x2, y1, lam, R) = build_birank21_equations(B_val)

    # Substitute numeric values
    subs_dict = {x1: x1_val, x2: x2_val, y1: y1_val, lam: lam_val, R: R_val}
    residuals = [float(p.subs(subs_dict)) for p in p_list]

    if verbose:
        print("\n=== Testing failed_escape_samples[{}] ===".format(index))
        for i, r in enumerate(residuals, start=1):
            print(f"p{i}(sample) = {r:.6e}")
        print(f"Max residual = {max(abs(r) for r in residuals):.6e}\n")

    passed = all(abs(r) < tol for r in residuals)

    if passed:
        print("✅ This sample satisfies the birank(2,1) critical point equations within tolerance.")
    else:
        print("❌ This sample does NOT satisfy the birank(2,1) equations (critical-point conditions fail).")

    return residuals
if __name__ == "__main__":
    test_build_A_numpy()
    test_f_from_B()
    test_build_birank21_equations()
    test_failed_sample()