import numpy as np
from itertools import product
from scipy.sparse import lil_matrix
import sympy as sp
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from cupyx.scipy.sparse.linalg import lsmr


# ------------------ 1) Convert SymPy polynomial to term list ------------------
def sympy_poly_to_exp_coeff_list(p, vars_order):
    """
    Convert a SymPy polynomial into a list of monomial exponent tuples and numeric coefficients.

    Parameters
    ----------
    p : sympy.Expr
        The polynomial expression.
    vars_order : tuple
        Variables used to define the monomial order.

    Returns
    -------
    terms : list of ((e1,e2,...), coeff)
        Exponent tuples and coefficients. Coefficients are converted to int if possible,
        otherwise to float.
    """
    P = sp.Poly(p, *vars_order)
    terms = []
    for exp, coeff in zip(P.monoms(), P.coeffs()):
        if coeff.is_Number:
            coeff_out = int(coeff) if coeff.is_integer else float(coeff)
        else:
            coeff_out = float(coeff)
        terms.append((tuple(int(e) for e in exp), coeff_out))
    return terms


# ------------------ 2) Construct monomial exponent sets -----------------------
def monomials_total_degree_exps(n_vars, degree):
    """
    Generate all monomials of total degree <= degree.

    Example (n_vars=2, degree=2):
      -> (0,0), (1,0), (0,1), (2,0), (1,1), (0,2)
    """
    exps = []
    for e in product(range(degree+1), repeat=n_vars):
        if sum(e) <= degree:
            exps.append(tuple(e))
    return exps


def monomials_tensor_product_exps(n_vars, degree):
    """
    Generate full tensor product monomials:
    Each variable exponent ranges independently from 0..degree.
    """
    return [tuple(e) for e in product(range(degree+1), repeat=n_vars)]


# ------------------ 3) Build the linear system A * x = b ----------------------
def build_A_numpy(p_list_terms, monomial_exps, force_constant_row=True):
    """
    Construct matrix A and vector b for the system:
       sum_k q_k * p_k = 1

    The matrix is constructed by expanding (p_k * each monomial in q_k).

    Parameters
    ----------
    p_list_terms : list
        Each entry is a list of (exp_tuple, coeff) for a polynomial p_k.
    monomial_exps : list
        List of exponent tuples defining the basis for q_k.
    force_constant_row : bool
        Ensures that the constant monomial (0,...,0) appears as one row,
        corresponding to enforcing the equality to 1.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Sparse matrix of the linear system.
    b : ndarray
        Right-hand side vector.
    row_exps : list
        Monomials indexing each row.
    var_cols : list of (k, j)
        Mapping from column index → which polynomial index k and monomial index j in q_k.
    """
    K = len(p_list_terms)
    J = len(monomial_exps)
    n_vars = len(monomial_exps[0])

    # Determine all monomials that can appear in expanded products
    exp_set = set()
    for terms in p_list_terms:
        for (exp_p, _) in terms:
            exp_p_arr = np.array(exp_p, dtype=np.int64)
            for exp_m in monomial_exps:
                exp_set.add(tuple((exp_p_arr + np.array(exp_m, dtype=np.int64)).tolist()))

    zero = tuple(0 for _ in range(n_vars))
    if force_constant_row:
        exp_set.add(zero)

    row_exps = sorted(exp_set)
    exp_to_row = {exp: i for i, exp in enumerate(row_exps)}

    n_rows = len(row_exps)
    n_cols = K * J

    A = lil_matrix((n_rows, n_cols), dtype=np.float64)
    b = np.zeros(n_rows, dtype=np.float64)

    # Fill A by matching monomial powers
    for k, terms in enumerate(p_list_terms):
        for j, exp_m in enumerate(monomial_exps):
            col = k * J + j
            exp_m_arr = np.array(exp_m, dtype=np.int64)
            for (exp_p, coeff_p) in terms:
                row = exp_to_row[tuple((np.array(exp_p, dtype=np.int64) + exp_m_arr).tolist())]
                A[row, col] += float(coeff_p)

    # Enforce constant term = 1
    if zero in exp_to_row:
        b[exp_to_row[zero]] = 1.0

    var_cols = [(k, j) for k in range(K) for j in range(J)]
    return A.tocsr(), b, row_exps, var_cols


# ------------------ 4) Construct f and derivatives from coefficient matrix B --
def f_from_B(B, x, y, dX=4, dY=4):
    """
    Construct a bivariate polynomial f(x,y) = sum_{i,j} B[i,j] * x^i * y^j.
    """
    if B.ndim == 1:
        B = B.reshape(dX+1, dY+1)
    f = sp.Integer(0)
    for i in range(dX+1):
        for j in range(dY+1):
            f += sp.Rational(1,1) * B[i,j] * (x**i) * (y**j)
    return sp.expand(f)


def build_birank21_equations(B, dX=4, dY=4):
    """
    Construct the 5 polynomial equations for the birank(2,1) critical point system.

    Returns
    -------
    p_list : list of sympy.Expr
        The 5 constraint polynomials.
    vars5 : tuple
        Variables (x1, x2, y1, lam, R).
    """
    x1, x2, y1, lam, R = sp.symbols('x1 x2 y1 lam R')
    x, y = sp.symbols('x y')

    f = f_from_B(B, x, y, dX, dY)
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    f_x1y1 = sp.expand(f.subs({x:x1, y:y1}))
    f_x2y1 = sp.expand(f.subs({x:x2, y:y1}))
    fx_x1y1 = sp.expand(fx.subs({x:x1, y:y1}))
    fx_x2y1 = sp.expand(fx.subs({x:x2, y:y1}))
    fy_x1y1 = sp.expand(fy.subs({x:x1, y:y1}))
    fy_x2y1 = sp.expand(fy.subs({x:x2, y:y1}))

    p1 = sp.expand(lam*f_x1y1 + (1 - lam)*f_x2y1-R)
    p2 = sp.expand(f_x1y1 - R)
    p3 = sp.expand(f_x2y1 - R)
    p4 = sp.expand(lam*fy_x1y1 + (1 - lam)*fy_x2y1)
    p5 = sp.expand(fx_x1y1)
    p6 = sp.expand(fx_x2y1)

    vars5 = (x1, x2, y1, lam, R)
    return [p1,p2, p3, p4, p5, p6], vars5


# ------------------ 5) Solve A x ≈ b using GPU LSMR ---------------------------
def nullstellensatz_certificate_fast(
    p_list_sym, vars_sym, q_degree, basis="total", force_constant_row=True
):
    """
    Construct and solve the linear system A x ≈ b using GPU-based LSMR.
    This does NOT enforce exact algebraic solvability; instead it attempts
    to minimize the residual norm ||A x - b|| in floating point arithmetic.

    A small residual indicates that the chosen q-degree is sufficient
    to approximate a Nullstellensatz certificate. A large residual indicates
    that the q-degree is likely too small.
    """
    # Convert polynomials to exponent form
    p_list_terms = [sympy_poly_to_exp_coeff_list(p, vars_sym) for p in p_list_sym]

    # Choose basis for q polynomials
    n_vars = len(vars_sym)
    if basis == "total":
        monomial_exps = monomials_total_degree_exps(n_vars, q_degree)
    elif basis == "tensor":
        monomial_exps = monomials_tensor_product_exps(n_vars, q_degree)
    else:
        raise ValueError("basis must be 'total' or 'tensor'")

    # Build linear system
    A, b, _, _ = build_A_numpy(p_list_terms, monomial_exps, force_constant_row=force_constant_row)

    # Move system to GPU
    A_gpu = cpx_sparse.csr_matrix(A)
    b_gpu = cp.asarray(b)

    # Solve using LSMR (least-squares iterative solver) on GPU
    result = lsmr(A_gpu, b_gpu, atol=1e-12, btol=1e-12, maxiter=5000)

    x_gpu = result[0]   # Solution vector on GPU
    istop = result[1]   # Convergence/termination flag
    residual = result[3]  # Final residual norm ||A x - b||

    print("istop =", istop)
    print("residual =", float(residual))
    return float(residual)
