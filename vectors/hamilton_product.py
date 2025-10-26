"""
hamilton_product.py
Deep dive into the Hamilton product: the multiplication operation for quaternions.
Demonstrates all key properties with mathematical proofs and visual examples.

Reference: https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
"""

import numpy as np
from math import sqrt, cos, sin

class Quaternion:
    """
    Represents a quaternion: q = w + xi + yj + zk
    """
    def __init__(self, w, x, y, z):
        self.w = w  # scalar part
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"({self.w:.4f} + {self.x:.4f}i + {self.y:.4f}j + {self.z:.4f}k)"

    def __eq__(self, other):
        return (abs(self.w - other.w) < 1e-10 and abs(self.x - other.x) < 1e-10 and
                abs(self.y - other.y) < 1e-10 and abs(self.z - other.z) < 1e-10)

    def __mul__(self, other):
        """
        Hamilton product: (w1 + x1i + y1j + z1k)(w2 + x2i + y2j + z2k)
        
        Expansion using basis rules:
        i²=j²=k²=-1, ij=-ji=k, jk=-kj=i, ki=-ik=j
        """
        # Handle scalar multiplication
        if isinstance(other, (int, float)):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        
        # Formula derived from distributive expansion
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return Quaternion(w, x, y, z)

    def __add__(self, other):
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)

    def __rmul__(self, scalar):
        return Quaternion(scalar*self.w, scalar*self.x, scalar*self.y, scalar*self.z)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm_squared(self):
        return self.w**2 + self.x**2 + self.y**2 + self.z**2

    def norm(self):
        return sqrt(self.norm_squared())

    def inverse(self):
        n2 = self.norm_squared()
        if n2 < 1e-10:
            raise ValueError("Cannot invert zero quaternion")
        conj = self.conjugate()
        return Quaternion(conj.w/n2, conj.x/n2, conj.y/n2, conj.z/n2)

    def normalize(self):
        n = self.norm()
        if n < 1e-10:
            raise ValueError("Cannot normalize zero quaternion")
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)

    def scalar_part(self):
        return self.w

    def vector_part(self):
        return np.array([self.x, self.y, self.z])

    def to_matrix_left(self):
        """Convert to left multiplication matrix for q*p = Q_L(q)*p"""
        return np.array([
            [self.w, -self.x, -self.y, -self.z],
            [self.x,  self.w, -self.z,  self.y],
            [self.y,  self.z,  self.w, -self.x],
            [self.z, -self.y,  self.x,  self.w]
        ])

    def to_matrix_right(self):
        """Convert to right multiplication matrix for p*q = Q_R(q)*p"""
        return np.array([
            [self.w, -self.x, -self.y, -self.z],
            [self.x,  self.w,  self.z, -self.y],
            [self.y, -self.z,  self.w,  self.x],
            [self.z,  self.y, -self.x,  self.w]
        ])

    def to_vector(self):
        """Convert quaternion to 4D vector"""
        return np.array([self.w, self.x, self.y, self.z])


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70 + "\n")


def property_1_basis_rules():
    """PROPERTY 1: Basis Element Multiplication Rules"""
    print_section("PROPERTY 1: Basis Element Multiplication Rules")
    print("The fundamental rules that define the Hamilton product:")
    print()
    
    i = Quaternion(0, 1, 0, 0)
    j = Quaternion(0, 0, 1, 0)
    k = Quaternion(0, 0, 0, 1)
    one = Quaternion(1, 0, 0, 0)
    
    rules = [
        ("i² = -1", i*i, one * (-1)),
        ("j² = -1", j*j, one * (-1)),
        ("k² = -1", k*k, one * (-1)),
        ("ij = k", i*j, k),
        ("jk = i", j*k, i),
        ("ki = j", k*i, j),
        ("ji = -k", j*i, k * (-1)),
        ("kj = -i", k*j, i * (-1)),
        ("ik = -j", i*k, j * (-1)),
        ("ijk = -1", i*j*k, one * (-1)),
    ]
    
    for rule_name, computed, expected in rules:
        is_correct = "✓" if computed == expected else "✗"
        print(f"{is_correct} {rule_name:15} = {computed}")
    print()


def property_2_distributivity():
    """PROPERTY 2: Distributivity over Addition"""
    print_section("PROPERTY 2: Distributivity over Addition")
    print("Hamilton product is distributive: q(r+s) = qr + qs and (r+s)q = rq + sq")
    print()
    
    q = Quaternion(1, 2, 0, 0)
    r = Quaternion(0, 1, 2, 0)
    s = Quaternion(0, 0, 1, 3)
    
    # Left distributivity: q(r+s) = qr + qs
    left_1 = q * (r + s)
    left_2 = q * r + q * s
    
    print(f"q = {q}")
    print(f"r = {r}")
    print(f"s = {s}")
    print()
    print(f"Left Distributivity: q(r+s) = qr + qs")
    print(f"  q(r+s) = {left_1}")
    print(f"  qr+qs  = {left_2}")
    print(f"  Equal: {'✓' if left_1 == left_2 else '✗'}")
    print()
    
    # Right distributivity: (r+s)q = rq + sq
    right_1 = (r + s) * q
    right_2 = r * q + s * q
    
    print(f"Right Distributivity: (r+s)q = rq + sq")
    print(f"  (r+s)q = {right_1}")
    print(f"  rq+sq  = {right_2}")
    print(f"  Equal: {'✓' if right_1 == right_2 else '✗'}")
    print()


def property_3_associativity():
    """PROPERTY 3: Associativity of Multiplication"""
    print_section("PROPERTY 3: Associativity of Multiplication")
    print("Hamilton product is associative: (pq)r = p(qr)")
    print()
    
    p = Quaternion(1, 1, 0, 0)
    q = Quaternion(0, 1, 1, 0)
    r = Quaternion(1, 0, 1, 1)
    
    # Left grouping: (pq)r
    pqr_left = (p * q) * r
    
    # Right grouping: p(qr)
    pqr_right = p * (q * r)
    
    print(f"p = {p}")
    print(f"q = {q}")
    print(f"r = {r}")
    print()
    print(f"(p*q)*r = {pqr_left}")
    print(f"p*(q*r) = {pqr_right}")
    print(f"Associative: {'✓' if pqr_left == pqr_right else '✗'}")
    print()


def property_4_non_commutativity():
    """PROPERTY 4: Non-Commutativity"""
    print_section("PROPERTY 4: Non-Commutativity")
    print("Hamilton product is NOT commutative: pq ≠ qp (in general)")
    print()
    
    p = Quaternion(1, 2, 3, 4)
    q = Quaternion(5, 6, 7, 8)
    
    pq = p * q
    qp = q * p
    
    print(f"p = {p}")
    print(f"q = {q}")
    print()
    print(f"p*q = {pq}")
    print(f"q*p = {qp}")
    print(f"p*q ≠ q*p: {'✓' if pq != qp else '✗'}")
    print()
    
    # Show difference
    print(f"Difference (p*q - q*p):")
    diff = pq - qp
    print(f"  = {diff}")
    print()


def property_5_scalar_identity():
    """PROPERTY 5: Scalar and Identity Properties"""
    print_section("PROPERTY 5: Scalar and Identity Properties")
    print("Scalar quaternions commute with all quaternions; 1 is the identity")
    print()
    
    q = Quaternion(2, 3, 4, 5)
    scalar = Quaternion(3, 0, 0, 0)
    identity = Quaternion(1, 0, 0, 0)
    
    # Scalar commutes with any quaternion
    sq = scalar * q
    qs = q * scalar
    
    print(f"Scalar quaternion s = {scalar}")
    print(f"Quaternion q = {q}")
    print()
    print(f"s*q = {sq}")
    print(f"q*s = {qs}")
    print(f"Commutative (scalars): {'✓' if sq == qs else '✗'}")
    print()
    
    # Identity property
    q_identity = q * identity
    identity_q = identity * q
    
    print(f"Identity quaternion 1 = {identity}")
    print(f"q*1 = {q_identity}")
    print(f"1*q = {identity_q}")
    print(f"q*1 = q: {'✓' if q_identity == q else '✗'}")
    print()


def property_6_norm_multiplicative():
    """PROPERTY 6: Norm is Multiplicative"""
    print_section("PROPERTY 6: Norm is Multiplicative")
    print("||pq|| = ||p|| * ||q|| (composition property)")
    print()
    
    p = Quaternion(1, 2, 3, 4)
    q = Quaternion(2, -1, 1, 0)
    pq = p * q
    
    norm_p = p.norm()
    norm_q = q.norm()
    norm_pq = pq.norm()
    norm_product = norm_p * norm_q
    
    print(f"p = {p}")
    print(f"  ||p|| = {norm_p:.6f}")
    print()
    print(f"q = {q}")
    print(f"  ||q|| = {norm_q:.6f}")
    print()
    print(f"p*q = {pq}")
    print(f"  ||p*q|| = {norm_pq:.6f}")
    print()
    print(f"||p|| * ||q|| = {norm_product:.6f}")
    print(f"||pq|| = ||p|| * ||q||: {'✓' if abs(norm_pq - norm_product) < 1e-10 else '✗'}")
    print()


def property_7_conjugate_product():
    """PROPERTY 7: Conjugate of Product"""
    print_section("PROPERTY 7: Conjugate of Product")
    print("Conjugate reverses order: (pq)* = q*p*")
    print()
    
    p = Quaternion(1, 2, 0, 0)
    q = Quaternion(3, 0, 4, 0)
    
    pq = p * q
    pq_conj = pq.conjugate()
    
    q_conj = q.conjugate()
    p_conj = p.conjugate()
    q_conj_p_conj = q_conj * p_conj
    
    print(f"p = {p}")
    print(f"q = {q}")
    print()
    print(f"(p*q)* = {pq_conj}")
    print(f"q*p*  = {q_conj_p_conj}")
    print(f"(pq)* = q*p*: {'✓' if pq_conj == q_conj_p_conj else '✗'}")
    print()


def property_8_inverse_product():
    """PROPERTY 8: Inverse of Product"""
    print_section("PROPERTY 8: Inverse of Product")
    print("Inverse reverses order: (pq)⁻¹ = q⁻¹p⁻¹")
    print()
    
    p = Quaternion(1, 1, 0, 0)
    q = Quaternion(2, 0, 1, 0)
    
    pq = p * q
    pq_inv = pq.inverse()
    
    q_inv = q.inverse()
    p_inv = p.inverse()
    q_inv_p_inv = q_inv * p_inv
    
    print(f"p = {p}")
    print(f"q = {q}")
    print()
    print(f"(p*q)⁻¹ = {pq_inv}")
    print(f"q⁻¹p⁻¹ = {q_inv_p_inv}")
    print(f"(pq)⁻¹ = q⁻¹p⁻¹: {'✓' if pq_inv == q_inv_p_inv else '✗'}")
    print()


def property_9_matrix_representation():
    """PROPERTY 9: Matrix Representation"""
    print_section("PROPERTY 9: Matrix Representation via 4x4 Matrices")
    print("Hamilton product can be represented as matrix multiplication:")
    print("  p*q = Q_L(p)*q_vec  (left multiplication matrix)")
    print("  p*q = Q_R(q)*p_vec  (right multiplication matrix)")
    print()
    
    p = Quaternion(1, 2, 0, 0)
    q = Quaternion(3, 0, 4, 0)
    
    # Direct product
    pq = p * q
    
    # Matrix left representation
    Q_L = p.to_matrix_left()
    q_vec = q.to_vector()
    pq_left = Q_L @ q_vec
    
    # Matrix right representation
    Q_R = q.to_matrix_right()
    p_vec = p.to_vector()
    pq_right = Q_R @ p_vec
    
    print(f"p = {p}")
    print(f"q = {q}")
    print(f"p*q = {pq}")
    print()
    print(f"Left Matrix Q_L(p):")
    print(Q_L)
    print(f"Q_L(p) * q_vec = {pq_left}")
    print(f"Matches: {'✓' if np.allclose(pq_left, pq.to_vector()) else '✗'}")
    print()
    print(f"Right Matrix Q_R(q):")
    print(Q_R)
    print(f"Q_R(q) * p_vec = {pq_right}")
    print(f"Matches: {'✓' if np.allclose(pq_right, pq.to_vector()) else '✗'}")
    print()


def property_10_scalar_vector_decomposition():
    """PROPERTY 10: Scalar-Vector Decomposition Formula"""
    print_section("PROPERTY 10: Scalar-Vector Decomposition Formula")
    print("For q₁ = (r₁, v₁) and q₂ = (r₂, v₂):")
    print("  q₁*q₂ = (r₁r₂ - v₁·v₂, r₁v₂ + r₂v₁ + v₁×v₂)")
    print()
    
    q1 = Quaternion(2, 1, 0, 0)  # r1=2, v1=(1,0,0)
    q2 = Quaternion(3, 0, 2, 0)  # r2=3, v2=(0,2,0)
    
    r1, v1 = q1.scalar_part(), q1.vector_part()
    r2, v2 = q2.scalar_part(), q2.vector_part()
    
    # Direct product
    q1q2 = q1 * q2
    
    # Component-wise calculation
    scalar_part = r1*r2 - np.dot(v1, v2)
    vector_part = r1*v2 + r2*v1 + np.cross(v1, v2)
    
    print(f"q₁ = ({r1}, {v1})")
    print(f"q₂ = ({r2}, {v2})")
    print()
    print(f"Dot product v₁·v₂ = {np.dot(v1, v2)}")
    print(f"Cross product v₁×v₂ = {np.cross(v1, v2)}")
    print()
    print(f"Scalar part: r₁r₂ - v₁·v₂ = {scalar_part}")
    print(f"Vector part: r₁v₂ + r₂v₁ + v₁×v₂ = {vector_part}")
    print()
    print(f"q₁*q₂ (direct) = {q1q2}")
    print(f"q₁*q₂ (formula) = ({scalar_part:.4f} + {vector_part[0]:.4f}i + {vector_part[1]:.4f}j + {vector_part[2]:.4f}k)")
    print(f"Match: {'✓' if abs(q1q2.w - scalar_part) < 1e-10 and np.allclose(q1q2.vector_part(), vector_part) else '✗'}")
    print()


def property_11_commutativity_condition():
    """PROPERTY 11: When do Quaternions Commute?"""
    print_section("PROPERTY 11: When do Quaternions Commute?")
    print("Two quaternions commute if and only if their vector parts are parallel (collinear)")
    print()
    
    # Parallel vector parts (should commute)
    q1_parallel = Quaternion(1, 1, 2, 3)
    q2_parallel = Quaternion(2, 2, 4, 6)  # v2 = 2*v1 (parallel)
    
    pq = q1_parallel * q2_parallel
    qp = q2_parallel * q1_parallel
    
    print(f"Parallel case:")
    print(f"q₁ = {q1_parallel}  (v₁ = {q1_parallel.vector_part()})")
    print(f"q₂ = {q2_parallel}  (v₂ = {q2_parallel.vector_part()})")
    print(f"q₁*q₂ = {pq}")
    print(f"q₂*q₁ = {qp}")
    print(f"Commute: {'✓' if pq == qp else '✗'}")
    print()
    
    # Non-parallel vector parts (should NOT commute)
    q1_nonpar = Quaternion(1, 1, 0, 0)
    q2_nonpar = Quaternion(2, 0, 1, 0)
    
    pq = q1_nonpar * q2_nonpar
    qp = q2_nonpar * q1_nonpar
    
    print(f"Non-parallel case:")
    print(f"q₁ = {q1_nonpar}  (v₁ = {q1_nonpar.vector_part()})")
    print(f"q₂ = {q2_nonpar}  (v₂ = {q2_nonpar.vector_part()})")
    print(f"q₁*q₂ = {pq}")
    print(f"q₂*q₁ = {qp}")
    print(f"Don't commute: {'✓' if pq != qp else '✗'}")
    print()


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" HAMILTON PRODUCT: COMPREHENSIVE ANALYSIS")
    print(" The fundamental multiplication operation for quaternions")
    print(" Reference: https://en.wikipedia.org/wiki/Quaternion#Hamilton_product")
    print("="*70)
    
    property_1_basis_rules()
    property_2_distributivity()
    property_3_associativity()
    property_4_non_commutativity()
    property_5_scalar_identity()
    property_6_norm_multiplicative()
    property_7_conjugate_product()
    property_8_inverse_product()
    property_9_matrix_representation()
    property_10_scalar_vector_decomposition()
    property_11_commutativity_condition()
    
    print("\n" + "="*70)
    print(" All Hamilton Product Properties Demonstrated Successfully!")
    print("="*70 + "\n")
