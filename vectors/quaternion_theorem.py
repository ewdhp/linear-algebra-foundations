"""
quaternion_theorem.py
Demonstrates fundamental properties and theorems of quaternions from Wikipedia.
Reference: https://en.wikipedia.org/wiki/Quaternion

Key theorems covered:
1. Hamilton product (non-commutative multiplication)
2. Basis element multiplication rules (i²=j²=k²=ijk=-1)
3. Conjugation and norm properties
4. Multiplicative inverse
5. Unit quaternions (versors)
6. Rotation representation via quaternions
7. Polar decomposition
8. Exponential and logarithm of quaternions
"""

import numpy as np
from math import sqrt, cos, sin, acos, atan2

class Quaternion:
    """
    Represents a quaternion: q = w + xi + yj + zk
    where w is the scalar part and (x,y,z) is the vector part.
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
        Hamilton product (non-commutative multiplication).
        Rules: i²=j²=k²=ijk=-1
        ij=-ji=k, jk=-kj=i, ki=-ik=j
        """
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return Quaternion(w, x, y, z)

    def __add__(self, other):
        """Vector space addition"""
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        """Vector space subtraction"""
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)

    def __rmul__(self, scalar):
        """Scalar multiplication"""
        return Quaternion(scalar*self.w, scalar*self.x, scalar*self.y, scalar*self.z)

    def conjugate(self):
        """Conjugate: q* = w - xi - yj - zk"""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self):
        """Norm (magnitude): ||q|| = sqrt(w² + x² + y² + z²)"""
        return sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def norm_squared(self):
        """Norm squared: ||q||² = q*q*"""
        return self.w**2 + self.x**2 + self.y**2 + self.z**2

    def inverse(self):
        """
        Multiplicative inverse: q⁻¹ = q* / ||q||²
        Such that q * q⁻¹ = 1
        """
        n2 = self.norm_squared()
        if n2 < 1e-10:
            raise ValueError("Cannot invert zero quaternion")
        conj = self.conjugate()
        return Quaternion(conj.w/n2, conj.x/n2, conj.y/n2, conj.z/n2)

    def normalize(self):
        """Unit quaternion (versor): U_q = q / ||q||"""
        n = self.norm()
        if n < 1e-10:
            raise ValueError("Cannot normalize zero quaternion")
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)

    def dot_product(self, other):
        """Dot product of quaternion vector parts"""
        return self.x*other.x + self.y*other.y + self.z*other.z

    def scalar_part(self):
        """Extract scalar part"""
        return self.w

    def vector_part(self):
        """Extract vector part (x, y, z)"""
        return np.array([self.x, self.y, self.z])

    def polar_form(self):
        """
        Polar decomposition: q = ||q|| * U_q
        where U_q is the unit quaternion (versor)
        """
        magnitude = self.norm()
        unit = self.normalize() if magnitude > 1e-10 else None
        return magnitude, unit

    def exponential(self):
        """
        Exponential of quaternion: exp(q) = e^w * (cos||v|| + (v/||v||)*sin||v||)
        where q = w + v (v is vector part)
        """
        w = self.w
        v_norm = sqrt(self.x**2 + self.y**2 + self.z**2)
        
        exp_w = np.exp(w)
        
        if v_norm < 1e-10:
            return Quaternion(exp_w, 0, 0, 0)
        
        cos_v = cos(v_norm)
        sin_v = sin(v_norm) / v_norm
        
        return Quaternion(
            exp_w * cos_v,
            exp_w * sin_v * self.x,
            exp_w * sin_v * self.y,
            exp_w * sin_v * self.z
        )

    def logarithm(self):
        """
        Logarithm of quaternion: ln(q) = ln||q|| + (v/||v||)*arccos(w/||q||)
        """
        norm = self.norm()
        if norm < 1e-10:
            raise ValueError("Cannot take logarithm of zero quaternion")
        
        v_norm = sqrt(self.x**2 + self.y**2 + self.z**2)
        angle = acos(min(1, max(-1, self.w / norm)))
        
        if v_norm < 1e-10:
            return Quaternion(np.log(norm), 0, 0, 0)
        
        scale = angle / v_norm
        return Quaternion(
            np.log(norm),
            scale * self.x,
            scale * self.y,
            scale * self.z
        )


def test_basis_multiplication():
    """Test: i²=j²=k²=ijk=-1"""
    print("="*60)
    print("THEOREM 1: Basis Element Multiplication")
    print("="*60)
    
    i = Quaternion(0, 1, 0, 0)
    j = Quaternion(0, 0, 1, 0)
    k = Quaternion(0, 0, 0, 1)
    one = Quaternion(1, 0, 0, 0)
    
    print(f"i² = {i * i} (should be -1)")
    print(f"j² = {j * j} (should be -1)")
    print(f"k² = {k * k} (should be -1)")
    print(f"ijk = {i * j * k} (should be -1)")
    print(f"ij = {i * j} (should equal k)")
    print(f"jk = {j * k} (should equal i)")
    print(f"ki = {k * i} (should equal j)")
    print()


def test_non_commutativity():
    """Test: Quaternion multiplication is not commutative"""
    print("="*60)
    print("THEOREM 2: Non-Commutativity of Multiplication")
    print("="*60)
    
    a = Quaternion(1, 2, 3, 4)
    b = Quaternion(5, 6, 7, 8)
    
    ab = a * b
    ba = b * a
    
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a*b = {ab}")
    print(f"b*a = {ba}")
    print(f"a*b ≠ b*a: {not (ab == ba)}")
    print()


def test_conjugate_and_norm():
    """Test: q*q* = ||q||²"""
    print("="*60)
    print("THEOREM 3: Conjugate and Norm Property")
    print("="*60)
    
    q = Quaternion(1, -2, 3, -4)
    q_conj = q.conjugate()
    product = q * q_conj
    norm_sq = q.norm_squared()
    
    print(f"q = {q}")
    print(f"q* = {q_conj}")
    print(f"q*q* = {product}")
    print(f"||q||² = {norm_sq:.6f}")
    print(f"q*q* = ||q||²: {abs(product.w - norm_sq) < 1e-10}")
    print()


def test_norm_multiplicativity():
    """Test: ||pq|| = ||p|| ||q||"""
    print("="*60)
    print("THEOREM 4: Norm Multiplicativity")
    print("="*60)
    
    p = Quaternion(1, 2, 3, 4)
    q = Quaternion(2, -1, 1, 0)
    
    pq = p * q
    norm_pq = pq.norm()
    norm_product = p.norm() * q.norm()
    
    print(f"p = {p}")
    print(f"q = {q}")
    print(f"||p|| = {p.norm():.6f}")
    print(f"||q|| = {q.norm():.6f}")
    print(f"||pq|| = {norm_pq:.6f}")
    print(f"||p|| ||q|| = {norm_product:.6f}")
    print(f"||pq|| = ||p|| ||q||: {abs(norm_pq - norm_product) < 1e-10}")
    print()


def test_inverse():
    """Test: q*q⁻¹ = 1"""
    print("="*60)
    print("THEOREM 5: Multiplicative Inverse")
    print("="*60)
    
    q = Quaternion(2, 1, -1, 2)
    q_inv = q.inverse()
    identity = q * q_inv
    one = Quaternion(1, 0, 0, 0)
    
    print(f"q = {q}")
    print(f"q⁻¹ = {q_inv}")
    print(f"q*q⁻¹ = {identity}")
    print(f"q*q⁻¹ ≈ 1: {identity == one}")
    print()


def test_unit_quaternions():
    """Test: Unit quaternions (versors) have norm 1"""
    print("="*60)
    print("THEOREM 6: Unit Quaternions (Versors)")
    print("="*60)
    
    q = Quaternion(1, 2, 3, 4)
    u = q.normalize()
    
    print(f"q = {q}")
    print(f"||q|| = {q.norm():.6f}")
    print(f"U_q = {u}")
    print(f"||U_q|| = {u.norm():.6f}")
    print()


def test_polar_decomposition():
    """Test: q = ||q|| * U_q"""
    print("="*60)
    print("THEOREM 7: Polar Decomposition")
    print("="*60)
    
    q = Quaternion(2, 4, -2, 4)
    magnitude, unit = q.polar_form()
    
    reconstructed = Quaternion(magnitude, 0, 0, 0) * unit
    
    print(f"q = {q}")
    print(f"||q|| = {magnitude:.6f}")
    print(f"U_q = {unit}")
    print(f"||q|| * U_q = {reconstructed}")
    print(f"Decomposition valid: {q == reconstructed}")
    print()


def test_rotation():
    """Test: Unit quaternion rotation of vectors"""
    print("="*60)
    print("THEOREM 8: 3D Rotations via Unit Quaternions")
    print("="*60)
    
    # Rotation by 90° around z-axis
    angle = np.pi / 2
    axis = np.array([0, 0, 1])  # z-axis
    
    # Unit quaternion for rotation: q = cos(θ/2) + sin(θ/2)(n_x i + n_y j + n_z k)
    q = Quaternion(
        cos(angle/2),
        sin(angle/2) * axis[0],
        sin(angle/2) * axis[1],
        sin(angle/2) * axis[2]
    )
    
    # Vector to rotate (in quaternion form with w=0)
    v = Quaternion(0, 1, 0, 0)  # unit vector along x
    
    # Rotation: v' = q*v*q*
    q_conj = q.conjugate()
    v_rotated = q * v * q_conj
    
    print(f"Quaternion q (90° rotation around z): {q}")
    print(f"Original vector v = (1, 0, 0): {v}")
    print(f"Rotated vector v' = q*v*q*: {v_rotated}")
    print(f"Expected: (0, 1, 0) ≈ (0, 1.0000, 0)")
    print()


def test_exponential_logarithm():
    """Test: exp(q) and ln(q)"""
    print("="*60)
    print("THEOREM 9: Exponential and Logarithm")
    print("="*60)
    
    q = Quaternion(0.5, 0.2, 0.3, 0.1)
    exp_q = q.exponential()
    
    print(f"q = {q}")
    print(f"exp(q) = {exp_q}")
    
    # For unit quaternion
    u = q.normalize()
    exp_u = u.exponential()
    print(f"\nUnit quaternion u = {u}")
    print(f"exp(u) = {exp_u}")
    print(f"||exp(u)|| = {exp_u.norm():.6f}")
    print()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("QUATERNION THEOREMS - COMPREHENSIVE DEMONSTRATION")
    print("Reference: https://en.wikipedia.org/wiki/Quaternion")
    print("="*60 + "\n")
    
    test_basis_multiplication()
    test_non_commutativity()
    test_conjugate_and_norm()
    test_norm_multiplicativity()
    test_inverse()
    test_unit_quaternions()
    test_polar_decomposition()
    test_rotation()
    test_exponential_logarithm()
    
    print("="*60)
    print("All theorem demonstrations completed!")
    print("="*60)
