import numpy as np

def calculate_los(r1, r2, R_planet=6371.0):
    # R_safe = R_planet + 100 # Margin
    R_safe = R_planet # Strict for testing
    
    d = r2 - r1
    f = r1
    
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - R_safe**2
    
    discriminant = b*b - 4*a*c
    
    if discriminant < 0:
        return True, "No intersection (Disc < 0)"
    else:
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        
        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return False, f"Blocked (t1={t1:.2f}, t2={t2:.2f})"
        
        # Check for passing through (t1 < 0 and t2 > 1)?
        # Since c > 0 (start outside), t1 and t2 must be same sign if they exist?
        # No, product of roots is c/a. c>0, a>0 -> product > 0.
        # So roots are either both positive or both negative.
        # If both negative: intersection was "behind" us.
        # If both positive: intersection is "ahead" of us.
        # So checking [0,1] is sufficient.
        
        return True, f"Clear (t1={t1:.2f}, t2={t2:.2f})"

# Test Cases
def run_tests():
    R = 6371.0
    alt = 1000.0
    r = R + alt
    
    # 1. Opposite sides of Earth (Should be Blocked)
    p1 = np.array([r, 0, 0])
    p2 = np.array([-r, 0, 0])
    print(f"Test 1 (Opposite): {calculate_los(p1, p2)}")

    # 2. 90 degrees apart (Should be Blocked if Chord cuts deep)
    # Chord distance d from center at midpoint:
    # Midpoint = (p1+p2)/2 = [r/2, r/2, 0]. Len = r/sqrt(2) = r * 0.707
    # r*0.707 = 7371 * 0.707 = 5211 < 6371.
    # So midpoint is inside Earth. Blocked.
    p3 = np.array([0, r, 0])
    print(f"Test 2 (90 deg): {calculate_los(p1, p3)}")
    
    # 3. Close neighbors (Should be Clear)
    # 10 degrees apart
    angle = np.radians(10)
    p4 = np.array([r * np.cos(angle), r * np.sin(angle), 0])
    print(f"Test 3 (10 deg): {calculate_los(p1, p4)}")
    
    # 4. Tangent grazing?
    # Horizon angle alpha = arccos(R/r)
    # cos(alpha) = 6371/7371 = 0.86
    # alpha = 30 degrees.
    # So if separation is > 60 degrees, it should block.
    # Let's test 61 degrees.
    angle_block = np.radians(61)
    p5 = np.array([r * np.cos(angle_block), r * np.sin(angle_block), 0])
    print(f"Test 4 (61 deg - Grazing Block): {calculate_los(p1, p5)}")

    angle_clear = np.radians(59)
    p6 = np.array([r * np.cos(angle_clear), r * np.sin(angle_clear), 0])
    print(f"Test 5 (59 deg - Grazing Clear): {calculate_los(p1, p6)}")

run_tests()
