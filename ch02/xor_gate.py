from and_gate import AND
from or_gate import OR
from nand_gate import NAND

def XOR(x1: int, x2: int) -> int:
    """Implmentation of the XOR gate using NAND, OR, and AND gates.

    Args:
        x1 (int): Input 1 (0 or 1)
        x2 (int): Input 2 (0 or 1)

    Returns:
        int: Output of the XOR gate (0 or 1)
    """
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

if __name__ == '__main__':
    test_cases = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for xs in test_cases:
        y = XOR(xs[0], xs[1])
        print(f"{xs} -> {y}")