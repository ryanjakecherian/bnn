def calc_desired_var(dim: int, bit_shift: int) -> float:
    bit_shift_var = 2 ** (bit_shift * 2)
    return  1 * bit_shift_var / dim

