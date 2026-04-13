def float_ReLU(
    x: float,
) -> float:
    if x >= 0:
        return x
    if x < 0:
        return 0.3 * x


def clamped_value(
    num1: float,
    num2: float,
    x: float,
) -> float:
    """
    Clamp a number `x` to be within the range defined by `num1` and `num2`.
    Parameters:
    num1 (float): The lower bound of the range.
    num2 (float): The upper bound of the range.
    x (float): The number to be clamped.
    Returns:
    float: The clamped value of `x` such that `num1 <= x <= num2`.
    """
    if x >= num2:
        return num2
    if x <= num1:
        return num1
    return x
