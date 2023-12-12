import numpy as np

def one_cycle_lr_schedule(iteration: int, max_lr: float, total_iterations: int, warmup_fraction: float=0.1) -> float:
    """Generate learning rate values based on the one-cycle learning rate schedule.

    This function calculates the learning rate for a given iteration using the one-cycle learning rate scheule.
    The schedule involves a warm-up phase followed by a cosine annealing phase.

    Args:
        iteration (int): The current iteration.
        max_lr (float): The maximum learning rate during the cycle.
        total_iterations (int): The total number of iterations in the cycle.
        warmup_fraction (float, optional): The fraction of total iterations used for warm-up. Defaults to 0.1.

    Returns:
        float: The computed learning rate for the given iteration.
    """
    warmup_iterations = int(total_iterations * warmup_fraction)
    
    if iteration < warmup_iterations:
        return max_lr * iteration / warmup_iterations
    
    cycle_iterations = total_iterations - warmup_iterations
    cycle_fraction = (iteration - warmup_iterations) / cycle_iterations
    return max_lr * (1 + np.cos(np.pi * cycle_fraction)) / 2