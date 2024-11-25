JOINT_LIMITS = {
    1: (-180, 180),
    2: (-128.9, 128.9),
    3: (-180, 180),
    4: (-147.8, 147.8),
    5: (-180, 180),
    6: (-120.3, 120.3),
    7: (-180, 180),
}

def normalize_angle_cyclic(value):
    return (value + 180) % 360 - 180

def normalize_angle_with_limits(value, limits):
    min_limit, max_limit = limits
    if min_limit == -180 and max_limit == 180:
        return normalize_angle_cyclic(value)
    return max(min(value, max_limit), min_limit)

