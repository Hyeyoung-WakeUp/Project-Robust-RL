MAX_EPISODE_STEPS = 500
POLE_SIZE = 0.2
BETA = 3
EPSILON = 0.0015
ENV = "normal"   # mountain_car #
# for Mountain Car
GRAVITY = 0.003

#Min and max for normalization
is_cartpole = True
XMAX = 0.418 if is_cartpole else 0.6
XMIN = -0.418 if is_cartpole else -1.2
YMAX = 4 if is_cartpole else 0.07
YMIN = -4 if is_cartpole else -0.07