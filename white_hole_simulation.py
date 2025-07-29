import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp

def lqg_dynamics(t, y, rho_c, rho_0):
    a, da_dt = y
    if a <= 0:  # Prevent negative scale factor
        return [0, 0]
    rho = rho_0 / a**3
    d2a_dt2 = - (4 * np.pi / 3) * rho * (1 - 2 * rho / rho_c) * a
    return [da_dt, d2a_dt2]

rho_0 = 1.0
rho_c = 100.0
a0 = 1.0
initial_rho = rho_0 / a0**3
H0 = np.sqrt((8 * np.pi / 3) * initial_rho * (1 - initial_rho / rho_c))
da0 = -a0 * H0
t_span = [0, 4]

def bounce_event(t, y, rho_c, rho_0):
    return y[1]
bounce_event.terminal = False
bounce_event.direction = 1

sol = solve_ivp(
    lqg_dynamics,
    t_span,
    [a0, da0],
    args=(rho_c, rho_0),
    method='RK45',
    rtol=1e-10,
    atol=1e-10,
    events=bounce_event
)

min_idx = np.argmin(sol.y[0])
a_min = sol.y[0][min_idx]
t_bounce = sol.t[min_idx]
a_min_sym = (rho_0 / rho_c)**sp.Rational(1, 3)

print(f"Minimum scale factor (numerical): {a_min:.10f}")
print(f"Theoretical bounce point: {float(a_min_sym):.10f}")
print(f"Bounce time: {t_bounce:.6f}")
print(f"Final scale factor at t=4: {sol.y[0][-1]:.4f}")
