import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tools import get_curve


def func_to_fit(theta, *coeffs):
    result = 0
    for n in range(1, len(coeffs)):
        result += coeffs[n] * np.cos(n * theta)
    return result


def objective_function(coeffs, theta, y, ):
    residuals = y - func_to_fit(theta, *coeffs)
    objective_value = np.sum(residuals ** 2)
    return objective_value


def select_top_coeffs(optimized_coeffs, k=4):
    if k >= len(optimized_coeffs):
        return optimized_coeffs

    result = np.zeros_like(optimized_coeffs)
    top_indices = np.argpartition(np.abs(optimized_coeffs), -k)[-k:]
    result[top_indices] = optimized_coeffs[top_indices]

    return result


def print_cosine_series(coeffs):
    terms = []
    for n, coeff in enumerate(coeffs):
        if coeff != 0:
            if n == 0:
                terms.append(f"{coeff}")
            else:
                terms.append(f"{coeff} * cos({n} * theta)")
    formula = " + ".join(terms)
    print(f"Fitted formula: {formula}")

theta0, r0 = get_curve(20)

original_r0 = r0
r0 = r0 - np.mean(r0)
initial_guess = np.zeros(20)
result = minimize(objective_function, initial_guess, args=(theta0, r0))
optimized_coeffs = result.x

k = 4

optimized_coeffs = select_top_coeffs(optimized_coeffs[:], k)

theta = np.linspace(0, 2 * np.pi, 100)
fitted_y = func_to_fit(theta, *optimized_coeffs)

print_cosine_series(optimized_coeffs)

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, projection='polar')
ax.plot(theta0 + np.pi / 2, original_r0, 'o', label='Original data', alpha=0.2)
ax.plot(theta + np.pi / 2, fitted_y + np.mean(original_r0), '-', label='Fitted curve')
ax.legend()

# ax.spines['polar'].set_visible(False)
# ax.set_xticks([])
# ax.yaxis.set_tick_params(labelcolor='none')
# ax.yaxis.set_ticks_position('none')
# ax.grid(False)
# ax.patch.set_visible(False)

plt.title('Elephant from ' + str(k) + ' non-zero parameters')
plt.savefig('elephant.pdf')
plt.show()
