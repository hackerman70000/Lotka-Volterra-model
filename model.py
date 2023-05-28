import matplotlib
import numpy as np
from numpy import inf

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def sim(variables, t, params):
    V = variables[0]  # Prey population
    P = variables[1]  # Predator population

    alpha = params[0]
    beta = params[1]
    epsilon = params[2]
    b = params[3]
    K = params[4]

    dVdt = (beta * (1 - V / K) - alpha * P) * V  # growth rate of Prey
    dPdt = (alpha * b * V - epsilon) * P  # growth rate of Predators

    return [dVdt, dPdt]


def plot_population_vs_time(t, y):
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8, 10))

    line1, = ax1.plot(t, y[:, 0], color="b")
    line2, = ax2.plot(t, y[:, 1], color="r")
    ax1.set_ylabel("Prey")
    ax1.set_xlabel("Time")
    ax2.set_ylabel("Predators")
    ax2.set_xlabel("Time")

    ax3.plot(t, y[:, 0], color="b", label="Prey")
    ax3.plot(t, y[:, 1], color="r", label="Predators")
    ax3.set_ylabel("Population")
    ax3.set_xlabel("Time")
    ax3.legend()

    ax1.set_title("Prey population over time")
    ax2.set_title("Predator population over time")
    ax3.set_title("Prey and predator population over time")
    plt.tight_layout()


def plot_phase_graph(y):
    fig_phase = plt.figure()
    ax_phase = fig_phase.add_subplot(111)
    line3, = ax_phase.plot(y[:, 0], y[:, 1], color="g")
    ax_phase.set_xlabel("Prey")
    ax_phase.set_ylabel("Predators")
    ax_phase.set_title("Phase portrait of prey and predator populations")

    arrow_interval = 1000

    for i in range(0, len(y) - 1, arrow_interval):
        ax_phase.annotate(
            '',
            xy=(y[i + 1, 0], y[i + 1, 1]),
            xytext=(y[i, 0], y[i, 1]),
            arrowprops=dict(
                arrowstyle='simple',
                lw=1.5,
                alpha=0.5,
                color='black'
            )
        )


if __name__ == '__main__':
    t = np.linspace(0, 50, num=1000)

    alpha = 0.4  # hunting efficiency
    beta = 1.1  # reproduction rate of the prey
    epsilon = 0.4  # mortality rate of predators
    b = 0.25  # reproduction rate of predators

    # Different use cases for K
    K1 = inf  # capacity of the environment
    K2 = 100
    K3 = 10

    y0 = [10, 2]  # [Prey, Predators] units

    params = [alpha, beta, epsilon, b, K1]

    y = odeint(sim, y0, t, args=(params,))

    plot_population_vs_time(t, y)
    plot_phase_graph(y)
    plt.show()
