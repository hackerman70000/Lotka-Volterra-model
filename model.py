import matplotlib
import numpy as np

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

    dVdt = (beta - alpha * P) * V  # growth rate of Preys
    dPdt = (alpha * b * V - epsilon) * P  # growth rate of Predators

    return [dVdt, dPdt]


def plot_population_vs_time(t, y):
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8, 10))

    # Plot population vs. time
    line1, = ax1.plot(t, y[:, 0], color="b")
    line2, = ax2.plot(t, y[:, 1], color="r")
    ax1.set_ylabel("Preys")
    ax2.set_ylabel("Predators")
    ax2.set_xlabel("Time")

    # Plot combined graph
    ax3.plot(t, y[:, 0], color="b", label="Preys")
    ax3.plot(t, y[:, 1], color="r", label="Predators")
    ax3.set_ylabel("Population")
    ax3.set_xlabel("Time")
    ax3.legend()

    plt.tight_layout()
    # plt.show()


def plot_phase_graph(y):
    fig_phase = plt.figure()
    ax_phase = fig_phase.add_subplot(111)
    line3, = ax_phase.plot(y[:, 0], y[:, 1], color="g")
    ax_phase.set_xlabel("Prey")
    ax_phase.set_ylabel("Predators")
    # plt.show()


if __name__ == '__main__':
    t = np.linspace(0, 50, num=1000)

    alpha = 0.4  # hunting efficiency
    beta = 1.1  # reproduction rate of the prey
    epsilon = 0.4  # mortality rate of predators
    b = 0.25  # reproduction rate of predators (proportion of "biomass" of captured prey utilized by predators in the reproductive process

    y0 = [10, 1]  # [Preys, Predators] units in hundreds
    # y0 = [epsilon/(alpha*b), beta/alpha]

    params = [alpha, beta, epsilon, b]

    y = odeint(sim, y0, t, args=(params,))

    plot_population_vs_time(t, y)
    plot_phase_graph(y)
    plt.show()
