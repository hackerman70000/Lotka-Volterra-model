import numpy as np
import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def sim(variables, t, params):
    # fish population level
    x = variables[0]
    # bear population level
    y = variables[1]

    alpha = params[0]
    beta = params[1]
    delta = params[2]
    gamma = params[3]

    dxdt = alpha * x - beta * x * y  # growth rate of fish population
    dydt = delta * x * y - gamma * y  # growth rate of bear population

    return [dxdt, dydt]


def plot_population_vs_time(t, y):
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8, 10))

    # Plot population vs. time
    line1, = ax1.plot(t, y[:, 0], color="b")
    line2, = ax2.plot(t, y[:, 1], color="r")
    ax1.set_ylabel("Fish (hundreds)")
    ax2.set_ylabel("Bears (hundreds)")
    ax2.set_xlabel("Time")

    # Plot combined graph
    ax3.plot(t, y[:, 0], color="b", label="Fish")
    ax3.plot(t, y[:, 1], color="r", label="Bears")
    ax3.set_ylabel("Population (hundreds)")
    ax3.set_xlabel("Time")
    ax3.legend()

    plt.tight_layout()
    # plt.show()


def plot_phase_graph(y):
    fig_phase = plt.figure()
    ax_phase = fig_phase.add_subplot(111)
    line3, = ax_phase.plot(y[:, 0], y[:, 1], color="g")
    ax_phase.set_xlabel("Fish (hundreds)")
    ax_phase.set_ylabel("Bears (hundreds)")
    # plt.show()


if __name__ == '__main__':
    t = np.linspace(0, 50, num=1000)

    alpha = 1.1
    beta = 0.4
    delta = 0.1
    gamma = 0.4

    y0 = [10, 1]  # [fish, bears] units in hundreds
    # y0 = [gamma/delta, alpha/beta]  # [fish, bears] units in hundreds
    params = [alpha, beta, delta, gamma]

    y = odeint(sim, y0, t, args=(params,))

    plot_population_vs_time(t, y)
    plot_phase_graph(y)
    plt.show()
