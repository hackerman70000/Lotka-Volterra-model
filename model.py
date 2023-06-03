import matplotlib
import numpy as np
from numpy import inf

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.widgets import Slider
from matplotlib import gridspec


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


def plot_population_vs_time(t, y, y_phase):
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 10))

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

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(y_phase[:, 0], y_phase[:, 1], color="g")
    ax4.set_xlabel("Prey")
    ax4.set_ylabel("Predators")
    ax4.set_title("Phase portrait of prey and predator populations")

    arrow_interval = 1000

    for i in range(0, len(y) - 1, arrow_interval):
        ax4.annotate(
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

    plt.tight_layout()


def update(val):
    alpha = s_alpha.val
    beta = s_beta.val
    epsilon = s_epsilon.val
    b = s_b.val
    K = s_K.val

    params = [alpha, beta, epsilon, b, K]

    y = odeint(sim, y0, t, args=(params,))
    y_phase = y[:, :2]

    for line in lines:
        line.set_ydata(y[:, lines.index(line)])

    line_phase.set_xdata(y_phase[:, 0])
    line_phase.set_ydata(y_phase[:, 1])

    fig.canvas.draw_idle()


if __name__ == '__main__':
    t = np.linspace(0, 50, num=1000)

    alpha = 0.4  # hunting efficiency
    beta = 1.1  # reproduction rate of the prey
    epsilon = 0.4  # mortality rate of predators
    b = 0.25  # reproduction rate of predators
    K1 = inf  # capacity of the environment


    y0 = [10, 2]  # [Prey, Predators] units

    params = [alpha, beta, epsilon, b, K1]

    y = odeint(sim, y0, t, args=(params,))
    y_phase = y[:, :2]

    fig = plt.figure(figsize=(18, 5))  # Adjust the figure size
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1])  # Adjust the grid specifications

    ax1 = plt.subplot2grid((5, 5), (0, 0), rowspan=4, colspan=4)
    ax2 = plt.subplot2grid((5, 5), (0, 4), rowspan=4, colspan=1)

    lines = ax1.plot(t, y)
    line_phase, = ax2.plot(y_phase[:, 0], y_phase[:, 1], color="g")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Population")
    ax1.legend(['Prey', 'Predators'])
    ax1.set_title("Prey and predator populations over time")

    ax2.set_xlabel("Prey")
    ax2.set_ylabel("Predators")
    ax2.set_title("Phase portrait of prey and predator populations")

    axcolor = 'lightgoldenrodyellow'
    ax_alpha = plt.axes([0.15, 0.05, 0.3, 0.03], facecolor=axcolor)
    ax_epsilon = plt.axes([0.15, 0.1, 0.3, 0.03], facecolor=axcolor)
    ax_b = plt.axes([0.65, 0.1, 0.3, 0.03], facecolor=axcolor)
    ax_beta = plt.axes([0.65, 0.05, 0.3, 0.03], facecolor=axcolor)
    ax_K = plt.axes([0.15, 0.15, 0.3, 0.03], facecolor=axcolor)

    s_alpha = Slider(ax_alpha, 'Alpha - hunting efficiency', 0.1, 1.0, valinit=alpha)
    s_epsilon = Slider(ax_epsilon, 'Epsilon - mortality rate of predators', 0.1, 1.0, valinit=epsilon)
    s_K = Slider(ax_K, 'K - capacity of the environment', 1, 100, valinit=K1)
    s_b = Slider(ax_b, 'b - reproduction rate of predators', 0.1, 1.0, valinit=b)
    s_beta = Slider(ax_beta, 'Beta - reproduction rate of the prey', 0.1, 2.0, valinit=beta)


    s_alpha.on_changed(update)
    s_beta.on_changed(update)
    s_epsilon.on_changed(update)
    s_b.on_changed(update)
    s_K.on_changed(update)

    plt.show()
