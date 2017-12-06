import numpy
import matplotlib.pyplot as plt
from matplotlib import colors


class Gui(object):

    def __init__(self, nelx, nely):
        # Initialize plot and plot the initial design
        plt.ion()  # Ensure that redrawing is possible
        self.fig, ax = plt.subplots()
        self.im = ax.imshow(-numpy.zeros((nelx, nely)).T, cmap='gray',
                            interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
        self.fig.canvas.set_window_title("[Float] Resolution " + str(nelx) + "x" + str(nely))
        self.fig.show()
        self.nelx, self.nely = nelx, nely

    def update(self, xPhys):
        # Plot to screen
        self.im.set_array(-xPhys.reshape((self.nelx, self.nely)).T)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    @staticmethod
    def wait():
        # Make sure the plot stays and that the shell remains
        plt.show()
        input("Press any key...")
