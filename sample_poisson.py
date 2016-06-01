import scipy
import numpy
import pylab
from poisson.poisson_tools import *
from focal import raster_plot_spike

# Demonstration file for the Poisson tools module. It also includes functions
# to load all the MNIST images/labels and converting between different formats

# load image
filename  = "./t10k-images-idx3-ubyte__idx_000__lbl_7_.png"
img = pylab.imread(filename)
height, width = img.shape

max_freq = 1000 #Hz
on_duration = 200 #ms
off_duration = 100 #ms
pylab.figure()
pylab.imshow(img, cmap="Greys_r")

spikes = mnist_poisson_gen(numpy.array([img.reshape(height*width)]), #notice reshape
                           height, width, 
                           max_freq, on_duration, off_duration)

pylab.figure()

raster_plot_spike(spikes)

pylab.show()
