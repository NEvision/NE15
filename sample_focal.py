import scipy
import numpy
import pylab
from focal import *

# load image
filename  = "./t10k-images-idx3-ubyte__idx_000__lbl_7_.png"
img = pylab.imread(filename) #grayscale image
pylab.figure()
pylab.imshow(img, cmap="Greys_r")

fcl = Focal()
num_kernels = len(fcl.kernels.full_kernels) # four simulated layers

# spikes contains a rank-ordered list of triples with the following
# information:
# [ pixel/neuron index (int), pixel value (float), layer id (int) ]
spikes = fcl.apply(img)

# we convert the spike list to 4 images, spike_imgs is a dictionary
# containing an image per simulated layer
spike_imgs = spike_trains_to_images_g(spikes, img, num_kernels)

pylab.figure()
i = 1
for k in spike_imgs.keys():
  pylab.subplot(2, 2, i)
  pylab.imshow(spike_imgs[k], cmap="Greys_r")
  i += 1

#convert to spike source array
pylab.figure()
spk_src = focal_to_spike(spikes, img.shape, 
                         spikes_per_time_block=10, 
                         start_time=0., time_step=1.)
raster_plot_spike(spk_src)

pylab.show()
