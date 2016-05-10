import scipy
import numpy
import pylab
from focal import Focal, rgb2gray, spike_trains_to_images_g

# load image
filename  = "./lena.jpg"
img = rgb2gray(pylab.imread(filename))
img = scipy.misc.imresize(img, (256, 256)) # reduce size to improve time
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

pylab.show()
