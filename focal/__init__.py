from focal import *
from convolution import *
from correlation import *
from dog import *

import pylab as plt

idx_to_name = ["midget_off", "midget_on", "parasol_off", "parasol_on"]


def rgb2gray(rgb):
    '''Convert an RGB array into grayscale
    '''
    #return numpy.int16(numpy.dot(rgb[:,:,:3], [0.299, 0.587, 0.144]))
    #return numpy.floor(numpy.dot(rgb[:,:,:3], [0.299, 0.587, 0.144]))
    return rgb[:,:,0]*0.299 + rgb[:,:,1]*0.587 + rgb[:,:,2]*0.144


def idx2coord(idx, width):
    '''Convert a 1D index into a 2D coordinate'''
    return (int(idx/width), idx%width)


def spike_trains_to_images_g(spike_trains, base_img, num_kernels=4):
    '''Transform a FoCal spike set into images
       :param spike_trains: Focal encoded image
       :param base_img: Original image, used to get the shape
       :num_kernels: How many of the representations to convert
    '''
    imgs ={}
    for cell_type in range(num_kernels):
        imgs[cell_type] = numpy.zeros_like(base_img, dtype=numpy.float32)
        
        
    for idx, val, cell_type in spike_trains:
        adjusted_idx = idx
        coords = idx2coord(adjusted_idx, base_img.shape[1])
        imgs[cell_type][coords] = val
        if val == numpy.nan:
            print "spike is nan"
        if val == numpy.inf:
            print "spike is inf"
        if val == -numpy.inf:
            print "spike is -inf"
        
    return imgs


def plot_images(images, original_img=None, use_abs_vals=False):
    '''Create a figure with:
         - If images is a dictionary: all images in it
         - If original_img is not None: Two pictures (original and images)
         - Finally just plot one image (images)
    '''
    plt.close("all")
    if type(images) == dict:
        #num_kernels = numpy.int(numpy.sqrt(len(images.keys())))
        num_kernels = len(images.keys())
        fig = plt.figure()#figsize=(num_kernels, 2), dpi=300)
        fig.clear()
        ax= plt.subplot(2, num_kernels + 1, 1)
        ax.imshow(original_img, cmap=plt.cm.Greys_r)
        ax.get_axes().axis('off')

        for cell_type in images:
            idx = cell_type
            slot = 2 + cell_type
            slot += 1 if cell_type >= num_kernels else 0
            ax= plt.subplot(2, num_kernels + 1, slot)
            if use_abs_vals:
                img = numpy.abs(images[idx].copy()) 
            else:
                img = images[idx].copy()
            ax.imshow(img, cmap=plt.cm.Greys_r)
            ax.get_axes().axis('off')
    elif original_img is not None:
        fig = plt.figure()#figsize=(4, 2), dpi=300)
        fig.clear()
        ax= plt.subplot(1, 2, 1)
        ax.imshow(original_img, cmap=plt.cm.Greys_r)
        ax.get_axes().axis('off')

        ax= plt.subplot(1, 2, 2)
        ax.imshow(images, cmap=plt.cm.Greys_r)
        ax.get_axes().axis('off')
    else:
        fig = plt.figure()#figsize=(4, 2), dpi=300)
        fig.clear()
        plt.imshow(images, cmap=plt.cm.Greys_r)
        plt.axis('off')

    # thismanager = plt.get_current_fig_manager()
    # thismanager.window.SetPosition((1920, 0))
    # thismanager.window.wm_geometry("+1921+0")
    plt.show()
    plt.close("all")


def save_images(images, prefix, cmap=plt.cm.Greys_r, title_source=idx_to_name): #plt.cm.Paired
    '''Save figures with filename = prefix-cell_type if images is a dictionary
                         filename = prefix otherwise
    '''
    if type(images) == dict:
        num_kernels = len(images.keys())

        for cell_type in images:
            
            img = images[cell_type]
            img_min = numpy.min(img)
            img_max = numpy.max(img)
            negatives = numpy.sum(img<0)
            positives = numpy.sum(img>0)

            # define the colormap
            
            fig = plt.figure(figsize=(3,3), dpi=300)
            
            fig.clear()
            

            im = plt.imshow(img, interpolation='none', cmap=cmap)
            plt.tight_layout()
            plt.colorbar(im, use_gridspec=True)
            plt.title("%s \n min = %10.4f, max = %10.4f\n num pos = %s, num neg = %s"%\
                       (title_source[cell_type], img_min, img_max, positives, negatives))
            
            plt.axis('off')
            plt.savefig("%s-%s"%(prefix, cell_type), dpi=300)
    else:
        fig = plt.figure(figsize=(3,3), dpi=300)
        fig.clear()
        plt.imshow(images, cmap=plt.cm.Greys_r)
        plt.axis('off')
        plt.savefig(prefix, dpi=300)
    
    plt.close("all")


def count_non_zero(img_set):
    '''Count non-zero pixels in an image set
    '''
    non_zero = 0
    for i in img_set:
        non_zero += numpy.sum(img_set[i] != 0)
    return non_zero


def focal_to_spike(spikes, img_shape, spikes_per_time_block=10, start_time=0., time_step=1.):
    '''Convert FoCal-coded spikes into a SpikeSourceArray
       :param spikes: FoCal, rank order-coded spikes
       :param img_shape: (Height, Width) of image
       :param spikes_per_time_block: How many spikes will go into each time bin
       :param start_time: When did the spikes start appearing (milliseconds)
       :param time_step: How much time between time bins
    '''
    neurons_per_layer = img_shape[0]*img_shape[1]
    width = img_shape[1]
    height = img_shape[0]
    total_width = 2*width
    total_height = 2*height 
    spike_array = [[] for i in range(total_height*total_width)]
    pack_time = start_time
    spikes_per_block_count = 0
    for spike in spikes:
        layer = spike[2]
        pad_x = width  if layer == 1 or layer == 3 else 0
        pad_y = height if layer == 2 or layer == 3 else 0
        
        loc_idx = spike[0]
        loc_x = loc_idx%width
        loc_y = loc_idx/width
        glb_x = pad_x + loc_x
        glb_y = pad_y + loc_y
        glb_idx = glb_y*total_width + glb_x
        
        spike_array[glb_idx].append(pack_time)
        
        spikes_per_block_count += 1
        if spikes_per_block_count == spikes_per_time_block:
            spikes_per_block_count = 0
            pack_time += time_step
        
    return spike_array
    

def raster_plot_spike(spikes, marker='|', markersize=2):
    '''Plot PyNN SpikeSourceArrays
        :param spikes: The array containing spikes
    '''
    x = []
    y = []
    
    for neuron_id in range(len(spikes)):
        for t in spikes[neuron_id]:
            x.append(t)
            y.append(neuron_id)
    
    plt.plot(x, y, marker, markersize=markersize)
    
