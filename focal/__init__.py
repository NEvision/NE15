from focal import *

def rgb2gray(self, rgb):
  #return numpy.int16(numpy.dot(rgb[:,:,:3], [0.299, 0.587, 0.144]))
  #return numpy.floor(numpy.dot(rgb[:,:,:3], [0.299, 0.587, 0.144]))
  return rgb[:,:,0]*0.299 + rgb[:,:,1]*0.587 + rgb[:,:,2]*0.144

def idx2coord(self, idx, width):
  return (int(idx/width), idx%width)

def spike_trains_to_images_g(spike_trains, base_img, num_kernels):
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

    thismanager = plt.get_current_fig_manager()
    #thismanager.window.SetPosition((1920, 0))
    thismanager.window.wm_geometry("+1921+0")
    plt.show()
    plt.close("all")

    
def save_images(images, prefix, cmap=plt.cm.Greys_r, title_source=idx_to_name): #plt.cm.Paired
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
    non_zero = 0
    for i in img_set:
        non_zero += numpy.sum(img_set[i] != 0)
    return non_zero
 
    
def convert_image_to_spikes(image):
  pass
