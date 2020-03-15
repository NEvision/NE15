import numpy as np
import cv2
import sys
import os

import pylab

from poisson_tools import image_to_poisson_trains
from util_functions import *



def img_to_spike_array( img_file_name, save_as_pickle=True ):
    img = cv2.imread( img_file_name, cv2.IMREAD_GRAYSCALE )
    if img is not None:
        height, width = img.shape

        spikes = image_to_poisson_trains( np.array( [img.reshape(height*width)] ), # notice reshape
                                          height, width,
                                          max_freq, on_duration, off_duration )
        pylab.figure()
        raster_plot_spike( spikes )
        pylab.show()

        #--- Pickle the spike array for further use -------------------------------------------#
        if save_as_pickle:
            img_file_name = img_file_name[ img_file_name.rfind('/')+1 : img_file_name.rfind('.') ]
            pickle_file = "spike_array_{}".format( img_file_name )
            pickle_it( spikes, pickle_file )
    else:
        print( "Image couldn't be read! -> from file ({}) to ({})".format( img_file_name, img ) )



if __name__ == '__main__':
    if len( sys.argv ) != 2 and len( sys.argv ) != 5:
        print( "Usage:" )
        print( "\t python  convert_image_to_spike_array.py  <img_file_name>  <max_freq>  <on_duration>  <off_duration>" )
        print( "or  (with the default values for up to a 32x32 image  {max_freq=1000}  {on_duration=200}  {off_duration=100}):" )
        print( "\t python  convert_image_to_spike_array.py  <img_file_name>" )
    else:
        img_file_name = sys.argv[1]

        if len( sys.argv ) > 2:
            max_freq = int(sys.argv[2])       # Hz
            on_duration = int(sys.argv[3])    # ms
            off_duration = int(sys.argv[4])   # ms
        else:
            max_freq = 1000      # Hz
            on_duration = 200    # ms
            off_duration = 100   # ms

        print( "max_freq: {}".format( max_freq ) )
        print( "on_duration: {}".format( on_duration ) )
        print( "off_duration: {}".format( off_duration ) )

        if os.path.isdir( img_file_name ):
            import glob2
            image_list = glob2.glob( os.path.join( img_file_name, "**/*.png" ) )
            for img in image_list:
                if os.path.isfile( img ):
                    img_to_spike_array( img )
        elif os.path.isfile( img_file_name ):
            img_to_spike_array( img_file_name )
