import pickle
import cv2
import os
from matplotlib import pyplot as plt



#--- File Operations ----------------------------------------------------------#
def pickle_it( pickle_to_be, pickle_file_name ):
    if pickle_to_be is not None:
        dir_name = 'pickles'
        if not os.path.exists( dir_name ):
            os.makedirs( dir_name )

        pickle_file_name = "{}/{}".format( dir_name, pickle_file_name )
        pickle_file_obj = open( pickle_file_name, 'wb' ) # open the file for writing
        pickle.dump( pickle_to_be, pickle_file_obj, protocol=2 ) # so that python2.x can also read it
        pickle_file_obj.close()
        print( "Pickle is ready! -> {}".format( pickle_file_name ) )

        """ For more info., about bz2 etc.:
        https://www.datacamp.com/community/tutorials/pickle-python-tutorial """


def unpickle( pickle_file_name ):
    pickle_file_obj = open( pickle_file_name, 'rb' ) # read-binary
    pickled_one = pickle.load( pickle_file_obj )
    pickle_file_obj.close()
    return pickled_one


def save_img( img_file_name, img_matrix, show_image=True ):
    if img_matrix is not None:
        dir_name = 'images'
        if not os.path.exists( dir_name ):
            os.makedirs( dir_name )

        img_file_name = "{}/{}".format( dir_name, img_file_name )
        cv2.imwrite( img_file_name, img_matrix )

        if show_image:
            imshow_matplot( img_matrix )


#--- Display Images -----------------------------------------------------------#
def raster_plot_spike( spikes, marker='|', markersize=2 ):
    '''Plot PyNN SpikeSourceArrays
        :param spikes: The array containing spikes
    '''
    x = []
    y = []

    for neuron_id in range( len(spikes) ):
        for t in spikes[neuron_id]:
            x.append(t)
            y.append(neuron_id)

    plt.plot( x, y, marker, markersize=markersize )


def imshow_opencv( img ):
    if img is not None:
        window_name = 'Your New Image'
        height, width = img.shape
        window_size = 100 if (height < 50 and width < 50) else cv2.WINDOW_AUTOSIZE

        cv2.namedWindow( window_name, window_size )
        cv2.imshow( window_name, img )

        key = cv2.waitKey(0) & 0xFF
        if key == 27: # ESC key
            cv2.destroyAllWindows()
            """ Closing the window won't end the program! Press ESC both to close
            the window, and also kill the program. """


def imshow_matplot( img, hide_ticks=False ):
    if img is not None:
        plt.imshow( img, cmap = 'gray' )
        if hide_ticks:
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
