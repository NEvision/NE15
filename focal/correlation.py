from scipy.signal import sepfir2d, convolve2d
import numpy
import pickle
from os import listdir
from os.path import isfile, join
import sys
import md5

class Correlation():
  '''
    Class to compute the correlations between Difference of Gaussian
    kernels.
  '''

  def __init__(self, full_kernels):
    ''' :params: full_kernels - Dictionary that contains 2D DoG kernels 
                                for each simulated (ganglion cell) layer 
    '''
      
    self.full_kernels = full_kernels
    self.data = self.create_all_correlations()
  

  def kernels_to_string(self, kernels):
    '''helper for hashing'''
    
    my_string = ""
    for k in kernels:
        my_string = "%s%s"%(my_string, kernels[k])
    return my_string


  def create_all_correlations(self):

    mode = "full" #same, full, valid
    seed = self.kernels_to_string(self.full_kernels)
    seed += mode
    
    filename = "correlation-cache-%s.p"%( md5.new(seed).hexdigest() )
    
    if isfile(filename):
      correlations = pickle.load( open( filename, "rb" ) )
      print("Loaded correlations from file")
      return correlations
    
    correlations = {}
    
    num_kernels = len(self.full_kernels.keys())
    
    for cell_type in range(num_kernels):
      correlations[cell_type] = {}
      for overlap_cell_type in range(num_kernels):
        percent = ((cell_type*num_kernels+overlap_cell_type)*100. + 1.)/float(num_kernels**2)
        sys.stdout.write("\rCorrelations cell(%s, %s) %03d%%"%(cell_type, overlap_cell_type, percent))
        sys.stdout.flush()
        
        # using the correlate2d method threw different matrices than
        # the ones Basab gets
        correlation =  convolve2d(self.full_kernels[cell_type], 
                                  self.full_kernels[overlap_cell_type],
                                  boundary='fill', fillvalue=0, mode=mode)
        
        correlations[cell_type][overlap_cell_type] = correlation
        
    pickle.dump( correlations, open( filename, "wb" ) )
    
    print("All correlations calculated\n")

    return correlations


  def __getitem__(self, index):
    return self.data[index]
