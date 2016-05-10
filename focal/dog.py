import numpy

MIDGET_OFF, MIDGET_ON, PARASOL_OFF, PARASOL_ON = range(4)

KERNEL_TYPES = [MIDGET_OFF, MIDGET_ON, PARASOL_OFF, PARASOL_ON]

KERNEL_STR = {"midget_off":  MIDGET_OFF,  MIDGET_OFF:  "midget_off",
              "midget_on":   MIDGET_ON,   MIDGET_ON:   "midget_on",
              "parasol_off": PARASOL_OFF, PARASOL_OFF: "parasol_off",
              "parasol_on":  PARASOL_ON,  PARASOL_ON:  "parasol_on"} 


class DifferenceOfGaussians():
  
  
  def __init__(self):
    self.max_num_kernels = 4
    self.kernels = None
    self.full_kernels = None
    self.kernels, self.full_kernels = self.create_all_kernels()


  def __getitem__(self, index):
    return self.kernels[index]


  @staticmethod
  def kernel_width(self, cell_type):
    is_off_centre, width, sigma, sigma_mult = self.get_params(cell_type)
    return width


  def create_single_kernel(self, cell_type):
    
    is_off_centre, width, sigma, sigma_mult = self.get_params(cell_type)
    
    kernels = self.diff_of_gauss(is_off_centre, width, sigma, sigma_mult)
            
    gauss1 = numpy.outer(kernels[0], kernels[1])
    gauss2 = numpy.outer(kernels[2], kernels[3])
    full_kernel = gauss1 + gauss2
    
    return kernels, full_kernel


  def create_all_kernels(self):
    kernels = {}
    full_kernels = {}

    for cell_type in range(self.max_num_kernels):
      kernels[cell_type], full_kernels[cell_type] = self.create_single_kernel(cell_type)
        
    return kernels, full_kernels


  def diff_of_gauss(self, is_off_centre, width, sigma, sigma_mult):
      half_width = width/2
      x, y = numpy.meshgrid(numpy.arange(-half_width, half_width + 1),
                            numpy.arange(-half_width, half_width + 1))
      y = -y

      coord_range = numpy.arange(-half_width, half_width + 1)
      
      sigma_c = sigma
      sigma_s = sigma_mult*sigma_c
      sigma_c2 = (sigma_c**2)
      sigma_s2 = (sigma_s**2)
      x2_plus_y2 = x**2 + y**2

      #get signs for selected centre-surround behaviour
      sign_sigma_c, sign_sigma_s = (-1., 1.) if is_off_centre == True else (1., -1.)

      #calculate 2D centre kernel, just to get normalizing weight
      kernel_c = sign_sigma_c*(1./(2.*numpy.pi*sigma_c2))*numpy.exp((-x2_plus_y2)/(2.*sigma_c2))
      mat_norm_weight = 1./numpy.sum(numpy.abs(kernel_c))
      #normalize it to sum to 1
      kernel_c *= mat_norm_weight
      
      #calculate 1D centre kernels and normalize them
      vec_norm_weight = sign_sigma_c*numpy.sqrt(mat_norm_weight)*numpy.sqrt(1./(2.*numpy.pi*sigma_c2))
      vertical_c = (numpy.exp((-coord_range**2)/(2.*sigma_c2))*vec_norm_weight).astype(numpy.float32)
      horizontal_c = (sign_sigma_c*vertical_c).astype(numpy.float32)
      
      #calculate 2D surround kernel, just to get normalizing weight
      kernel_s = sign_sigma_s*(1./(2.*numpy.pi*sigma_s2))*numpy.exp((-x2_plus_y2)/(2.*sigma_s2))
      mat_norm_weight = 1./numpy.sum(numpy.abs(kernel_s))
      #normalize it to sum to 1
      kernel_s *= mat_norm_weight
      
      #calculate 1D surround kernels and normalize them
      vec_norm_weight = sign_sigma_s*numpy.sqrt(mat_norm_weight)*numpy.sqrt(1./(2.*numpy.pi*sigma_s2))
      vertical_s = (numpy.exp((-coord_range**2)/(2.*sigma_s2))*vec_norm_weight).astype(numpy.float32)
      horizontal_s = (sign_sigma_s*vertical_s).astype(numpy.float32)
      
      #calculate difference of gaussians (sums to 0)
      kernel = kernel_s
      kernel += kernel_c
      
      #get auto-convolution to 1 normalization
      final_weight = 1.0/numpy.sqrt(numpy.sqrt(numpy.sum(kernel*kernel)))
      kernel /= numpy.sqrt(numpy.sum(kernel*kernel))
      
      vertical_c   *= final_weight
      horizontal_c *= final_weight
      vertical_s   *= final_weight
      horizontal_s *= final_weight
      
      return vertical_c, horizontal_c, vertical_s, horizontal_s


  def get_params(self, cell_centre_type):
    '''PARAMETERS FROM:
       Filter Overlap Correction ALgorithm, simulates the foveal pit
       region of the human retina.
       Created by Basabdatta Sen Bhattacharya.
       See DOI: 10.1109/TNN.2010.2048339
    '''
    
    if cell_centre_type == MIDGET_OFF:
      off_centre = True
      #width = 5
      width = 3
      sigma = 0.8
      #sigma_mult = 6.5
      sigma_mult = 6.7
    elif cell_centre_type == MIDGET_ON:
      off_centre = False
      width = 11
      sigma = 1.04
      #sigma_mult = 6.5
      sigma_mult = 6.7
    elif cell_centre_type == PARASOL_OFF:
      off_centre = True
      width = 61
      sigma = 8
      sigma_mult = 4.8
    elif cell_centre_type == PARASOL_ON:
      off_centre = False
      width = 243
      sigma = 10.4
      sigma_mult = 4.8

    return off_centre, width, sigma, sigma_mult



