class DifferenceOfGaussians():
  
  
  def __init__(self):
    self.max_num_kernels = 4
    self.kernels, self.full_kernels = self.create_all_kernels()
    self.midget_off, self.midget_on, self.parasol_off, self.parasol_on = range(4)
    self.idx_to_name = ["midget_off", "midget_on", "parasol_off", "parasol_on"]


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
    
    return kernels, full_kernels


  def create_all_kernels(self):
    kernels = {}
    full_kernels = {}

    for cell_type in range(self.max_num_kernels):
      kernels[cell_type], full_kernels[cell_type] = self.create_single_kernel(cell_type)
        
    return kernels, full_kernels


  def diff_of_gauss(is_off_centre, width, sigma, sigma_mult):
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
      
      vertical_c *=
      return vertical_c*final_weight, horizontal_c*final_weight, vertical_s*final_weight, horizontal_s*final_weight

  def get_params(self, cell_centre_type):
      if cell_centre_type == midget_off:
          off_centre = True
          #surround_width = 5
          surround_width = 3
          sigma = 0.8
          #sigma_mult = 6.5
          sigma_mult = 6.7
      elif cell_centre_type == midget_on:
          off_centre = False
          surround_width = 11
          sigma = 1.04
          #sigma_mult = 6.5
          sigma_mult = 6.7
      elif cell_centre_type == parasol_off:
          off_centre = True
          surround_width = 61
          sigma = 8
          sigma_mult = 4.8
      elif cell_centre_type == parasol_on:
          off_centre = False
          surround_width = 243
          sigma = 10.4
          sigma_mult = 4.8

      return off_centre, surround_width, sigma, sigma_mult



