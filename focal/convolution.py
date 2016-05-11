import numpy
from scipy.signal import sepfir2d

class Convolution():
  '''
    Utility class to wrap around different functions for convolution 
    (i.e. separable convolution)
  '''
  def __init__(self):
    pass
    
  def sep_convolution(self, img, horz_k, vert_k, col_keep=1, row_keep=1, mode="full"):
    ''' Separated convolution -
        img      => image to convolve
        horiz_k  => first convolution kernel vector (horizontal)
        vert_k   => second convolution kernel vector (horizontal)
        col_keep => which columns are we supposed to calculate
        row_keep => which rows are we supposed to calculate
        mode     => if "full": convolve all the image otherwise just valid pixels
    '''
    width  = img.shape[1]
    height = img.shape[0]
    half_k_width = horz_k.size/2
    half_img_width  = width/2
    half_img_height = height/2

    tmp = numpy.zeros_like(img, dtype=numpy.float32)

    if mode == "full":
      horizontal_range = xrange(width) 
      vertical_range   = xrange(height)
    else:
      horizontal_range = xrange(half_k_width, width  - half_k_width + 1)
      vertical_range   = xrange(half_k_width, height - half_k_width + 1)

    for y in xrange(height):
        for x in horizontal_range:
            if (x - half_img_width)%col_keep != 0:
                continue

            k_sum = 0.
            k = 0

            for i in xrange(-half_k_width, half_k_width + 1):
                img_idx = x + i
                if img_idx >= 0 and img_idx < img.shape[1]:
                    k_sum += img[y,img_idx]*horz_k[k]
                k += 1

            tmp[y,x] = k_sum

    tmp2 = numpy.zeros_like(img, dtype=numpy.float32)
    for y in vertical_range:
      if (y - half_img_height)%row_keep != 0:
        continue

      for x in horizontal_range:
        if (x - half_img_width)%col_keep != 0:
          continue

        k_sum = 0.
        k = 0
        for i in xrange(-half_k_width, half_k_width + 1):
          img_idx = y + i
          if img_idx >= 0 and img_idx < img.shape[0]:
            k_sum += tmp[img_idx, x]*vert_k[k]
              
          k += 1

        tmp2[y,x] = k_sum

    return tmp2


  def dog_sep_convolution(self, img, k, cell_type, originating_function="filter",
                          force_homebrew = False, mode="full"):
    ''' Wrapper for separated convolution for DoG kernels in FoCal, 
        enables use of NumPy based sepfir2d.
        
        img                  => the image to convolve
        k                    => 1D kernels to use
        cell_type            => ganglion cell type, useful for sampling 
                                resolution numbers
        originating_function => if "filter": use special sampling resolution,
                                else: use every pixel
        force_hombrew        => if True: use my code, else: NumPy's
        mode                 => "full" all image convolution, else only valid
    '''

    if originating_function == "filter":
        row_keep, col_keep = self.get_subsample_keepers(cell_type)
    else:
        row_keep, col_keep = 1, 1

    if not force_homebrew:
      # has a problem with images smaller than kernel
      right_img = sepfir2d(img.copy(), k[0], k[1])
      left_img  = sepfir2d(img.copy(), k[2], k[3])
    else:
      right_img = self.sep_convolution(img, k[0], k[1], col_keep=col_keep, 
                                       row_keep=row_keep, mode=mode)
      left_img  = self.sep_convolution(img, k[2], k[3], col_keep=col_keep, 
                                       row_keep=row_keep, mode=mode )

    conv_img = left_img + right_img

    if not force_homebrew and originating_function == "filter":
        conv_img = self.subsample(conv_img, cell_type)

    return conv_img


  def get_subsample_keepers(self, cell_type):
    ''' return which (modulo) columns and rows to keep for cell_type
    '''
    if cell_type > 1:
      #~ col_keep = 7
      #~ row_keep = 7
      col_keep = 5
      row_keep = 3
    else:
      col_keep = 1
      row_keep = 1

    return row_keep, col_keep


  def subsample(self, img, cell_type):
    ''' remove unwanted rows/columns '''
    row_keep, col_keep = self.get_subsample_keepers(cell_type)
    
    if col_keep < img.shape[1] and row_keep < img.shape[0]:
      width = img.shape[1]
      height = img.shape[0]
      half_img_width  = width/2
      half_img_height = height/2
      
      col_range = numpy.arange(width)
      row_range = numpy.arange(height)
      
      img[:, [x for x in col_range if (x - half_img_width)%(col_keep)!= 0]] = 0
      img[[x for x in row_range if (x - half_img_height)%(row_keep)!= 0], :] = 0
      #~ img[:, [x for x in col_range if (x)%(col_keep)!= 0]] = 0
      #~ img[[x for x in row_range if (x)%(row_keep)!= 0], :] = 0
    else:
      img[:,:] = 0

    return img
