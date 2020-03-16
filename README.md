# Convert Images to Poissonian Spikes

Convert your image data to a Poisson spike source to be able to use with Spiking Neural Networks.

<p align="center">
  <table>
    <tr>
      <td> <img src="images/pumpkins.jpeg" alt="Pumpkins-RGB" height="120"> </td>
      <td> &rarr; </td>
      <td> <img src="images/pumpkins_gray.jpeg" alt="Pumpkins-GrayScale" height="120"> </td>
      <td> &rarr; </td>
      <td> <img src="images/spikes_plot_pumpkins.png" alt="Pumpkins-SpikesPlot" height="135"> </td> 
    </tr>
  </table>
</p>  

<i>
  The parameters below are used when running <a href="convert_image_to_spike_array.py">convert_image_to_spike_array.py</a> in order to turn <a href="https://unsplash.com/photos/KnZDAYgRsz8">pumpkins</a> above into a spike array. 
  <br> max_freq = 60000 (Hz)
  <br> on_duration = 10000 (ms)
  <br> off_duration = 5000 (ms)
</i>

### Requirements
I use Python 3.5.2 on Linux, necessary packages are listed below along with their versions for reference.
* matplotlib (3.0.3)
* numpy (1.17.3)
* opencv-python (4.1.1.26)

Run `pip install -r requirements.txt` to install them all.

### Project Files and Their Usage
```
images-to-spikes/
├── convert_image_to_spike_array.py
├── draw_image.py
├── images
│   ├── cross.png
│   ├── horizontal_line_10x.png
│   ├── horizontal_lines.png
│   └── t10k-images-idx3-ubyte__idx_000__lbl_7_.png
├── poisson_tools.py
└── util_functions.py
```
**[convert_image_to_spike_array.py](convert_image_to_spike_array.py)** is the main file. 
  - Please see its usage by running it: `python convert_image_to_spike_array.py`
  - The program will store the output spike array as a _pickle_ under _pickles/_ folder in the same directory after the run. 
  - If you do not want a _pickle_ at the end, change the parameter inside the file, i.e. `save_as_pickle=False`.
  - You may use a single image file (extension could be anything _OpenCV_ accepts) or a folder which contains multiple images (extensions need to be _.png_) as input.

**[draw_image.py](draw_image.py)** enables you to draw your own images by adding simple shapes into it via _OpenCV_. For more information please see the file.

**[images](images/)** folder contains three of the images that I generated by using _draw_image.py_, and one example from MNIST dataset (t10k-images-idx3-ubyte__idx_000__lbl_7_.png).

**[poisson_tools.py](poisson_tools.py)** is where the Poisson distribution modelling takes place.

**[util_functions.py](util_functions.py)** includes utility functions of files and images.

## References and Citation
I only used the Poissonian spikes approach to obtain spike arrays from images in this project. The original project also contains _Focal Rank Code Order_ approach in this sense.

Please refer to the original project's [Wiki page](https://github.com/NEvision/NE15/wiki) for further information.
