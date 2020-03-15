""" Draw (onto)black(canvas) & white images with basic shapes (i.e. lines, circles, rectangles)
    via the functions below.

    Functions:
    - draw_horizontal_lines
    - draw_vertical_lines
    - draw_line_with_angle
    - draw_a_rectangle
    - draw_a_circle

    Feel free to mix and match them to have multiple shapes in an image
    by passing the same "img" parameter into the functions.
"""

import numpy as np
import cv2

from util_functions import *



""" @Params:

* width: Width of the image
* height: Height of the image
* thickness: Thickness of the line
* key: Sets where the line would stand in the image.
**  key options: start | middle | end | random
* line_count: Only matters if the key equals to 'random', otherwise there would
  always be 1 line in the image.
* img: When not supplied, a new image is created, otherwise the passed image is used.
"""
def draw_horizontal_lines( width, height, thickness, line_count=1, key='middle', img=None ):
    if thickness < 1 or thickness >= height:
        print( "Thickness value is invalid for this image. -> {}".format( thickness ) )
    else:
        img = np.zeros( (height, width) ) if img is None else img

        if key == 'random':
            random_idx = np.random.random_integers( 0, height-thickness, line_count )
            for i in range( line_count ):
                cv2.line( img, (0, random_idx[i]), (width-1, random_idx[i]), 255, thickness )
        else:
            if key == 'middle':
                idx = height//2
            elif key == 'start':
                idx = thickness//2
            elif key == 'end':
                idx = height-thickness//2
            cv2.line( img, (0, idx), (width-1, idx), 255, thickness )

        return img


""" For @Params see above.
"""
def draw_vertical_lines( width, height, thickness=1, line_count=1, key='middle', img=None ):
    if thickness < 1 or thickness >= width:
        print( "Thickness value is invalid for this image. -> {}".format( thickness ) )
    else:
        img = np.zeros( (height, width) ) if img is None else img

        if key == 'random':
            random_idx = np.random.random_integers( 0, width-thickness, line_count )
            for i in range( line_count ):
                cv2.line( img, (random_idx[i], 0), (random_idx[i], height-1), 255, thickness )
        else:
            if key == 'middle':
                idx = width//2
            elif key == 'start':
                idx = thickness//2
            elif key == 'end':
                idx = width-thickness//2
            cv2.line( img, (idx, 0), (idx, height-1), 255, thickness )

        return img


""" @Params:

* angle: Determines the angle of the line to be drawn.
    Might be either 45 or 135 degrees.

For the rest of the @Params see above.
"""
def draw_line_with_angle( width, height, angle, thickness=1, img=None ):
    if thickness < 1:
        print( "Thickness value is invalid for this image. -> {}".format( thickness ) )
    elif angle not in (45, 135):
        print( "Angle is not right. {} is not in [45, 135]".format( angle ) )
    else:
        img = np.zeros( (height, width) ) if img is None else img

        if angle == 45:
            cv2.line( img, (0, height-1), (width-1, 0), 255, thickness )
        elif angle == 135:
            cv2.line( img, (0, 0), (width-1, height-1), 255, thickness )

        return img


""" @Params:

* top_left_pt: Top left point tuple of the rectangle to be drawn, e.g. (10, 12)
* bottom_right_pt: Bottom right point tuple of the rectangle to be drawn, e.g. (29, 31)
* thickness: -1 to fill inside the rectangle, otherwise determines the thickness
    of the rectangle's outer lines.

For the rest of the @Params see above.
"""
def draw_a_rectangle( width, height, top_left_pt, bottom_right_pt, thickness=1, img=None ):
    img = np.zeros( (height, width) ) if img is None else img
    cv2.rectangle( img, top_left_pt, bottom_right_pt, 255, thickness ) # 255 is the colour, default is white obviously
    return img


""" @Params:

* center_pt: Center point tuple of the circle to be drawn, e.g. (14, 20)
* radius: Radius of the circle, e.g. 6.

For the rest of the @Params see function draw_a_rectangle.
"""
def draw_a_circle( width, height, center_pt, radius, thickness=1, img=None ):
    img = np.zeros( (height, width) ) if img is None else img
    cv2.circle( img, center_pt, radius, 255, thickness )
    return img



if __name__ == '__main__':
    # Some usage scenarios

    width = height = 32
    thickness = 3
    # img = draw_a_rectangle( width, height, (10, 12), (29, 31), -1 )
    # img = draw_a_circle( width, height, (14, 20), 6, 2 )

    img = draw_line_with_angle( width, height, 45, thickness )
    # img = draw_line_with_angle( width, height, 135, thickness, img )

    # img = draw_horizontal_lines( width, height, thickness )
    # img = draw_horizontal_lines( width, height, 1, 5, 'random' )

    # img = draw_vertical_lines( width, height, thickness, img=img )
    # img = draw_vertical_lines( width, height, 1, 5, 'random' )

    imshow_opencv( img )
    # img_name = "new_image_etc.png"
    # save_img( img_name, img )
