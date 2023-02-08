import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from PIL import ImageFile
from PIL import ImageShow

from PIL import Image
frames = []
for index in range(394):
    frame = Image.open('frames/frame%i.png' % (index))
    frames.append(frame)

frames[0].save('pillow_imagedraw.gif',
               save_all=True, append_images=frames[1:], optimize=False, duration=40, loop=0)
