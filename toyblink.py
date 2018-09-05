# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 17:55:58 2017

@author: kremin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation




def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
fig = plt.figure()
ims = []
for i in range(2):
    x += np.pi / 15.
    y += np.pi / 20.
    im = plt.imshow(f(x, y), animated=True)
    ims.append([im])


ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True)#,repeat_delay=1000)

ani.save('dynamic_images.mp4')

plt.show()