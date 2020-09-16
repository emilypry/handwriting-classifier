from matplotlib import image 
from matplotlib import pyplot 
from PIL import Image
from numpy import asarray
from numpy import savetxt
import numpy as np
import os


# Only run once!
x = open('number_images.txt', 'x')
x.close()
x = open('number_images.txt', 'w')    

y = open('number_labels.txt', 'x')
y.close()
y = open('number_labels.txt', 'w')  

# Create a list to store all of the xs in (will convert to np array)
X_list = []

# Loop through each folder of numbers 
for folder in range(6):
  # Loop through each image in that folder 
  for filename in os.listdir('Numbers/%s' % folder):
    if filename.endswith('.png'): 
      # Open the image, make it grayscale
      image = Image.open('Numbers/%d/%s' % (folder,filename)).convert(mode='L') 
      image.thumbnail((30,30))   # Make the image smaller
      data = asarray(image)      # Turn the image into a matrix
      data = data.ravel()       # Unroll the image into a one-row vector

      X_list.append(data)

      y.write(str(folder))     # Write the number into a .txt file of ys
      y.write('\n')

X = np.array(X_list)    # Turn the X_list into a np matrix
print(X.shape)

savetxt(x, X.astype(int), fmt='%i') # Save the matrix as integers

x.close()
y.close()
