# How to build a image dataset object.
# How to perform pre-build transforms from Torchvision Transforms to the dataset. .

def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + data_sample[1])
import torch 
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(0)
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image
import pandas as pd
import os

# Read CSV file from the URL and print out the first five samples
# directory=""
# csv_file ='index.csv'
# csv_path=os.path.join("/Neural Network AND PYTORCH/index.csv")
# data_name = pd.read_csv("index.csv")
# data_name.head()
# Get the value on location row 0, column 1 (Notice that index starts at 0)
#rember this dataset has only 100 samples to make the download faster  
print('File name:', data_name.iloc[0, 1])
# Get the value on location row 0, column 0 (Notice that index starts at 0.)
print('y:', data_name.iloc[0, 0])
# Print out the file name and the class number of the element on row 1 (the second row)
print('File name:', data_name.iloc[1, 1])
print('class or y:', data_name.iloc[1, 0])
# Print out the total number of rows in traing dataset
print('The number of rows: ', data_name.shape[0])
# Combine the directory path with file name
image_name =data_name.iloc[1, 1]
image_name
image_path=os.path.join(directory,image_name)
image_path
# Plot the second training image
image = Image.open(image_path)
plt.imshow(image,cmap='gray', vmin=0, vmax=255)
plt.title(data_name.iloc[1, 0])
plt.show()
# Plot the 20th image
image_name = data_name.iloc[19, 1]
image_path=os.path.join(directory,image_name)
image = Image.open(image_path)
plt.imshow(image,cmap='gray', vmin=0, vmax=255)
plt.title(data_name.iloc[19, 0])
plt.show()