import os
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
from utils.preprocess import *
from utils.generate import *
from math import ceil

## Get parser arguments
#parser = argparse.ArgumentParser()
#parser.add_argument("-i", "--image_path", required=True, type=str, help="path of image you want to quilt")
#parser.add_argument("-b", "--block_size", type=int, default=20, help="block size in pixels")
#parser.add_argument("-o", "--overlap", type=int, default=1.0/6, help="overlap size in pixels (defaults to 1/6th of block size)")
#parser.add_argument("-s", "--scale", type=float, default=4, help="Scaling w.r.t. to image size")
#parser.add_argument("-n", "--num_outputs", type=int, default=1, help="number of output textures required")
#parser.add_argument("-f", "--output_file", type=str, default="output.png", help="output file name")
#parser.add_argument("-p", "--plot", type=int, default=1, help="Show plots")
#parser.add_argument("-t", "--tolerance", type=float, default=0.1, help="Tolerance fraction")

#args = parser.parse_args()
def Texture_Generator(input, output, num_outputs):
	#if __name__ == "__main__":
	# Start the main loop here
	path = input
	block_size = 20
	scale = 4
	overlap = 1.0/6
	#print("Using plot {}".format(0))
	# Set overlap to 1/6th of block size
	if overlap > 0:
		overlap = int(block_size*1.0/6)
	else:
		overlap = int(block_size/6.0)

	# Get all blocks
	image = cv2.imread(path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0
	#print("Image size: ({}, {})".format(*image.shape[:2]))

	H, W = image.shape[:2]
	outH, outW = int(scale*H), int(scale*W)

	for i in range(num_outputs):
		textureMap = generateTextureMap(image, block_size, overlap, outH, outW, 0.1)

		# Save
		textureMap = (255*textureMap).astype(np.uint8)
		textureMap = cv2.cvtColor(textureMap, cv2.COLOR_RGB2BGR)
		if num_outputs == 1:
			cv2.imwrite(output, textureMap)
			#print("Texture ",i, 'of ', str(num_outputs), 'generated successfully!')
		else:
			cv2.imwrite(output.replace(".", "_{}.".format(i)), textureMap)
			#print("Texture ",i, 'of ', str(num_outputs), 'generated successfully!')
