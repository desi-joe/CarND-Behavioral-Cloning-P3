import numpy as np
from moviepy.editor import ImageSequenceClip
import argparse
import cv2
import csv
from sklearn.model_selection import train_test_split
import sklearn

csv_file_name = 'driving_log.csv'

def process_image_folder(dir):



    csv_file = dir + csv_file_name
    dir = dir + "IMG/"

    print('procesing [' + csv_file +'] and image dir[' + dir + ']')

    correction = 0.2

    img_list = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            steering_center = float(row[3])
            # create adjusted steering measurements for the side camera images
            correction = 0.2 # this is a parameter to tune
            steering_left 	= steering_center + correction
            steering_right 	= steering_center - correction

            # read in images from center, left and right cameras
             # fill in the path to your training IMG directory
            img_center 	= dir + row[0].split('/')[-1]
            img_left 	= dir + row[1].split('/')[-1]
            img_right 	= dir + row[2].split('/')[-1]

            img_list.append((img_left, steering_left))
            img_list.append((img_right, steering_right))

            #for each image, create a flip image
            img_list.append(flip_img(dir, row[1].split('/')[-1], steering_left))
            img_list.append(flip_img(dir, row[2].split('/')[-1], steering_right))

            if(steering_center == 0):
            	# if the center is 0, transform the img to left and rigth prospective
            	img_list.append(left_prospective (dir, row[0].split('/')[-1], steering_center))
            	img_list.append(right_prospective(dir, row[0].split('/')[-1], steering_center))
            else:
            	img_list.append((img_center, steering_center))
            	img_list.append(flip_img(dir, row[0].split('/')[-1], steering_center))

    return img_list

def flip_img(dir, image, streeing_angle):

	fliped_img = dir + 'fliped_' + image

	print('>>>' + fliped_img)
	img = cv2.imread(dir + image)


	image_flipped = np.fliplr(img)
	cv2.imwrite(fliped_img, image_flipped)

	return fliped_img, -streeing_angle
 
def left_prospective(dir, image, streeing_angle):
	
	lf_img = dir + 'lp_' + image
	img = cv2.imread(dir + image)

	rows,cols,_ = img.shape
	off = 50
	pts1 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
	pts2 = np.float32([[0,0 - off],[cols,0 ],[0,rows + off],[cols,rows]])

	M = cv2.getPerspectiveTransform(pts1,pts2)
	cv2.imwrite(lf_img, cv2.warpPerspective(img, M,(cols,rows)))

	return lf_img, streeing_angle + 2.0

def right_prospective(dir, image, streeing_angle):
	rp_img = dir + 'rp_' + image

	img = cv2.imread(dir + image)
	rows,cols,_ = img.shape
	
	off = 50
	pts1 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
	pts2 = np.float32([[0,0],[cols,0 - off],[0,rows],[cols,rows + off]])

	M = cv2.getPerspectiveTransform(pts1,pts2)
	cv2.imwrite(rp_img, cv2.warpPerspective(img, M,(cols,rows)))

	return rp_img, streeing_angle - 2.0

def write_master_list(filename, lst):
	print('writing [xxx] records in [%s]' % filename)
	file = open(filename,'a') 
	for x in range(0, len(lst)):
		#ln = "%s,%f\n" % lst[x]
		file.write("%s,%f\n" % lst[x]) 
	file.close() 


def read_master_list(filename):
	print('reading records from [%s]' % filename)
	lst = []
	with open(filename) as f:
		for line in f:
			lst.append((line.split(',')[0], line.split(',')[1]))
	return lst


def generator(samples, batch_size=32):
	n_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		samples = sklearn.utils.shuffle(samples)
		for offset in range(0, n_samples, batch_size):
			batch_samples = samples[offset:offset + batch_size]

			images = []
			angles = []
			for imagePath, measurement in batch_samples:
				
				originalImage = cv2.imread(imagePath)
				images.append(cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB))
				angles.append(measurement)

				# trim image to only see section with road
				inputs = np.array(images)
				outputs = np.array(angles)
				yield sklearn.utils.shuffle(inputs, outputs)







