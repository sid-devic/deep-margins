import os
import shutil

dir = '/home/sid/Downloads/cifar/'
cat_train_save = '/home/sid/Downloads/cifar/binary_train/cat/'
cat_test_save = '/home/sid/Downloads/cifar/binary_test/cat/'

dog_train_save = '/home/sid/Downloads/cifar/binary_train/dog/'
dog_test_save = '/home/sid/Downloads/cifar/binary_test/dog/'

for file in os.listdir(dir + 'train'):
	if 'cat' in file:
		old_path = dir + 'train/' + file
		new_path = cat_train_save + file
		shutil.copyfile(old_path, new_path)
	if 'dog' in file:
		old_path = dir + 'train/' + file
		new_path = dog_train_save + file
		shutil.copyfile(old_path, new_path)

for file in os.listdir(dir + 'test'):
	if 'cat' in file:
		old_path = dir + 'test/' + file
		new_path = cat_test_save + file
		shutil.copyfile(old_path, new_path)
	if 'dog' in file:
		old_path = dir + 'test/' + file
		new_path = dog_test_save + file
		shutil.copyfile(old_path, new_path)	
