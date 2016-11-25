import os
import sys
import glob


file_list = glob.glob(sys.argv[1])

for file in file_list:

	parts = file.split('_')

	num = int(parts[1])

	if num >= int(sys.argv[2]) and num <= int(sys.argv[3]):
		
		print file
