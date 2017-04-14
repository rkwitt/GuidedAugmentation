import os
import sys

for line in sys.stdin:
	print os.path.splitext(line)[0]
