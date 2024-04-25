import sys
sys.path.append('..')

import numpy as np
import os
import pandas
import openpyxl


img_dir = '/home/axel/dev/neonatal_brain_segmentation/data/images/case/raw'
subject_name = []

for path_patient in os.listdir(img_dir):
	if(path_patient[0] != '.'):
		subject_name.append(path_patient[:-7])


col = np.array(subject_name).reshape(len(subject_name), 1)


outputFile = pandas.DataFrame(col, columns=['subject_name'])
outputFile.to_excel("/home/axel/dev/neonatal_brain_segmentation/data/subject_case_name.xlsx")
