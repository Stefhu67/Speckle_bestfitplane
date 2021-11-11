from Module import *
import os
import glob
import pandas as pd

data_csv = glob.glob('./Data/*.csv')
data_csv.sort()
coef_frame = pd.DataFrame()

i=0
for csv in data_csv:
    normal_vector = bestfitplane(csv).findbestfitplane()
    coef = [normal_vector[0], normal_vector[1], normal_vector[2]]
    #import pdb; pdb.set_trace()
    coef_frame[i] = coef
    i = i+1

print(coef_frame)