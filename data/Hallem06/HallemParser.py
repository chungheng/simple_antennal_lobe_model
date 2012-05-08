import numpy as np
import csv

fileReader = csv.reader(open('Hallem.tsv','rb'), delimiter='\t')
OSN_list = fileReader.next()
spon_fire_rate = np.array(fileReader.next()[1:],dtype=(int)
odorant_list = [[] * 110];


for i in xrange(110):
    line = fileReader.next()
    odorant_list[i] = line[0]
    
  

