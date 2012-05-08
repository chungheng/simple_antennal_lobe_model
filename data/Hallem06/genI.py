import numpy as np
import sys
import csv 
if __name__=='__main__':
    odor = sys.argv[1]
    reader = csv.reader(open(odor,'rb'),delimiter='\t')
    writer = csv.writer(open('Hallem_I_'+odor,'w'))
    OSN = reader.next()[1:]
    I   = np.array(reader.next()[1:],dtype=np.float64)

for i in xrange(len(OSN)):
    writer.writerow([OSN[i]+' 0.0 1.0 '+repr(I[i])])
