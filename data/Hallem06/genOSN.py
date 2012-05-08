import numpy as np
import sys
import csv 
if __name__=="__main__":
    file = sys.argv[1]
    reader = csv.reader(open(file,'rb'),delimiter='\t')
    writer = csv.writer(open('Hallem_OSN','w'))
    OSN = reader.next()[1:]
    tau = np.array(reader.next()[1:],dtype=np.float64)
    Vr  = np.array(reader.next()[1:],dtype=np.float64)
    Vt  = np.array(reader.next()[1:],dtype=np.float64)

    writer.writerow(['Dt 1e-5'])
    writer.writerow(['Duration 1.0'])
    writer.writerow(['Neuron '+repr(len(OSN))])

    for i in xrange(len(OSN)):
        writer.writerow([OSN[i]+' -0.05 '+repr(Vr[i])+' '\
                   +repr(Vt[i])+' '+repr(tau[i])+' 1.0'])
