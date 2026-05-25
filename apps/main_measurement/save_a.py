import HopTrack as ht
import time
import pickle
##############<< start plot setting parameters >> ################

t0 = 0 # start frame
t1 = 0 # end frame
filename = 'test.dat' # data file
radiifile = 'N0128_0805_AVE.dat0'
###############<< end >>################

starttime = time.time()
# a.read_dcd(128, 64, 64)
a=ht.Particles(filename)
a.t1 = t1
a.read_isobe(4096*4, radiifile=radiifile, Lx=128, Ly=128, mode=1, t1=t1)
a.normalize_radius()
a.pbc()
with open("a_AVE_test.obj", "wb") as f:
    pickle.dump(a, f)
endtime = time.time()
print(f'Saving data costs {endtime - starttime} seconds')
