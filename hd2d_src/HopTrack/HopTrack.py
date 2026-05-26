# This is core of the OOP version for particle trajectory plotting.
# __init__ parameters:
# <filename> the file name of trajectory data
# <lx> box size along x-axis
# <ly> box size alone y-axis
# <n> particle number in total
# include functions:
# class Particles
# 1, readlmp(): reading data from lammps output file with format : <id> <type> <x> <y> <z>
# 2, read_npy(): read numpy format data
# 3, read_dcd(ntype, lx, ly)  read .dcd file
# 4, setduration(begin, finish)   set duration of total frame
# 5, unwrap(): unwrap displacement unwrap the particle position based on the final frame to avoid the particle displace bigger than 1/2 of the box size
# unwrap2: unwrap the particle position based on the first frame to avoid the particle displace bigger than 1/2 of the box size. And will update the original data
#unwrap3: the same as unwrap 2, but won't update the original data
# 6, pbc(): periodic boundary condition
# 7, coarsening(<nseg>): do time coarsening by parameter <nseg>(the first and last frame do not)
# 8, short_burst: The same average method with lammps
#         # example: short_burst(2, 3, 1000), value on times 1000, 998 and 996 to calculate the final average
# 9, chooseWithoutCoarsening(self, nchoose): choose raw data without any average. the same as short_burst(1, 1, N/nchoose), N is total frame number.
# 10, chbox(<x_low>, <x_up>, <y_low>, <y_up>):  change the visual region
# 11, hops(hopcut) , show displacement bigger than hopcut
# 12, readvoid( filename): read void coordinates, <x> <y>£¬ n lines, 2 columns for 2-D system
# 13, showdisp( ax, lw=0.5, ms=0.5, nodot=False, showvoid=False, showpid=False, showforcepid=0, showforcedid=0, colorbar=False, radii=None, showradii=False)
#            ax: the plotting object
#            lw: line width
#            nodot: don't show the red dot representing particle at inital position
#            showvoid: show the void
#            showpid: show particle index
#            showforceid: show force index
#            colorbar: show colorbar
#            showradii: show radii of particles
# 13, findstring(self, HopThreshold, ConnectThreshold, microstring=True, ignoreLoop=False, stlength = 0, dr_ht=4, hop_whole=False, silent=True):
#         # deep first search, output string and level
#         # hop_whole : check a particle is hopped or not at the total simulation time.
#           HopThreshold: when displacement bigger than this value, it will be considered as a hop .
#           ConnectThreshold: if two hopps are closed than this value, they will be considered as connected.
#           microstring: ignore the connection between two time intervals. True: ignore
#           stlength = 0: criteria of string length, only return longer than this one
#           dr_ht=4, criteria of distance between string head and tail
#           hop_whole=False, search the hop in the whole time duration
#           silent=True, print info or not
# 14, showstring(self, ax, lw=0.8, nodot=False, findquasivoid=Ture, size=5, showid=False,
#                    select=False, show_length_distribusion=False,
#                    stringID=None, SSC=None, WL=False, show_localdensity_region=False, Rc = 0, mode=0):
#

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
from hd2d_src.HopTrack.core import *
from hd2d_src.HopTrack.core.share import *
from hd2d_src.HopTrack.data import *
from hd2d_src.HopTrack.viz import *







class Particles:
    def __init__(self, filename):
        self.fstart = 0
        self.frames = []
        self.frames0 = []
        self.filename = filename
        self.n = None  # particle number
        self.n0 = None  # store initial particle number before change visualized box
        self.skiprows = None
        self.tmax = None
        self.x_low = 0
        self.y_low = 0
        self.x_up = None
        self.y_up = None
        self.z_up = None
        self.z_low = None
        self.L = None  # box size along x-axis and y-axis
        self.t1 = 0
        self.radii = []

    def readlmp(self):
        return readlmp(self)

    def read_npy(self):
        return read_npy(self)

    def read_isobe(self, N, radiifile=None, Lx=64, Ly=64, mode=0, t1=0):
        return read_isobe(self, N=N, radiifile=radiifile, Lx=Lx, Ly=Ly, mode=mode, t1=t1)

    def read_dcd(self, ntype, lx, ly):
        return read_dcd(self, ntype=ntype, lx=lx, ly=ly)

    def readvoid(self, voidlocation):
        return readvoid(self, voidlocation=voidlocation)

    def setduration(self, begin=0, finish=0):
        return setduration(self, begin=begin, finish=finish)

    def unwrap(self):
        return unwrap(self)

    def unwrap2(self):
        return unwrap2(self)

    def unwrap3(self):
        return unwrap3(self)
    
    def unwrap_vec(self):
        return unwrap_vec(self)

    def pbc(self, noto0=True):
        return pbc(self, noto0=noto0)

    def coarsening(self, nseg, NoCoarsenOnHeadandTail=True):
        return coarsening(self, nseg=nseg, NoCoarsenOnHeadandTail=NoCoarsenOnHeadandTail)

    def short_burst(self, N_every, N_rep, N_freq, noaveonhead):
        return short_burst(self, N_every=N_every, N_rep=N_rep, N_freq=N_freq, noaveonhead=noaveonhead)

    def chooseWithoutCoarsening(self, nchoose, shift=0):
        return chooseWithoutCoarsening(self, nchoose=nchoose, shift=shift)

    def chbox(self, x_low, x_up, y_low, y_up, update_bak=False):
        return chbox(self, x_low=x_low, x_up=x_up, y_low=y_low, y_up=y_up, update_bak=update_bak)

    def shiftbox(self, dx, dy):
        return shiftbox(self, dx=dx, dy=dy)

    def normalize_radius(self):
        return normalize_radius(self)

    def hops(self, hopcut, hop_whole=False):
        return hops(self, hopcut=hopcut, hop_whole=hop_whole)

    def activeParticleInHops(self, pid):
        return activeParticleInHops(self, pid=pid)

    def showdisp(self, fig, ax, t_start, t_end, sid=0, lw=0.5, ms=0.5, nodot=False, showvoid=False,
                 showpid=False, showforcepid=0, showforcedid=0, colorbar=False,
                 radii=None, showradii=False, overlap=True, blacktraj=False, crange=[0, 100]):
        return showdisp(self, fig=fig, ax=ax, t_start=t_start, t_end=t_end, sid=sid, lw=lw, ms=ms, nodot=nodot, showvoid=showvoid,
                                showpid=showpid, showforcepid=showforcepid,
                                showforcedid=showforcedid, colorbar=colorbar,
                        radii=radii, showradii=showradii,
                        overlap=overlap, blacktraj=blacktraj, crange=crange)

    def findstring(self, HopThreshold, ConnectThreshold, microstring=True, ignoreLoop=False,
                   stlength=0, dr_ht=4, hop_whole=False, silent=True):
        return findstring(self, HopThreshold=HopThreshold, ConnectThreshold=ConnectThreshold, microstring=microstring,
                                  ignoreLoop=ignoreLoop,
                                  stlength=stlength, dr_ht=dr_ht, hop_whole=hop_whole, silent=silent)

    def showstring(self, ax, lw=0.8, nodot=False, findquasivoid=False, size=5, showid=False,
                   select=False, sid=0, show_length_distribusion=False, showtraj=True,
                   stringID=None, SSC=None, WL=False, show_localdensity_region=False, Rc=4, mode=0, reverse=False,
                   colors= ['m', 'y', 'g', 'b'], showvec=False):
        return showstring(self, ax=ax, lw=lw, nodot=nodot, findquasivoid=findquasivoid, size=size, showid=showid,
                                  select=select, sid=sid, show_length_distribusion=show_length_distribusion, showtraj=showtraj,
                                  stringID=stringID, SSC=SSC, WL=WL, show_localdensity_region=show_localdensity_region,
                          Rc=Rc, mode=mode, reverse=reverse, colors=colors, showvec=showvec)

    def findloop(self, HopThreshold, ConnectThreshold, microstring=True, dr_ht=4, hop_whole=False, silent=True):
        return findloop(self,HopThreshold=HopThreshold, ConnectThreshold=ConnectThreshold,microstring=microstring,
                        dr_ht=dr_ht, hop_whole=hop_whole,silent=silent)

    def showdensity(self, regionradius, r_insert, rho, savefile=None, figshow=False,
                    pid=None, select=False, mode=0, Length_of_string=3, crt_in=False, choose_middle=False, dr_ht=5):
        return showdensity(self, regionradius=regionradius, r_insert=r_insert, rho=rho,
                                   savefile=savefile, figshow=figshow,
                                   pid=pid, select=select, mode=mode, Length_of_string=Length_of_string, crt_in=crt_in,
                                   choose_middle=choose_middle, dr_ht=dr_ht)

    def luv_fix_cr(self, regionradius, rho, savefile=None, figshow=False, dr_ht=5):
        return luv_fix_cr(self, regionradius, rho, savefile=savefile, figshow=figshow, dr_ht=dr_ht)

    def showdensity_no_overlap(self, regionradius, r_insert, rho, savefile=None, figshow=False,
                    pid=None, select=False, mode=0, Length_of_string=3, crt_in=False, choose_middle=False, dr_ht=5):
        return showdensity_no_overlap(self, regionradius=regionradius, r_insert=r_insert, rho=rho,
                                   savefile=savefile, figshow=figshow,
                                   pid=pid, select=select, mode=mode, Length_of_string=Length_of_string, crt_in=crt_in,
                                   choose_middle=choose_middle, dr_ht=dr_ht)
    
    def showdensity_no_overlap_tail_only(self, regionradius, r_insert, rho, savefile=None, figshow=False,
                pid=None, select=False, dr_ht=5):
        return showdensity_no_overlap_tail_only(self, regionradius=regionradius, r_insert=r_insert, rho=rho, savefile=savefile, figshow=figshow,
                pid=pid, select=select, dr_ht=dr_ht)    

    def shiftdensity(self, shifting_window, regionradius, r_insert, rho, savefile=None, figshow=False,
                    pid=None, select=False, dr_ht=5):
        return shiftdensity(self,  shifting_window=shifting_window, regionradius=regionradius, r_insert=r_insert, rho=rho,
                                   savefile=savefile, figshow=figshow, pid=pid, select=select, dr_ht=dr_ht)

    def showdensity_simu_diff(self, regionradius, dr_ht=5):
        return showdensity_simu_diff(self, regionradius, dr_ht)

    def cal_area_of_fragments(self, Rc, WL=False, entire_sys=False):
        return cal_area_of_fragments(self, Rc=Rc, WL=WL, entire_sys=entire_sys)

    def push_off_area(self, WL=False):
        return push_off_area(self, WL=WL)

    def MSD(self, fr1=0, interval=10, COM=False, frinc=1.4, realdt=1, spf=1):
        return MSD(self, fr1=fr1, interval=interval, COM=COM, frinc=frinc, realdt=realdt,
                           spf=spf)
