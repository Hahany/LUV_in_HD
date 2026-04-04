# Code and Data for **"Strong Correlation Between Local Free Volume and Stringlike Motions in a Hard-Disk Glass Former"**
## License
This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
This repository contains all code, raw data, and generated figures for the two systems studied in our manuscript#
1. **2D Hard Disk System**
2. **3D Metallic Glass System**

Each system has its own dedicated subdirectory with a self-contained workflow.

## 📁 Repository Structure
```
.
├── apps                            # All code, data and figure for the 2D HD system
│   ├── DvHT_vs_Rc                  # Local unoccupied volume (LUV) of the head and tail of the string with different head-tail seperation 
│   ├── Dv_vs_Rc                    # Size estimation of quasivoid
│   ├── hist_dr                     # Histogram of displacement
│   ├── hop                         # LUV of hop particles with different time intervals
│   ├── LUV_distribution            # LUV distributions of the tail of the string, hop and all particles
│   ├── LUV_vs_time                 # Temporal evolution of LUV at the head, middle and tail of the string
│   ├── main_measurement            # Main program to measure LUV of the head and tail of the string, hop and all particles
│   ├── rdf                         # Radial distribution function of the system
│   ├── SISF                        # Self-interaction scattering function
│   ├── time_reverse_symmetry       # Time reverse symmetry of LUV at head and tail of the string
│   └── traj                        # Particle trajectories of the system
├── data                            # All raw data for the 2D HD system
├── scripts                         # All scripts for the 2D HD system 
├── readme.md                       # The current file
├── LICENSE
└── setup.py                        # Setup file for the 2D HD system, to install the 2D HD system
```

## ▶️ How to Reproduce Figures

To generate the figures in this repository:

1. **Install the analysis package for each system** (in editable mode):
   ```bash
   # For the 2D Hard Disk system
   cd LUV_in_HD
   pip install -e .
   ```
   Other dependencies can be installed using `pip` or `conda`.  

2. **Run the analysis scripts in the corresponding `apps/` subdirectories**:
    example:
    ```bash
        cd apps/SISF
        python plot_sisf.py
    ```
