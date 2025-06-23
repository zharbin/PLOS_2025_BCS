# PLOS_2025_BCS
Breast Cavity Healing Following BCS Finite Element Code

Publication:
#### Computational modeling of patient-specific healing and deformation outcomes following breast-conserving surgery based on MRI data
#### Zachary Harbin, Carla Fisher, Sherry Voytik-Harbin, Adrian Buganza Tepole

The code has been designed to run on Purdue Research Computing Clusters (i.e., Negishi Community Cluster). It may not work as-is in your system. Here are a few tips to troubleshoot compiling:
<ul>
<li>The code requires EIGEN linear algebra libraries, as well as BOOST libraries. It currently runs with BOOST 1.80 and EIGEN 4.1.4</li>
<li>The code has a parallel version with OPEN MP. The parallel parts of the code are primarily in solver.cpp and can be easily commented out if desired</li>
<li>Compilation is done with CMAKE</li>
</ul>

To compile the code:
<ul>
<li>Make sure EIGEN and BOOST files are downloaded in the folder in which the folder PLOS_2025_BCS is located</li>
<li>Open terminal. Make sure CMAKE is installed, as well as C++ compilers. We use intel compilers (19.1.3.304) with support for OPEN MP.</li>
<li>Go to the folder where the code will be compiled (i.e., cmake-build-release)</li>
<li>Run CMAKE in the terminal to create the makefile (i.e., cmake ../   make/)</li>
<li>The code can then be executed through the following command: sbatch surgerywoundcpp3D.sub</li>
</ul>
