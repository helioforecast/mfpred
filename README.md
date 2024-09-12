# mfpred

Solar wind ICME flux rope prediction with machine learning building up on the code of Reiss et al. 2021, for real time deployment.

#### Authors: 
M.A. Reiss (1), C. Möstl (2), R.L. Bailey (3), U. Amerstorfer (2), Emma Davies (2), Eva Weiler (2)

(1) NASA CCMC, 
(2) Austrian Space Weather Office, GeoSphere Austria
(3) Conrad Observatory, GeoSphere Austria


### Usage

The notebooks

    mfrpred_real_btot.ipynb
    
    mfrpred_real_btot.ipynb

are used for training the machine learning algorithms.

The notebook

    mfrpred_deploy.ipynb

is used for running the application of the trained ML model to real time data, and for producing general plots on ICME Bz based on the ICMECAT catalog. This code is in development (status September 2024).

We develop the code in the notebooks, and each notebook is converted in its first cell to a .py script of the same name. These .py scripts are then called by cronjobs for real-time deployment.


### Solar wind data:
Copy Version 10 from https://figshare.com/articles/dataset/Solar_wind_in_situ_data_suitable_for_machine_learning_python_numpy_arrays_STEREO-A_B_Wind_Parker_Solar_Probe_Ulysses_Venus_Express_MESSENGER/12058065
into the folder /data.


### Installation 

Install python with miniconda:

on Linux:

	  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	  bash Miniconda3-latest-Linux-x86_64.sh

on MacOS:

	  curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
	  bash Miniconda3-latest-MacOSX-x86_64.sh

go to a directory of your choice

	  git clone https://github.com/helioforecast/mfpred
	  

Create a conda environment using the "environment.yml", and activate the environment:

	  conda env create -f environment.yml    

	  conda activate mfpred

