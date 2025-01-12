
![logo_corhix_white_border](https://github.com/user-attachments/assets/8f4efd41-14b9-4bf7-8c9c-5df17403aa5a)

CORHI-X is a tool enables you to explore spacecraft constellations, visualize instrument fields of view, check data availability for coronagraphs and heliospheric imagers, and propagate either a catalog of CMEs or your own CME data. We hope this tool helps you explore heliospheric events with ease!

CORHI-X is available in the online version at: https://corhix.streamlit.app/

INSTALLATION IN THE LOCAL MACHINE:

Due to limitations on the simultaneous users number. Here are also listed the steps to perform in order to install locally CORHI-X. If you do not have Anaconda, please install it following the steps at https://anaconda.org/

Then, open a terminal window and type:
1) cd <directory_to_save_repository>
2) git clone https://github.com/gretacappello/CORHI_X
4) conda env create -f corhix_v1.yml
5) Wait about 5 mins for the environment to set up
6) conda activate corhix_v1
7) streamlit run corhi_explorer_github.py &
8) You may need to wait a few minutes at the first run, since the code download some big files from a Drive (such as DONKI catalog, HelCats catalog, observation dates for each instrument). These files are update regularly and at the bottom of the app is always reported the last update made.
10) Enjoy the app! You can find a tutorial on how to use it here: https://drive.google.com/file/d/1wTsF5r3o5HDXrzXEAK1e7-EzfPVXgYN1/view?usp=sharing

RUN CORHIX:
To run CORHIX, you must have installed it before following the steps you find above.
Then to run the app you will just need to follow the following simple steps in a terminal window:
1) cd <directory_to_file_corhi_explorer_github.py> 
2) conda activate corhix_v1
3) streamlit run corhi_explorer_github.py &
4) Enjoy :)

Please note that the app requires internet to run since it download the ephemeridis coordinates from NASA JPL Horizon (https://ssd.jpl.nasa.gov/horizons/app.html#/)

For any question or request please contact me at: greta.cappello@uni-graz.at

A paper of CORHI-X is currently under development. If you use CORHI-X for pubblications, please contact me for referencing purposes.

