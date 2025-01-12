
# CORHI_X
CORHI-X is a tool enables you to explore spacecraft constellations, visualize instrument fields of view, check data availability for coronagraphs and heliospheric imagers, and propagate either a catalog of CMEs or your own CME data. We hope this tool helps you explore heliospheric events with ease!

CORHI-X is available in the online version at: https://corhix.streamlit.app/

INSTALLATION IN THE LOCAL MACHINE:

Due to limitations on the simultaneous users number. Here are also listed the steps to perform in order to install locally CORHI-X:
1) If you do not have Anaconda, please install it following the steps at https://anaconda.org/
2) Download the the folder in this repository, including the file corhix_v1.yml, and save it in the directory you prefer. You can dowload it manually or you can do this directly in the terminal. Open a terminal 1) Go in the directory you wanna save the files (e.g., cd Users/GretaC/corhix/) 2) Write in the terminal: git clone https://github.com/gretacappello/CORHI_X
3) Open a terminal and go to the directory which contain the file corhix_v1.yml
4) Write in the terminal: conda env create -f corhix_v1.yml
5) Wait about 5 mins for the environment to set up
6) Write in the terminal: conda activate corhix_v1
7) Write in the terminal: streamlit run corhi_explorer_github.py &
8) You may need to wait a few minutes at the first run, since the code download some big files from a Drive (such as DONKI catalog, HelCats catalog, observation dates for each instrument). These files are update regularly and at the bottom of the app is always reported the last update made.
10) Enjoy the app! You can find a tutorial on how to use it here: https://drive.google.com/file/d/1wTsF5r3o5HDXrzXEAK1e7-EzfPVXgYN1/view?usp=sharing

Please note that the app requires internet to run since it download the ephemeridis coordinates from NASA JPL Horizon (https://ssd.jpl.nasa.gov/horizons/app.html#/)

For any question or request please contact me at: greta.cappello@uni-graz.at

A paper of CORHI-X is currently under development. If you use CORHI-X for pubblications, please contact me for referencing purposes.


RUN CORHIX:
To run CORHIX, you must have installed it before following the steps you find above.
Then to run the app you will just need to follow the following simple steps in a terminal window:
1) cd <directory_to_file> corhi_explorer_github.py
2) conda activate corhix_v1
3) streamlit run corhi_explorer_github.py &
4) Enjoy :)
