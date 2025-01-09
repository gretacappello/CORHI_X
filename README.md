# CORHI_X
CORHI-X is a tool enables you to explore spacecraft constellations, visualize instrument fields of view, check data availability for coronagraphs and heliospheric imagers, and propagate either a catalog of CMEs or your own CME data. We hope this tool helps you explore heliospheric events with ease!

CORHI-X is available in the online version at: https://corhix.streamlit.app/

Due to limitations on the simultaneous users number. Here are also listed the steps to perform in order to install locally CORHI-X:

1) If you do not have Anaconda, please install it following the steps at https://anaconda.org/
2) Download the the folder in this repository, including the file corhix_v1.yml, and save it in the directory you prefer.
3) Open a terminal and go to the directory which contain the file corhix_v1.yml
4) Write in the terminal: conda env create -f corhix_v1.yml
5) Wait about 5 mins for the environment to set up
6) Write in the terminal: conda activate corhix_v1
7) Write in the terminal: streamlit run corhi_explorer_github.py &
8) Enjoy the app! You can find a tutorial on how to use it here: https://drive.google.com/file/d/1wTsF5r3o5HDXrzXEAK1e7-EzfPVXgYN1/view?usp=sharing

Update Jan. 9, 2025) At the moment the app is not working due to the spacecraft coordinate request through the JPL Horizon website that does not work due to Eaton Fire in Los Angeles.


For any question or request please contact me at: greta.cappello@uni-graz.at
