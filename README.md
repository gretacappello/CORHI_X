![logo_corhix_white_border](https://github.com/user-attachments/assets/8f4efd41-14b9-4bf7-8c9c-5df17403aa5a)

CORHI-X is a versatile tool for exploring spacecraft constellations, visualizing instrument fields of view, checking data availability for coronagraphs and heliospheric imagers, and propagating CME data. Whether you're working with cataloged or custom CMEs, CORHI-X simplifies your heliophysics research.
Try the online version here: https://corhix.streamlit.app/

Local Installation Instructions

If you encounter user limitations with the online version, install CORHI-X locally by following these steps:

Ensure Anaconda is installed on your system.
Open a terminal and execute the following commands:

    cd <directory_to_save_repository>
    git clone https://github.com/gretacappello/CORHI_X
    conda env create -f corhix_v1.yml

Wait approximately 5 minutes for the environment setup.
Activate the environment and start the app:

    conda activate corhix_v1
    streamlit run corhi_explorer_github.py &

Note: The first run may take a few minutes to download large files (e.g., DONKI catalog, HelCats catalog, and observation dates). Regular updates are reflected at the bottom of the app.

You can find a tutorial on how to use CORHI-X here: https://drive.google.com/file/d/1wTsF5r3o5HDXrzXEAK1e7-EzfPVXgYN1/view?usp=sharing

Running CORHI-X: To run CORHI-X after installation:

Navigate to the directory containing corhi_explorer_github.py: 
        
    cd <directory_to_file_corhi_explorer_github.py>
    
Activate the environment and start the app: 

    conda activate corhix_v1 
    streamlit run corhi_explorer_github.py &
    
Note that an active internet connection is required to fetch ephemeris data from NASA JPL Horizons (https://ssd.jpl.nasa.gov/horizons/app.html#/).
Contact and Referencing: For any question or request please contact me at: greta.cappello@uni-graz.at 
A paper of CORHI-X is currently under development. If you use CORHI-X for pubblications, please 
