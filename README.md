![logo_corhix_white_border](https://github.com/user-attachments/assets/8f4efd41-14b9-4bf7-8c9c-5df17403aa5a)

CORHI-X is a versatile tool for exploring spacecraft constellations, visualizing instrument fields of view, checking data availability for coronagraphs and heliospheric imagers (e.g., the FoV of the instruments is plotted only if data is available in the online data archives of the instrument), and propagating CME data. Whether you're working with cataloged or custom CMEs, CORHI-X simplifies your heliophysics research. CORHIX has also an online version that you could try here: https://corhix.streamlit.app/

______________________________________________________________
LOCAL INSTALLATION INSTRUCTIONS
______________________________________________________________
Due to resources limitation given by streamlit, the users may want to download the local version in their personal machine and run the code in a straightforward way.
You can install CORHI-X locally by following these steps:

Ensure Anaconda is installed on your system (https://www.anaconda.com/download).
Open a terminal and execute the following commands:

    cd <directory_to_save_repository>
    git clone https://github.com/gretacappello/CORHI_X  
When git is not installed in you machine you can either download the entire package manually from Github or download "git" it using https://github.com/git-guides/install-git and then clone the repo. Once the repo is clonated or downloaded in the directory, execute the following commands:

    cd CORHI_X
    conda env create -f corhix_v1.yml

Wait approximately 5 minutes for the environment setup.
Activate the environment and start the app:

    conda activate corhix_v1
    streamlit run corhi_explorer_github.py &

Note: The first run may take a few minutes to download large files (e.g., DONKI catalog, HelCats catalog, and observation dates). Regular updates are performed and the code download the latest files of catalogues and instrument observation dates. If you have problem installing locally, consult the Issue page (https://github.com/gretacappello/CORHI_X/issues) or contact me (at the bottom of the page you find my contact informations).

You can find a tutorial on how to use CORHI-X here: https://drive.google.com/file/d/1wTsF5r3o5HDXrzXEAK1e7-EzfPVXgYN1/view?usp=sharing
______________________________________________________________
RUNNING CORHI-X
______________________________________________________________
To run CORHI-X after installation, navigate to the directory containing corhi_explorer_github.py: 
        
    cd <directory_to_file_corhi_explorer_github.py>
    
Activate the environment and start the app: 

    conda activate corhix_v1 
    streamlit run corhi_explorer_github.py &
    
Note that an active internet connection is required to fetch ephemeris data from NASA JPL Horizons (https://ssd.jpl.nasa.gov/horizons/app.html#/).
______________________________________________________________
CONTACT AND REFERENCING
______________________________________________________________
You can find here the Cappello et al. 2025 paper about CORHI-X pubblished in Frontiers (https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2025.1571024/full?utm_source=email-sig&utm_medium=email&utm_content=100_VIEWS&utm_campaign=imp_mile_2024_fall_en_aut-ww)

For any question or request please contact me at: greta.cappello@uni-graz.at 

Please cite the Cappello et al. 2025 paper and the code when used for science pubblication.


The version v1.0 of CORHI-X is archived on Zenodo at DOI 10.5281/zenodo.14800582 (https://zenodo.org/records/14803010)


If you find some problems installing the tool you can have a look at https://github.com/gretacappello/CORHI_X/issues or you can drop me an email!
