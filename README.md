CORHI-X: Explore Spacecraft Constellations and Heliophysics Events with Ease

CORHI-X is a versatile tool for exploring spacecraft constellations, visualizing instrument fields of view, checking data availability for coronagraphs and heliospheric imagers, and propagating CME data. Whether you're working with cataloged or custom CMEs, CORHI-X simplifies your heliophysics research.

Try the online version here: CORHI-X Online
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

For a tutorial on using CORHI-X, visit: CORHI-X Tutorial.
Running CORHI-X

To run CORHI-X after installation:

    Navigate to the directory containing corhi_explorer_github.py:

cd <directory_to_file_corhi_explorer_github.py>

Activate the environment and start the app:

    conda activate corhix_v1
    streamlit run corhi_explorer_github.py &

An active internet connection is required to fetch ephemeris data from NASA JPL Horizons.
Contact and Referencing

For inquiries, feature requests, or referencing CORHI-X in publications, please contact me at: greta.cappello@uni-graz.at

A paper on CORHI-X is currently under development. If you use CORHI-X for publications, please get in touch for referencing purposes.

