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

For a tutorial on how to use CORHI-X, visit: CORHI-X Tutorial.
