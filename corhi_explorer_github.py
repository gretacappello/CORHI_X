# cd /Users/gretacappello/Desktop/jupyter_notebooks/elevohi
# Activate GCS environment: conda activate elevohi
# Run the interface: streamlit run corhi_explorer.py &
#if files are not found, pay attention that they are not visible due to the icloud


import streamlit as st
import gdown
import pytz
import tempfile
#change path for ffmpeg for animation production if needed
#%matplotlib inline
#ffmpeg_path=''
from PIL import Image
import sys
import io
import os
import datetime
import glob
from datetime import datetime, timedelta
from sunpy.time import parse_time
import sunpy
import astropy.units as u
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import itertools
import multiprocessing as mp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#import spiceypy
import pandas as pd
from joblib import Parallel, delayed

import matplotlib.cm as cmap
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from scipy.signal import medfilt
import numpy as np
import pdb
import pickle
import seaborn as sns
#import sys
#import heliopy.data.spice as spicedata
#import heliopy.spice as spice
import astropy
import importlib    
import time
import numba
from numba import jit
import multiprocessing
import multiprocess as mp
import urllib
import copy
from astropy import constants as const
#import astrospice
#from datatime import datatime
import warnings
warnings.filterwarnings('ignore')
import os
import urllib.request
from astropy.io.votable import parse_single_table
from astropy.time import Time
import numpy as np

from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.net import Fido, attrs as a
from sunpy.coordinates import get_body_heliographic_stonyhurst, get_horizons_coord

from shapely.geometry import Polygon, GeometryCollection


from matplotlib.markers import MarkerStyle
import multiprocessing
from multiprocessing import Pool
import subprocess
from matplotlib.animation import FuncAnimation
from matplotlib import image as mpimg
#path_dates = '/Users/gretacappello/Desktop/PROJECT_2_METIS_TS/constellation_solohi_sterehi_wispr/dates_new_round_up/'
# 2) path with files containing higeocat_kinematics.p and donki_kinematics.p
#overview_path = '/Users/gretacappello/Desktop/jupyter_notebooks/elevohi/'
# 3) path to save pdf files:
#path_to_pdf = "/Users/gretacappello/Desktop/PROJECT_2_METIS_TS/constellation_solohi_sterehi_wispr/event_2021/" 
# 4) path logo
#path_to_logo = "/Users/gretacappello/Desktop/jupyter_notebooks/elevohi/"

path_local_greta = './'
#path_dates = path_local_greta+'date_github/dates_new_round_up/' #path_dates = path_local_greta+'/dates_new_round_up/'
# 2) path with files containing higeocat_kinematics.p and donki_kinematics.p
overview_path = path_local_greta
# 4) path logo
path_to_logo = path_local_greta


st.set_page_config(page_title="Cor-HI Explorer",page_icon=path_to_logo+"/logo_corhi.png", layout="wide")

# Display the logo at the top of the interfacex

# Add a title for the app below the logo


col1, col2 = st.columns([1, 2])
##url = "https://drive.google.com/file/d/1Ewl3l0t_LaggHb2n8jQyDOsU5fLQzyZy"
#output = "donki_kinematics_2019_now.p" 
#gdown.download(url, output, quiet=False)


# Function to add a new CME parameters input
def cme_kinematics(i):
    print(f"Calculating kinematics CME{i}..")
    t0_str = st.session_state.data[i]['t0']
    t0_num = mdates.date2num(datetime.strptime(t0_str, "%Y-%m-%d %H:%M:%S"))
    distance0 = 21.5*u.solRad.to(u.km)
    t00 = mdates.num2date(t0_num)
    
    gamma_init = 0.15
    ambient_wind_init = 450.
    kindays = 5
    n_ensemble = 50000
    halfwidth = np.deg2rad(st.session_state.data[i]['half angle'])
    res_in_min = 30
    f = 0.7

    #times for each event kinematic
    time1=[]
    tstart1=copy.deepcopy(t00)
    tend1=tstart1+timedelta(days=kindays)
    #make 30 min datetimes
    while tstart1 < tend1:

        time1.append(tstart1)  
        tstart1 += timedelta(minutes=res_in_min)    

    #make kinematics
    
    kindays_in_min = int(kindays*24*60/res_in_min)
    
    cme_lon=np.ones(kindays_in_min)*st.session_state.data[i]['longitude']
    cme_lat=np.ones(kindays_in_min)*st.session_state.data[i]['latitude']
    # Create the chararray to hold CME ID (as strings)
    cme_id = np.chararray(kindays_in_min, itemsize=27)
    # Assign string values from session state data, assuming it's purely for labeling
    cme_id[:] = np.array(st.session_state.data[i]['CME ID']).astype(str)


    cme_r_ensemble=np.zeros([n_ensemble,kindays_in_min])
    
    gamma = np.random.normal(gamma_init,0.025,n_ensemble)
    ambient_wind = np.random.normal(ambient_wind_init,50,n_ensemble)
    speed = np.random.normal(st.session_state.data[i]['speed'],50,n_ensemble)
    
    timesteps = np.arange(kindays_in_min)*res_in_min*60
    timesteps = np.vstack([timesteps]*n_ensemble)
    timesteps = np.transpose(timesteps)

    accsign = np.ones(n_ensemble)
    accsign[speed < ambient_wind] = -1.

    distance0_list = np.ones(n_ensemble)*distance0
    
    cme_r_ensemble=(accsign / (gamma * 1e-7)) * np.log(1 + (accsign * (gamma * 1e-7) * ((speed - ambient_wind) * timesteps))) + ambient_wind * timesteps + distance0_list
    
    cme_r=np.zeros([kindays_in_min,3])

    cme_mean = cme_r_ensemble.mean(1)
    cme_std = cme_r_ensemble.std(1)
    cme_r[:,0]= cme_mean*u.km.to(u.au)
    cme_r[:,1]=(cme_mean - 2*cme_std)*u.km.to(u.au) 
    cme_r[:,2]=(cme_mean + 2*cme_std)*u.km.to(u.au)

    #Ellipse parameters   
    theta = np.arctan(f**2*np.ones([kindays_in_min,3]) * np.tan(halfwidth*np.ones([kindays_in_min,3])))
    omega = np.sqrt(np.cos(theta)**2 * (f**2*np.ones([kindays_in_min,3]) - 1) + 1)   
    cme_b = cme_r * omega * np.sin(halfwidth*np.ones([kindays_in_min,3])) / (np.cos(halfwidth*np.ones([kindays_in_min,3]) - theta) + omega * np.sin(halfwidth*np.ones([kindays_in_min,3])))    
    cme_a = cme_b / f*np.ones([kindays_in_min,3])
    cme_c = cme_r - cme_b

            
    #### linear interpolate to 30 min resolution

    #find next full hour after t0
    format_str = '%Y-%m-%d %H'  
    t0r = datetime.strptime(datetime.strftime(t00, format_str), format_str) +timedelta(hours=1)
    time2=[]
    tstart2=copy.deepcopy(t0r)
    tend2=tstart2+timedelta(days=kindays)
    #make 30 min datetimes 
    while tstart2 < tend2:
        time2.append(tstart2)  
        tstart2 += timedelta(minutes=res_in_min)  

    time2_cme_user=parse_time(time2).plot_date        
    time1_cme_user=parse_time(time1).plot_date

    #linear interpolation to time_mat times    
    cme_user_r = [np.interp(time2_cme_user, time1_cme_user,cme_r[:,ii]) for ii in range(3)]
    cme_user_lat = np.interp(time2_cme_user, time1_cme_user,cme_lat)
    cme_user_lon = np.interp(time2_cme_user, time1_cme_user,cme_lon)
    cme_user_id = cme_id 
    cme_user_a = [np.interp(time2_cme_user, time1_cme_user,cme_a[:,ii]) for ii in range(3)]
    cme_user_b = [np.interp(time2_cme_user, time1_cme_user,cme_b[:,ii]) for ii in range(3)]
    cme_user_c = [np.interp(time2_cme_user, time1_cme_user,cme_c[:,ii]) for ii in range(3)]
    
    return time2_cme_user, cme_user_r, cme_user_lat, cme_user_lon, cme_user_a, cme_user_b, cme_user_c, cme_user_id


def write_file_cme():
    if not st.session_state.data:
        st.warning("No CME data available.")
        return

    # Convert session state data into a DataFrame
    df = pd.DataFrame(st.session_state.data)
    st.write("Dataframe User CMEs:", df)

    start_time = time.time()
    used = 7  # Adjust based on your machine
    print('Using multiprocessing, nr of cores', mp.cpu_count(), ', nr of processes used: ', used)

    # Create the pool for multiprocessing
    pool = mp.get_context('fork').Pool(processes=used)

    # Map the worker function onto the parameters
    results = pool.map(cme_kinematics, range(len(st.session_state.data)))

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    print('time in minutes: ',np.round((time.time()-start_time)/60))

    hc_time_num1_cme = [result[0] for result in results]
    hc_time_num1_cme = np.concatenate(hc_time_num1_cme)

    hc_r1_cme = [result[1] for result in results]
    hc_r1_cme = np.concatenate(hc_r1_cme, axis=1)

    hc_lat1_cme = [result[2] for result in results]
    hc_lat1_cme = np.concatenate(hc_lat1_cme)

    hc_lon1_cme = [result[3] for result in results]
    hc_lon1_cme = np.concatenate(hc_lon1_cme)

    a_ell_cme = [result[4] for result in results]
    a1_ell_cme = np.concatenate(a_ell_cme, axis=1)

    b_ell_cme = [result[5] for result in results]
    b1_ell_cme = np.concatenate(b_ell_cme, axis=1)

    c_ell_cme = [result[6] for result in results]
    c1_ell_cme = np.concatenate(c_ell_cme, axis=1)

    hc_id_cme = [result[7] for result in results]
    hc_id_cme = np.concatenate(hc_id_cme)

    pickle.dump([hc_time_num1_cme, hc_r1_cme, hc_lat1_cme, hc_lon1_cme, hc_id_cme, a1_ell_cme, b1_ell_cme, c1_ell_cme], open(overview_path+'user_cme_kinematics.p', "wb"))
                    

    print("len results:", len(results))
    print("len(st.session_state.data):",len(st.session_state.data))
    print("len(hc_r1_cme):", len(hc_r1_cme))
    print("hc_r1_cme:", hc_r1_cme[0,:])
    #print("len(hc_r1_cme[0]):",len(hc_r1_cme[0]))
    pickle.dump([hc_time_num1_cme, hc_r1_cme, hc_lat1_cme, hc_lon1_cme, hc_id_cme, a1_ell_cme, b1_ell_cme, c1_ell_cme], open(overview_path+'user_cme_kinematics.p', "wb"))
    

with col1:
    
    #st.header("Welcome to Cor-HI Explorer")
    st.image(path_to_logo+"/logo_corhi.png" , width=300)

    #st.header("🔍 **Select the interval of time**")
    st.markdown("<h4 style='color: magenta;'>🔍 Select the interval of time</h4>", unsafe_allow_html=True)

    # Input della data di inizio
    #t_start2 = st.text_input("initial time(YYYY-MM-DD H:M:S)", "2023-10-01 16:00:00")
    #st.markdown("<h2 style='font-size: 18px; color: black;'></h2>", unsafe_allow_html=True)
    t_start2 = st.text_input("Initial time", "2023-10-01 16:00:00")
    # Input della data di fine (formato stringa)

    #st.markdown("<h2 style='font-size: 18px; color: black;'>Final time</h2>", unsafe_allow_html=True)
    t_end2 = st.text_input("Final time", "2023-10-01 17:30:00")
    if t_end2  < t_start2:
        st.error('final time MUST be bigger than initial time.')
        st.stop()  # Interrompe l'esecuzione dell'applicazione

    

    option = st.radio("Select an option:", 
                  ("Plot all S/C and all instruments' FoV", 
                   "Let me select S/C and FoV"))
    if option == "Plot all S/C and all instruments' FoV":
        selected_sc = ["SOHO", "STA", "PSP", "SOLO", "BEPI"]
    elif option =="Let me select S/C and FoV":
        st.markdown("<h4 style='color: magenta;'>Select spacecraft </h4>", unsafe_allow_html=True)
        sc_options = ["SOHO", "STA", "PSP", "SOLO", "BEPI"]
        selected_sc = st.multiselect("Select spacecraft:", sc_options)


    
    
    if option == "Plot all S/C and all instruments' FoV":
        selected_coronagraphs = ["C2-C3", "COR1-COR2", "METIS"]
    elif option =="Let me select S/C and FoV":
        st.markdown("<h4 style='color: magenta;'>Show FoVs coronographs</h4>", unsafe_allow_html=True)
        coronagraph_options = []
        if "SOHO" in selected_sc:
            coronagraph_options.append("C2-C3")
        if "STA" in selected_sc: 
            coronagraph_options.append("COR1-COR2")
        if "SOLO" in selected_sc: 
            coronagraph_options.append("METIS")
        selected_coronagraphs = st.multiselect("Select coronographs:", coronagraph_options)
    
    
    if option == "Plot all S/C and all instruments' FoV":
        selected_his = ["WISPR", "STA HI", "SOLO HI"]
    elif option =="Let me select S/C and FoV":   
        st.markdown("<h4 style='color: magenta;'>Show FoVs HIs</h4>", unsafe_allow_html=True)
        his_options = []
        if "PSP" in selected_sc:
            his_options.append("WISPR")
        if "STA" in selected_sc: 
            his_options.append("STA HI")
        if "SOLO" in selected_sc: 
            his_options.append("SOLO HI")
        selected_his = st.multiselect("Select HIs:", his_options)
        
    st.markdown("<h4 style='color: magenta;'>Select Catalog (optional)</h4>", unsafe_allow_html=True)
    plot_hi_geo = st.checkbox("Plot HI-Geo catalog")
    plot_donki = st.checkbox("Plot DONKI catalog")
    plot_cme = st.checkbox("Plot user CMEs")

    if plot_cme:
        if 'cme_params' not in st.session_state:
            st.session_state.cme_params = []

        if 'data' not in st.session_state:
            st.session_state.data = []

        num_cmes = st.number_input("How many CMEs do you want to input?", min_value=1, max_value=6, step=1)


        with st.form("cme_form"):
            cme_params_list = []

            for i in range(num_cmes):
                st.write(f"### CME {i + 1}")
                k = st.number_input(f"Enter k for CME {i + 1}", min_value=0.0, max_value=1.0, key=f"k_{i}", value = 0.30)
                alpha = st.number_input(f"Enter alpha for CME {i + 1}", min_value=0.0, max_value=75.0, key=f"alpha_{i}", value = 45.0)
                longitude = st.number_input(f"Enter longitude (HGS) for CME {i + 1}", min_value=-180.0, max_value=180.0, key=f"longitude_{i}", value = 120.0)
                v = st.number_input(f"Enter speed (km/s) for CME {i + 1}", min_value=50.0, max_value=3000.0, key=f"v_{i}", value = 900.0)
                t_0 = st.text_input(f"Enter time at 21.5 Rsun(YYYY-MM-DD H:M:S) for CME {i + 1}", t_start2)

                cme_params_list.append((k, alpha, longitude, v, t_0))

            submitted = st.form_submit_button("Submit all CME parameters")
            if submitted:
                st.session_state.data = [] 
                st.session_state.data = [
                    {
                        'CME ID': f'CME{i + 1}',
                        't0': params[4],
                        'longitude': params[2],
                        'latitude': 0.0,
                        'speed': params[3],
                        'half angle': params[1]
                    }
                    for i, params in enumerate(cme_params_list)
                ]
                st.write("CME parameters submitted! Calculating kinematics...")

         
                write_file_cme()

            else:
                #st.write("No data collected. Be sure to submit the parameters!")
                st.warning("No data collected. Be sure to submit the parameters!")

url_donki = 'https://drive.google.com/file/d/1Ewl3l0t_LaggHb2n8jQyDOsU5fLQzyZy/view?usp=sharing'
url_C2 = 'https://drive.google.com/file/d/1lhMrhCXpJNS1FOlIPLcSrcbnTMLpspSR/view?usp=sharing'
url_cor1 = 'https://drive.google.com/file/d/1wTRUgwWqtkKbjLgW52WZxZW3T1jvb4Cy/view?usp=sharing'
url_metis = 'https://drive.google.com/file/d/1GogqQFdtTTIrcLWDmdUROXBZo54jbDFq/view?usp=sharing'
url_hi1A = 'https://drive.google.com/file/d/1W9XGlIUI4cuyIQGKb6Xizi0oZP9L1zOI/view?usp=sharing'
url_solohi ='https://drive.google.com/file/d/1gB40XdR2Vr3M9K9pD_iHLXwvZwDAb_tt/view?usp=sharing'
url_wispr = 'https://drive.google.com/file/d/14r2Vid2-OHs5oJDuzc1VvtYNVFEtTWCK/view?usp=sharing'

kinematic_donki_file = path_local_greta + "donki_kinematics_2019_now.p"
file_date_c2 = path_local_greta + "c2_custom_intervals.txt"
file_date_cor1 = path_local_greta + "cor1_custom_intervals.txt"
file_date_metis = path_local_greta + "metis_custom_intervals.txt"
file_date_hi1A = path_local_greta + "hi1A_custom_intervals.txt"
file_date_solohi= path_local_greta + "solohi_custom_intervals.txt"
file_date_wispr = path_local_greta + "wispr_custom_intervals.txt"

def download_from_gd(file_data_url, data_url):
    if not os.path.exists(file_data_url):
        # If it does not exist, download the file
        #st.write(f"Downloading the file {file_data_url}...")
        gdown.download(data_url, file_data_url, quiet=False,fuzzy=True)
    #else:
        #st.write(f"Folder {file_data_url} already exists. No need to download.")

download_from_gd(kinematic_donki_file, url_donki)
download_from_gd(file_date_c2, url_C2)
download_from_gd(file_date_cor1, url_cor1)
download_from_gd(file_date_metis, url_metis)
download_from_gd(file_date_hi1A, url_hi1A)
download_from_gd(file_date_solohi, url_solohi)
download_from_gd(file_date_wispr, url_wispr)

path_wispr_dates = file_date_wispr #path_dates + 'wispr_custom_intervals.txt'
path_solohi_dates = file_date_solohi #path_dates + 'solohi_custom_intervals.txt'
path_hi1A_dates = file_date_hi1A #path_dates + 'hi1A_custom_intervals.txt'

path_metis_dates = file_date_metis #path_dates +'metis_custom_intervals.txt'
path_cor1_dates = file_date_cor1 #path_dates +'cor1_custom_intervals.txt'
path_c2_dates = file_date_c2 #path_dates +'c2_custom_intervals.txt'

def reader_txt(file_path):
    times_obs = []
    with open(file_path, 'r') as f:
        for line in f:
            date_str = line.strip()
            date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            times_obs.append(date_obj)
    return times_obs


# HIs
times_wispr_obs = reader_txt(path_wispr_dates)
times_solohi_obs = reader_txt(path_solohi_dates)
times_hi1A_obs = reader_txt(path_hi1A_dates)

# CORs
times_metis_obs = reader_txt(path_metis_dates)
times_cor1_obs = reader_txt(path_cor1_dates)
times_c2_obs = reader_txt(path_c2_dates)

# Filter the dates    
t_start_dt = datetime.strptime(t_start2, '%Y-%m-%d %H:%M:%S') #%Y-%b-%d')
t_end_dt = datetime.strptime(t_end2, '%Y-%m-%d %H:%M:%S')

# Filter the dates    
dates_wispr_fov2 = [date for date in times_wispr_obs if t_start_dt <= date <= t_end_dt]
dates_solohi_fov2 = [date for date in times_solohi_obs if t_start_dt <= date <= t_end_dt]
dates_hi1A_fov2 = [date for date in times_hi1A_obs if t_start_dt <= date <= t_end_dt]
dates_metis_fov2 = [date for date in times_metis_obs if t_start_dt <= date <= t_end_dt]
dates_cor1_fov2 = [date for date in times_cor1_obs if t_start_dt <= date <= t_end_dt]
dates_c2_fov2 = [date for date in times_c2_obs if t_start_dt <= date <= t_end_dt]

# Create a set fot the dates in datatime
psp_set = set(pd.to_datetime(dates_wispr_fov2))
solo_set = set(pd.to_datetime(dates_solohi_fov2))
stereo_set = set(pd.to_datetime(dates_hi1A_fov2))
metis_set = set(pd.to_datetime(dates_metis_fov2))
cor1_set = set(pd.to_datetime(dates_cor1_fov2))
c2_set = set(pd.to_datetime(dates_c2_fov2))


date_overlap_all = []
date_overlap_wispr_stereohi = []
date_overlap_wispr_solohi = []
date_overlap_stereohi_solohi= []


# Function to display plots using a slider
def display_plots_with_slider(plot_files2):
    if len(plot_files2) > 0:
        selected_plot = st.slider('Select Plot', 0, len(plot_files2) - 1, 0)
        image_path = plot_files2[selected_plot]
        st.image(image_path, caption=f"Plot for {dates_hi1A_fov2[selected_plot].strftime('%Y-%m-%dT%H-%M-%S')}", use_column_width=True)

def clear_old_images(images_folder):
    for img_file in glob.glob(os.path.join(images_folder, '*')):
        os.remove(img_file)

def coord_to_polar(coord):
    return coord.lon.to_value('rad'), coord.radius.to_value('AU')

def create_custom_legend(ax, loc='upper right', fontsize=6, ncol=2, handlelength=2, bbox_to_anchor=(1.12, 1)):
    """Crea una leggenda con formato personalizzato e dimensione fissa."""
    legend = ax.legend(loc=loc, fontsize=fontsize, ncol=ncol, handlelength=handlelength, bbox_to_anchor=bbox_to_anchor)
    # Imposta una dimensione fissa per la box della leggenda
    frame = legend.get_frame()
    frame.set_boxstyle('round')
    frame.set_edgecolor('black')
    frame.set_linewidth(1.0)
    return legend

def create_animation(paths):
    fig, ax = plt.subplots()
    ax.axis('off')
    img = ax.imshow(plt.imread(paths[0]))
    
    def update(frame):
        img.set_array(plt.imread(paths[frame]))
        return img,

    ani = FuncAnimation(fig, update, frames=len(paths), interval=500, blit=True)
    return ani

# Function to save the animation
def save_animation(ani, filename, writer):
    ani.save(filename, writer=writer, fps=2)  # Adjust fps as needed
    return filename
min_date_solo = datetime(2020, 2, 10, 4, 56, 58)
min_date_solo_traj = datetime(2020, 2, 16, 4, 56, 58)

min_date_psp = datetime(2018, 8, 12, 23, 56, 58)
min_date_psp_traj = datetime(2018, 8, 17, 23, 56, 58)


min_date_bepi = datetime(2018, 10, 20, 10, 0, 0)
min_date_bepi_traj = datetime(2018, 10, 21, 10, 0, 0)

@st.cache_data
def get_coordinates(start_time_sc, end_time_sc):
    if start_time_sc <= min_date_psp and end_time_sc >= min_date_psp:
        start_time_sc_2 = min_date_psp
        psp_coord_func = get_horizons_coord('Parker Solar Probe',
                                {'start': parse_time(start_time_sc_2),
                                'stop': parse_time(end_time_sc),
                                'step': '30m'})
    else:
        psp_coord_func = get_horizons_coord('Parker Solar Probe',
                                {'start': parse_time(start_time_sc),
                                'stop': parse_time(end_time_sc),
                                'step': '30m'})
    
    return psp_coord_func

if not os.path.exists('images'):
            os.makedirs('images')

def clear_old_images(images_folder):
    for img_file in glob.glob(os.path.join(images_folder, '*')):
        os.remove(img_file)

def make_frame(ind):
    fsize = 10
    frame_time_num=parse_time(t_start2) #.plot_date
    res_in_days=1/48.
    starttime = parse_time(t_start2).datetime
    endtime = parse_time(t_end2).datetime
    mp.set_start_method('spawn', force=True)

    lock = mp.Lock()
    fig= plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='polar')
    ax.set_xticks(np.pi/180. * np.linspace(0,  360, 12, endpoint=False))

    ax.set_ylim(0,1.1)

    initial_datatime = datetime.strptime(t_start2, "%Y-%m-%d %H:%M:%S")
    final_datatime = datetime.strptime(t_end2, "%Y-%m-%d %H:%M:%S")

    start_date2 =  initial_datatime + timedelta(minutes=30 * ind)
    print(start_date2)
    date_obs_enc17 = start_date2

    start_time_make_frame11 = time.time()

    if initial_datatime >= min_date_psp or final_datatime>=min_date_psp:
        psp_coord_array = get_coordinates(initial_datatime, final_datatime)
        psp_coord = psp_coord_array[ind]
        ax.plot(*coord_to_polar(psp_coord),'v',label='PSP', color='blue',alpha=0.6)
    

    if plot_cme:   
        #hc_time_num1_cme, hc_r1_cme, hc_lat1_cme, hc_lon1_cme, hc_id1_cme, a1_ell_cme, b1_ell_cme, c1_ell_cme
        #the same for DONKI CMEs but with ellipse CMEs
        [hc_time_num1_cme, hc_r1_cme, hc_lat1_cme, hc_lon1_cme, hc_id_cme, a1_ell_cme, b1_ell_cme, c1_ell_cme]=pickle.load(open(overview_path+'user_cme_kinematics.p', "rb")) # last created: 2024-04-24
        cmeind1=np.where(hc_time_num1_cme == mdates.date2num(date_obs_enc17))
        #print(hc_id1[cmeind1])

        for p in range(0,np.size(cmeind1)):
            
            t = ((np.arange(201)-10)*np.pi/180)-(hc_lon1_cme[cmeind1[0][p]]*np.pi/180)
            t1 = ((np.arange(201)-10)*np.pi/180)
            
            longcirc1 = []
            rcirc1 = []
            for i in range(3):

                xc1 = c1_ell_cme[i][cmeind1[0][p]]*np.cos(hc_lon1_cme[cmeind1[0][p]]*np.pi/180)+((a1_ell_cme[i][cmeind1[0][p]]*b1_ell_cme[i][cmeind1[0][p]])/np.sqrt((b1_ell_cme[i][cmeind1[0][p]]*np.cos(t1))**2+(a1_ell_cme[i][cmeind1[0][p]]*np.sin(t1))**2))*np.sin(t)
                yc1 = c1_ell_cme[i][cmeind1[0][p]]*np.sin(hc_lon1_cme[cmeind1[0][p]]*np.pi/180)+((a1_ell_cme[i][cmeind1[0][p]]*b1_ell_cme[i][cmeind1[0][p]])/np.sqrt((b1_ell_cme[i][cmeind1[0][p]]*np.cos(t1))**2+(a1_ell_cme[i][cmeind1[0][p]]*np.sin(t1))**2))*np.cos(t)

                longcirc1.append(np.arctan2(yc1, xc1))
                rcirc1.append(np.sqrt(xc1**2+yc1**2))

            ax.plot(longcirc1[0], rcirc1[0], color='tab:brown', ls='-', alpha=0.5, lw=2.0) #2-abs(hc_lat1_cme[cmeind1[0][p]]/100)
            ax.fill_between(longcirc1[2], rcirc1[2], rcirc1[1], color='tab:brown', alpha=.08)
            plt.figtext(0.02, 0.060-p*0.02,f"CME {p+1}", fontsize=fsize, ha='left',color='tab:brown')

    
    ax.set_title(start_date2)

    create_custom_legend(ax)

    return fig



with col2:


    st.markdown(
            """
            <h1 style='font-size: 40px;'>Welcome to Cor-HI Explorer!</h1>
            <l3 style='font-size: 20px;'>
            This tool enables you to explore spacecraft constellations, visualize instrument fields of view, check data availability for coronagraphs and heliospheric imagers, and propagate either a catalog of CMEs or your own CME data.). We hope this tool helps you explore heliospheric events with ease!

            """, 
            unsafe_allow_html=True
        )

    
    plot_files = []

    if not os.path.exists('images'):
                os.makedirs('images')

    if 'plot_files' not in st.session_state:
        st.session_state.plot_files = []
    if 'dates_hi1A_fov2' not in st.session_state:
        st.session_state.dates_hi1A_fov2 = dates_hi1A_fov2
    # Button to generate the plots
    
    if st.button('Generate the plots'):
        st.session_state.plot_files.clear()  # Clear previous plots
        start_date_t = datetime.strptime(t_start2, "%Y-%m-%d %H:%M:%S")
        end_date_t = datetime.strptime(t_end2, "%Y-%m-%d %H:%M:%S")
        # Calcola la differenza tra le due date
        delta = end_date_t - start_date_t
        intervals_30_min = delta.total_seconds() / (30 * 60)

        print(intervals_30_min)

        start_time_make_frame = time.time() 
        #main()
        #print('time make frame in minutes: ',np.round((time.time()-start_time_make_frame)/60))
        figures = []
        paths_to_fig = []
        for interval in range(int(intervals_30_min)):
            title = datetime.strptime(t_start2, "%Y-%m-%d %H:%M:%S")  + timedelta(minutes=30 * interval)     
            try:
                # Create the plot
                fig = make_frame(interval)  # Replace with your actual plotting function                     
                file_path = os.path.join('images', f"{title}_sc_constellation.png")      
                paths_to_fig.append(file_path)       
                # Save the figure to the file and add to plot_files               
                fig.savefig(file_path)
                plt.close(fig)
                st.session_state.plot_files.append(file_path)
            except Exception as e:
                        st.error(f"Error generating plot for {title}: {e}")

        print('time make frame in minutes: ',np.round((time.time()-start_time_make_frame)/60))
  
        ani = create_animation(paths_to_fig)
      #  output_dir = 'animations'
      #  os.makedirs(output_dir, exist_ok=True)

        # Save the animation as MP4 and GIF
   #     mp4_file_path = os.path.join(output_dir, 'animation.mp4')
  #      gif_file_path = os.path.join(output_dir, 'animation.gif')

        # Save the MP4 animation
  #     ani.save(mp4_file_path, writer='ffmpeg')

        # Save the GIF animation
   #     ani.save(gif_file_path, writer='imagemagick')

        # Display the animation in Streamlit
        st.video(mp4_file_path)

        # Create download buttons for the saved files
#        with open(gif_file_path, "rb") as f:
#            gif_data = f.read()

#        with open(mp4_file_path, "rb") as f:
#           mp4_data = f.read()

#       st.download_button(
#            label="Download as GIF",
#            data=gif_data,
#           file_name='animation.gif',
#            mime='image/gif'
#        )

#        st.download_button(
#            label="Download as MP4",
#            data=mp4_data,
#            file_name='animation.mp4',
#            mime='video/mp4'
#        )
