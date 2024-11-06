# cd /Users/gretacappello/Desktop/jupyter_notebooks/elevohi
# Activate GCS environment: conda activate elevohi
# Run the interface: streamlit run corhi_explorer.py &
#if files are not found, pay attention that they are not visible due to the icloud


import streamlit as st
import gdown
import pytz
import tempfile
import matplotlib.lines as mlines
#change path for ffmpeg for animation production if needed
#%matplotlib inline
#ffmpeg_path=''
from PIL import Image
import concurrent.futures
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




with col1:
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
        
    #st.header("Welcome to Cor-HI Explorer")
    st.image(path_to_logo+"/logo_corhi.png" , width=400)

    #st.header("🔍 **Select the interval of time**")
    st.markdown("<h4 style='color: magenta;'>🔍 Select the interval of time</h4>", unsafe_allow_html=True)


    # Set up date selector
    selected_date = st.date_input("Select Initial Date:", datetime(2023, 10, 1))

    # Set up 30-minute intervals as options
    time_options = [(datetime.min + timedelta(hours=h, minutes=m)).strftime("%H:%M") 
                    for h in range(24) for m in (0, 30)]
    selected_time = st.selectbox("Select Initial Time:", time_options)

    # Combine selected date and time
    t_start2 = f"{selected_date} {selected_time}:00"

    if "t_start2" not in st.session_state:
        st.session_state["t_start2"] = t_start2

    if "t_end2" not in st.session_state:
        st.session_state["t_end2"] = (datetime.strptime(st.session_state["t_start2"], "%Y-%m-%d %H:%M:%S") + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")


    # Validate and update t_start2 if changed
    if t_start2 != st.session_state["t_start2"]:
        st.session_state["t_start2"] = t_start2
        st.session_state["t_end2"] = (datetime.strptime(st.session_state["t_start2"], "%Y-%m-%d %H:%M:%S") + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

    # Interval selection inside an expander
    with st.expander("Define interval of time (default = 1 day):", expanded=False):
        # Choose interval options
        interval_choice = st.radio("Select:", ["+1 Day", "+5 Days", "+20 Days", "Add Hours"])

        # Option for adding specific hours
        x_hours = 0
        if interval_choice == "Add Hours":
            x_hours = st.number_input("Add hours:", min_value=1, step=1)

        # Update t_end2 based on selected interval when "Apply" button is pressed
        
        if interval_choice == "+1 Day":
            st.session_state["t_end2"] = (datetime.strptime(st.session_state["t_start2"], "%Y-%m-%d %H:%M:%S") + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        elif interval_choice == "+5 Days":
            st.session_state["t_end2"] = (datetime.strptime(st.session_state["t_start2"], "%Y-%m-%d %H:%M:%S") + timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
        elif interval_choice == "+20 Days":
            st.session_state["t_end2"] = (datetime.strptime(st.session_state["t_start2"], "%Y-%m-%d %H:%M:%S") + timedelta(days=20)).strftime("%Y-%m-%d %H:%M:%S")
        elif interval_choice == "Add Hours":
            st.session_state["t_end2"] = (datetime.strptime(st.session_state["t_start2"], "%Y-%m-%d %H:%M:%S") + timedelta(hours=x_hours)).strftime("%Y-%m-%d %H:%M:%S")
        
        if datetime.strptime(st.session_state["t_start2"].year > 2019:
            st.success(f"Initial Time: {st.session_state['t_start2']}")
            st.success(f"Final Time: {st.session_state['t_end2']}")
    try:
        start_time = datetime.strptime(st.session_state["t_start2"], "%Y-%m-%d %H:%M:%S")
        if start_time.year < 2019:
            st.error('Initial time must be from the year 2019 or later.')

    except ValueError:
        st.error('Initial time is not in the correct format. Use YYYY-MM-DD HH:MM:SS.')
        st.stop()  # Stop execution if the format is invalid

   



    # Filter the dates    
    t_start_dt = datetime.strptime(st.session_state["t_start2"], '%Y-%m-%d %H:%M:%S') #%Y-%b-%d')
    t_end_dt = datetime.strptime(st.session_state["t_end2"], '%Y-%m-%d %H:%M:%S')




    with st.expander("Define plot cadence (default = 6 hrs):", expanded=False):
        time_cadence = st.select_slider(
            "Select:",
            options=["30 min", "1 hrs", "2 hrs", "6 hrs", "12 hrs"],
            value="6 hrs"  # Default selection
        )

        if time_cadence == "30 min":
            cad =  0.5
        if time_cadence == "1 hrs":
            cad = 1
        if time_cadence == "2 hrs":
            cad = 2
        if time_cadence == "6 hrs":
           cad = 6
        if time_cadence == "12 hrs":
           cad = 12

        
    option = st.radio("Select an option:", 
                  ("Plot all S/C and all instruments' FoV", 
                   "Let me select S/C and FoV"))

    if option == "Plot all S/C and all instruments' FoV":
        selected_sc = ["SOHO", "STA", "PSP", "SOLO", "BEPI"]
        selected_coronagraphs = ["C2-C3", "COR1-COR2", "METIS"]
        selected_his = ["WISPR", "STA HI", "SOLO HI"]

    elif option =="Let me select S/C and FoV":

        st.markdown("<h4 style='color: magenta;'>Select spacecraft </h4>", unsafe_allow_html=True)
        sc_options = ["SOHO", "STA", "PSP", "SOLO", "BEPI"]
        selected_sc = st.multiselect("Select spacecraft:", sc_options)

        st.markdown("<h4 style='color: magenta;'>Show FoVs coronographs</h4>", unsafe_allow_html=True)
        coronagraph_options = []
        if "SOHO" in selected_sc:
            coronagraph_options.append("C2-C3")
        if "STA" in selected_sc: 
            coronagraph_options.append("COR1-COR2")
        if "SOLO" in selected_sc: 
            coronagraph_options.append("METIS")
        selected_coronagraphs = st.multiselect("Select coronographs:", coronagraph_options)
    
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
                t_0 = st.text_input(f"Enter time at 21.5 Rsun(YYYY-MM-DD H:M:S) for CME {i + 1}", st.session_state["t_start2"])

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
# Play/Pause functionality




url_donki = 'https://drive.google.com/file/d/1pPlbsvjE6GaE2I6gGWcEC5Axls04cQmm/view?usp=sharing'
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




def generate_frames_parallel(num_frames):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        frames = list(executor.map(make_frame, range(num_frames)))
    return frames

def clear_old_images(images_folder):
    for img_file in glob.glob(os.path.join(images_folder, '*')):
        os.remove(img_file)

def coord_to_polar(coord):
    return coord.lon.to_value('rad'), coord.radius.to_value('AU')

def create_custom_legend(ax):
    # Define Field of View (FoV) lines
    fov_legend = [
        mlines.Line2D([], [], color='orange', linewidth=1, label='METIS FoV'),
        mlines.Line2D([], [], color='green', linewidth=1, label='C3 FoV'),
        mlines.Line2D([], [], color='magenta', linewidth=1, label='COR2 FoV'),
        mlines.Line2D([], [], color='black', linewidth=1, label='SolO-HI FoV'),
        mlines.Line2D([], [], color='blue', linewidth=1, label='WISPR-I FoV'),
        mlines.Line2D([], [], color='brown', linewidth=1, label='STEREO-A HI FoV'),
    ]
    
    # Define objects with markers
    object_legend = [
        mlines.Line2D([], [], color='yellow', marker='.', markersize=7, linestyle='None', label='Sun'),
        mlines.Line2D([], [], color='skyblue', marker='o', markersize=7, linestyle='None', label='Earth', alpha=0.6),
        mlines.Line2D([], [], color='blue', marker='v', markersize=7, linestyle='None', label='PSP', alpha=0.6),
        mlines.Line2D([], [], color='brown', marker='v', markersize=7, linestyle='None', label='STEREOA', alpha=0.6),
        mlines.Line2D([], [], color='green', marker='v', markersize=7, linestyle='None', label='SOHO', alpha=0.6),
        mlines.Line2D([], [], color='violet', marker='v', markersize=7, linestyle='None', label='BepiColombo', alpha=0.6),
        mlines.Line2D([], [], color='black', marker='v', markersize=7, linestyle='None', label='SolO', alpha=0.6),
    ]

    custom_legend_items = [
        mlines.Line2D([0], [0], color='tab:orange', lw=1, label='HELCATS'),
        mlines.Line2D([0], [0], color='tab:blue', lw=1, label='DONKI'),
        mlines.Line2D([0], [0], color='tab:brown', lw=1, label='CME'),
    ]   
    # Define overlap area
    overlap_legend = [
        mlines.Line2D([], [], color='y', alpha=0.15, linewidth=8, label='Overlap HI FoVs')
    ]
    
    # Combine all legend items
    all_legend_items = fov_legend + object_legend + overlap_legend + custom_legend_items
    
    # Add the legend to the plot
    #ax.legend(handles=all_legend_items, loc='upper left', title='Legend')
    ax.legend(
        handles=all_legend_items, 
        loc='lower center', 
        bbox_to_anchor=(1, -0.27),  # Position at the bottom right
        borderaxespad=0,
        fontsize=10,       # Smaller text size
        ncol=2                 # Number of columns
    )

def create_animation(paths):
    fig, ax = plt.subplots()
    ax.axis('off')
    img = ax.imshow(plt.imread(paths[0]))

    def update(frame):
        img.set_array(plt.imread(paths[frame]))
        return img,

    ani = FuncAnimation(fig, update, frames=len(paths), interval = 50, repeat=False)
    return ani


# Function to save the animation
def save_animation(ani, filename, writer):
    ani.save(filename, writer=writer, fps=2)  # Adjust fps as needed
    return filename

def get_coordinates(spacecraft, start_time_sc, end_time_sc, minimum_time, cadence='30m'):
    # Parse start and end times to ensure datetime format
    start_time_sc = parse_time(start_time_sc)
    end_time_sc = parse_time(end_time_sc)
    # Adjust start time if it is below the minimum time for this spacecraft
    start_time_query = max(start_time_sc, minimum_time)  # Ensures start_time_query is not earlier than minimum_time

    # Only proceed if the adjusted start time is not later than the end time
    if start_time_query > end_time_sc:
        return (0, 0, 0)
        #raise ValueError(f"Adjusted start time for {spacecraft} is after the requested end time.")
    else:
        sc_coordinates = get_horizons_coord(spacecraft,
                                            {'start': start_time_query.strftime("%Y-%m-%d %H:%M:%S"),
                                            'stop': end_time_sc.strftime("%Y-%m-%d %H:%M:%S"),
                                            'step': cadence})
        return sc_coordinates

min_date_solo = datetime(2020, 2, 10, 10, 4, 56)
#min_date_solo_traj = datetime(2020, 2, 16, 4, 56, 58)
min_date_psp = datetime(2018, 8, 12, 23, 56, 58)
#min_date_psp_traj = datetime(2018, 8, 17, 23, 56, 58)
min_date_bepi = datetime(2018, 10, 20, 10, 0, 0)
#min_date_bepi_traj = datetime(2018, 10, 21, 10, 0, 0)
min_date_soho = datetime(2019, 1, 1, 1, 0, 0)
min_date_stereo = datetime(2019, 1, 1,1, 0, 0)

@st.cache_data
def get_all_coordinates(start_time_sc, end_time_sc):

    psp_coord_func = get_coordinates("Parker Solar Probe", start_time_sc, end_time_sc, min_date_psp, cadence = '30m')    
    solo_coord_func= get_coordinates("Solar Orbiter", start_time_sc, end_time_sc, min_date_solo, cadence = '30m')
    bepi_coord_func = get_coordinates("BepiColombo", start_time_sc, end_time_sc, min_date_bepi, cadence = '30m')
    soho_coord_func = get_coordinates("SOHO", start_time_sc, end_time_sc, min_date_soho, cadence = '30m')
    sta_coord_func = get_coordinates("STEREO-A", start_time_sc, end_time_sc, min_date_stereo, cadence = '30m')
    
    return psp_coord_func, solo_coord_func, bepi_coord_func, soho_coord_func, sta_coord_func


if not os.path.exists('images'):
            os.makedirs('images')

def clear_old_images(images_folder):
    for img_file in glob.glob(os.path.join(images_folder, '*')):
        os.remove(img_file)

def fov_to_polygon(angles, radii):
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return Polygon(np.column_stack((x, y)))

@st.cache_data()
def read_donki():
    [hc_time_num1_date, hc_r1_func, hc_lat1_func, hc_lon1_func, hc_id1_func, a1_ell_func, b1_ell_func, c1_ell_func]=pickle.load(open(kinematic_donki_file, "rb")) 
    hc_time_num11 = mdates.date2num(hc_time_num1_date)
    return hc_time_num11, hc_r1_func, hc_lat1_func, hc_lon1_func, hc_id1_func, a1_ell_func, b1_ell_func, c1_ell_func

@st.cache_data()
def read_higeocat():
    [hc_time_num_func,hc_r_func,hc_lat_func,hc_lon_func,hc_id_func]=pickle.load(open('./higeocat_kinematics.p', "rb"))
    hc_time_num22 = mdates.date2num(hc_time_num_func)
    return hc_time_num22,hc_r_func,hc_lat_func,hc_lon_func,hc_id_func

def create_gif_animation(paths, duration):
    # Open all images and add them to a list
    frames = [Image.open(path) for path in paths]
    
    # Create a BytesIO buffer to save the GIF in memory
    gif_buffer = io.BytesIO()
    
    # Save the frames as a GIF in memory
    frames[0].save(
        gif_buffer, format="GIF", save_all=True, append_images=frames[1:], 
        duration=duration, loop=0
    )
    
    # Reset the buffer's position to the start
    gif_buffer.seek(0)
    
    return gif_buffer

def make_frame(ind):
    fsize = 14
    frame_time_num=parse_time(st.session_state["t_start2"]) #.plot_date
    res_in_days=1/48.
    starttime = parse_time(st.session_state["t_start2"]).datetime
    endtime = parse_time(st.session_state["t_end2"]).datetime
    mp.set_start_method('spawn', force=True)

    lock = mp.Lock()
    fig= plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(projection='polar')
    ax.set_xticks(np.pi/180. * np.linspace(0,  360, 12, endpoint=False))

    ax.set_ylim(0,1.1)

    initial_datatime = datetime.strptime(st.session_state["t_start2"], "%Y-%m-%d %H:%M:%S")
    final_datatime = datetime.strptime(st.session_state["t_end2"], "%Y-%m-%d %H:%M:%S")

    if time_cadence == "30 min":
        start_date2 =  initial_datatime + timedelta(minutes=30 * ind)
    if time_cadence == "1 hrs":
        start_date2 =  initial_datatime + timedelta(hours=1 * ind)
    if time_cadence == "2 hrs":
        start_date2 =  initial_datatime + timedelta(hours=2 * ind)
    if time_cadence == "6 hrs":
        start_date2 =  initial_datatime + timedelta(hours=6 * ind)
    if time_cadence == "12 hrs":
        start_date2 =  initial_datatime + timedelta(hours=12 * ind)

    print(start_date2)
    date_obs_enc17 = start_date2



    #if initial_datatime >= min_date_psp or final_datatime>=min_date_psp:
    #    psp_coord_array = get_coordinates(initial_datatime, final_datatime)
    #    psp_coord = psp_coord_array[ind]
    #    ax.plot(*coord_to_polar(psp_coord),'v',label='PSP', color='blue',alpha=0.6)
    

#!!!A condition must be placed to consider times before 2019, easily put a condition in start time
#*****************************
#POSITION SPACECRAFT 
#*****************************
    if parse_time(date_obs_enc17) >= min_date_solo:
        psp_coord_array, solo_coord_array, bepi_coord_array, soho_coord_array, sta_coord_array = get_all_coordinates(initial_datatime,final_datatime)       
        solo_coord = solo_coord_array[ind]
    else:
        psp_coord_array, solo_coord_array, bepi_coord_array, soho_coord_array, sta_coord_array = get_all_coordinates(initial_datatime,final_datatime)
        
    
    #psp_coord_array, solo_coord_array, bepi_coord_array, soho_coord_array, sta_coord_array = get_all_coordinates(initial_datatime,final_datatime)
    psp_coord = psp_coord_array[ind]
    r=psp_coord.radius
    #solo_coord = solo_coord_array[ind]
    bepi_coord = bepi_coord_array[ind]
    soho_coord = soho_coord_array[ind]
    stereo_coord =sta_coord_array[ind]
    sun_coord = (0, 0, 0)

    beta=90-13  #inner istrument - lim1
    beta2=90-53 #inner istrument - lim2

    beta3=90-50  #outer istrument - lim1
    beta4=90-108 #outer istrument - lim2


    beta_bis=90-4  #outer istrument - lim1
    beta2_bis=90+4 #outer istrument - lim2

    beta3_bis=90-15  #outer istrument - lim1
    beta4_bis=90+15 #outer istrument - lim2

    beta1_hi = 90-4
    beta2_hi = 90-88

    betaplus=89.99999 #per evitare divisione per cos90=0


    sun_coord = get_body_heliographic_stonyhurst('sun', time=date_obs_enc17)
    earth_coord = get_body_heliographic_stonyhurst('earth', time=date_obs_enc17)

    def fov_plotter_cori(coords, angle_fov, keyword2, color_line): 
        beta_bis=90-angle_fov  #outer istrument - lim1
        beta2_bis=90+angle_fov #outer istrument - lim2
        betaplus = 89.9
        r1fov_bis=coords.radius*np.sin(np.radians(-angle_fov))
        r2fov_bis=coords.radius*np.sin(np.radians(angle_fov))
        rb1fov_bis=r1fov_bis/np.cos(np.radians(betaplus))
        rb2fov_bis=r2fov_bis/np.cos(np.radians(betaplus)) 

        fov1_angles_bis=[coords.lon.to('rad').value,coords.lon.to('rad').value + np.radians(beta_bis+betaplus)]
        fov1_ra_bis=[coords.radius.value,rb1fov_bis.value]

        fov2_angles_bis=[coords.lon.to('rad').value,coords.lon.to('rad').value+np.radians(beta2_bis+betaplus)]
        fov2_ra_bis=[coords.radius.value,rb2fov_bis.value]

        ax.plot(fov1_angles_bis, fov1_ra_bis, color = color_line,linewidth=1, label = keyword2 +' FoV')
        ax.plot(fov2_angles_bis, fov2_ra_bis, color = color_line,linewidth=1)

        r1fov_bis=coords.radius*np.sin(np.radians(angle_fov))
        r2fov_bis=coords.radius*np.sin(np.radians(-angle_fov))
        rb1fov_bis=r1fov_bis/np.cos(np.radians(betaplus))
        rb2fov_bis=r2fov_bis/np.cos(np.radians(betaplus)) 

       # print(solo_coord.lon.to('rad').value)

        fov1_angles_bis=[coords.lon.to('rad').value,coords.lon.to('rad').value+
                        np.radians(beta_bis+betaplus)]
        fov1_ra_bis=[coords.radius.value,rb1fov_bis.value]

        fov2_angles_bis=[coords.lon.to('rad').value,coords.lon.to('rad').value+np.radians(beta2_bis+betaplus)]
        fov2_ra_bis=[coords.radius.value,rb2fov_bis.value]

        ax.plot(fov1_angles_bis, fov1_ra_bis, color = color_line, linewidth=1)
        ax.plot(fov2_angles_bis, fov2_ra_bis, color = color_line, linewidth=1)
    
    #if date_obs_enc17 >= min_date_solo:
    if parse_time(date_obs_enc17) > min_date_solo:
        if 'SOLO HI' in selected_his:
            if (start_date2 in solo_set):
                    #solo hi
                    r1fov_shi=solo_coord.radius*np.sin(np.radians(5))
                    r2fov_shi=solo_coord.radius*np.sin(np.radians(45))
                    rb1fov_shi=r1fov_shi/np.cos(np.radians(betaplus))
                    rb2fov_shi=r2fov_shi/np.cos(np.radians(betaplus)) 

                    fov1_angles_shi=[solo_coord.lon.to('rad').value,solo_coord.lon.to('rad').value+np.radians(180+5)]
                    fov1_ra_shi=[solo_coord.radius.value,rb1fov_shi.value]

                    fov2_angles_shi=[solo_coord.lon.to('rad').value,solo_coord.lon.to('rad').value+np.radians(180+45)]
                    fov2_ra_shi=[solo_coord.radius.value,rb2fov_shi.value]

                    ax.plot(fov1_angles_shi, fov1_ra_shi, color = 'black',linewidth=1,  label = 'SolO-HI FoV')
                    ax.plot(fov2_angles_shi, fov2_ra_shi, color = 'black',linewidth=1)


        if 'METIS' in selected_coronagraphs:
            if (start_date2 in metis_set):  
                fov_plotter_cori(solo_coord, 3, 'METIS', 'orange')

    if 'C2-C3' in selected_coronagraphs:
        if (start_date2 in c2_set):
            fov_plotter_cori(soho_coord, 7.5, 'C3', 'green')
       
    if 'COR1-COR2' in selected_coronagraphs:
        if (start_date2 in cor1_set):
            fov_plotter_cori(stereo_coord, 3, 'COR2', 'magenta')

    if 'WISPR' in selected_his:
        if date_obs_enc17 >= min_date_psp:
            if (start_date2 in psp_set):
                    #psp_coord=SkyCoord(header['HGLN_OBS']*3600*u.arcsec, header['HGLT_OBS']*3600*u.arcsec, r*u.au, obstime=header['date-obs'], observer="earth", frame="heliographic_stonyhurst")

                #if header['OBJECT']== 'InnerFFV' :
                r1fov=psp_coord.radius*np.sin(np.radians(13))
                r2fov=psp_coord.radius*np.sin(np.radians(53))
                rb1fov=r1fov/np.cos(np.radians(betaplus))
                rb2fov=r2fov/np.cos(np.radians(betaplus)) 

                #print(psp_coord.lon.to('rad').value)

                fov1_angles=[psp_coord.lon.to('rad').value,psp_coord.lon.to('rad').value+np.radians(beta+betaplus)]
                fov1_ra=[psp_coord.radius.value,rb1fov.value]

                fov2_angles=[psp_coord.lon.to('rad').value,psp_coord.lon.to('rad').value+np.radians(beta2+betaplus)]
                fov2_ra=[psp_coord.radius.value,rb2fov.value]

                angles=[psp_coord.lon.to('rad').value,psp_coord.lon.to('rad').value+np.radians(beta+betaplus), psp_coord.lon.to('rad').value+np.radians(beta2+betaplus)]
                rs=[psp_coord.radius.value,rb1fov.value,rb2fov.value]


                r_conj= [0.0, psp_coord.radius.value]
                theta_conj = [np.radians(0), psp_coord.lon.to('rad').value+np.radians(0)]
                #if header['OBJECT']== 'OuterFFV' :
                r1fov_outer=psp_coord.radius*np.sin(np.radians(50))
                r2fov_outer=psp_coord.radius*np.sin(np.radians(108))
                rb1fov_outer=r1fov_outer/np.cos(np.radians(betaplus))
                rb2fov_outer=r2fov_outer/np.cos(np.radians(betaplus)) 

                fov1_angles_outer=[psp_coord.lon.to('rad').value,psp_coord.lon.to('rad').value+np.radians(beta3+betaplus)]
                fov1_ra_outer=[psp_coord.radius.value,rb1fov_outer.value]

                fov2_angles_outer=[psp_coord.lon.to('rad').value,psp_coord.lon.to('rad').value+np.radians(beta4+betaplus)]
                fov2_ra_outer=[psp_coord.radius.value,rb2fov_outer.value]

                ax.plot(fov1_angles, fov1_ra, color = 'blue',linewidth=1, label = 'WISPR-I FoV')
                ax.plot(fov2_angles, fov2_ra, color = 'blue',linewidth=1)

                ax.plot(fov1_angles_outer, fov1_ra_outer, color = 'blue',linewidth=1, label = 'WISPR-O FoV', linestyle="dashed")
                ax.plot(fov2_angles_outer, fov2_ra_outer, color = 'blue',linewidth=1, linestyle="dashed")
   


    if 'STA HI' in selected_his:
        if (start_date2 in stereo_set):
            r1fov_stAi=stereo_coord.radius*np.sin(np.radians(4))
            r2fov_stAi=stereo_coord.radius*np.sin(np.radians(88))
            rb1fov_stAi=r1fov_stAi/np.cos(np.radians(betaplus))
            rb2fov_stAi=r2fov_stAi/np.cos(np.radians(betaplus)) 
            if stereo_coord.lon > earth_coord.lon:
                fov1_angles_stAi=[stereo_coord.lon.to('rad').value,stereo_coord.lon.to('rad').value+np.radians(4+180)]
                fov1_ra_stAi=[stereo_coord.radius.value,rb1fov_stAi.value]

                fov2_angles_stAi=[stereo_coord.lon.to('rad').value,stereo_coord.lon.to('rad').value+np.radians(88+180)]
                fov2_ra_stAi=[stereo_coord.radius.value,rb2fov_stAi.value]

            else:
                fov1_angles_stAi=[stereo_coord.lon.to('rad').value,stereo_coord.lon.to('rad').value+np.radians(4+betaplus)]
                fov1_ra_stAi=[stereo_coord.radius.value,rb1fov_stAi.value]

                fov2_angles_stAi=[stereo_coord.lon.to('rad').value,stereo_coord.lon.to('rad').value+np.radians(88+betaplus)]
                fov2_ra_stAi=[stereo_coord.radius.value,rb2fov_stAi.value]
            ax.plot(fov1_angles_stAi, fov1_ra_stAi, color = 'brown',linewidth=1, label = 'STEREO-A HI FoV')
            ax.plot(fov2_angles_stAi, fov2_ra_stAi, color = 'brown',linewidth=1)


    # Initialize polygons to None or empty GeometryCollection
    polygon_wispr_i = GeometryCollection()
    polygon_stereo_hi = GeometryCollection()
    polygon_solo_hi = GeometryCollection()
    overlap_wispr_stereo = GeometryCollection()
    overlap_solo_stereo = GeometryCollection() 
        # Funzione di utilità per convertire i contorni del FoV in un poligono Shapely
    if 'WISPR' in selected_his and 'PSP' in selected_sc:
        if date_obs_enc17 >= min_date_psp:
            if (start_date2 in psp_set):
            # Convertiamo i FoV in poligoni
                polygon_wispr_i = fov_to_polygon(np.concatenate((fov1_angles, fov2_angles_outer[::-1])), np.concatenate((fov1_ra, fov2_ra_outer[::-1])))
            #polygon_wispr_o = fov_to_polygon(np.concatenate((fov1_angles_outer, fov2_angles_outer[::-1])), np.concatenate((fov1_ra_outer, fov2_ra_outer[::-1])))
    
    if 'STA HI' in selected_his and 'STA' in selected_sc:    
        if (start_date2 in stereo_set):
            polygon_stereo_hi = fov_to_polygon(np.concatenate((fov1_angles_stAi, fov2_angles_stAi[::-1])), np.concatenate((fov1_ra_stAi, fov2_ra_stAi[::-1])))

    if 'SOLO HI' in selected_his and 'SOLO' in selected_sc:    
        if date_obs_enc17 >= min_date_solo:
            if (start_date2 in solo_set):
                polygon_solo_hi = fov_to_polygon(np.concatenate((fov1_angles_shi, fov2_angles_shi[::-1])), np.concatenate((fov1_ra_shi, fov2_ra_shi[::-1])))

    if 'WISPR' and 'STA HI' in selected_his and 'PSP' and 'STA' in selected_sc:
        if date_obs_enc17 >= min_date_psp:
        # Calcoliamo le intersezioni tra i poligoni
            if (start_date2 in psp_set) and (start_date2 in stereo_set):
                overlap_wispr_stereo = polygon_wispr_i.intersection(polygon_stereo_hi)
    else: 
        overlap_wispr_stereo = GeometryCollection()

    if 'SOLO HI' and 'STA HI' in selected_his and 'SOLO' and 'STA' in selected_sc:
        if date_obs_enc17 >= min_date_solo:
        # Calcoliamo le intersezioni tra i poligoni
            if (start_date2 in solo_set) and (start_date2 in stereo_set):
                overlap_solo_stereo = polygon_solo_hi.intersection(polygon_stereo_hi)
    else: 
        overlap_solo_stereo = GeometryCollection()    #it gives an empty intersection when i de-select e.g. sta


    if 'WISPR' and 'STA HI' and 'SOLO HI' in selected_his and 'PSP' and 'STA' and 'SOLO' in selected_sc:               
        if date_obs_enc17 >= min_date_solo:
            if (start_date2 in psp_set) and (start_date2 in stereo_set) and (start_date2 in solo_set):   
                overlap_wispr_stereo_solo = overlap_wispr_stereo.intersection(polygon_solo_hi)
    else: 
            overlap_wispr_stereo_solo = GeometryCollection()      

    if date_obs_enc17 >= min_date_solo:
        if 'WISPR' and 'SOLO HI' in selected_his and 'PSP' and 'SOLO' in selected_sc:
            if (start_date2 in psp_set) and (start_date2 in solo_set):
                overlap_wispr_solo = polygon_wispr_i.intersection(polygon_solo_hi)
    else: 
        overlap_wispr_solo = GeometryCollection()      
    
    ax.plot(0, 0, marker=".",markersize=10, label='Sun', color='yellow')
    ax.plot(*coord_to_polar(earth_coord), 'o', markersize=10 ,label='Earth', color='skyblue',alpha=0.6)
    if 'PSP' in selected_sc:
        if date_obs_enc17 >= min_date_psp:
            ax.plot(*coord_to_polar(psp_coord),'v', markersize=10 ,label='PSP', color='blue',alpha=0.6)
        #if date_obs_enc17 >= min_date_psp_traj:
        #    ax.plot(*coord_to_polar(psp_coord_traj.transform_to(earth_coord)),label='PSP -5/+5 day', color='blue', linestyle='solid',  linewidth=1.5)
    if 'STA' in selected_sc:
        ax.plot(*coord_to_polar(stereo_coord),'v', markersize=10 ,label='STEREOA', color='brown',alpha=0.6)
        #ax.plot(*coord_to_polar(stereo_coord_traj.transform_to(earth_coord)),label='STEREO A -1/+1 day', color='brown', linestyle='dashed',  linewidth=1.5)
    if 'SOHO' in selected_sc:
            ax.plot(*coord_to_polar(soho_coord),'v', markersize=10 ,label='SOHO', color='green',alpha=0.6)
        #    ax.plot(*coord_to_polar(soho_coord_traj.transform_to(earth_coord)),label='SOHO A -1/+1 day', color='green', linestyle='dashed',  linewidth=1.5)
    
    #ax.plot(*coord_to_polar(stereo_coord_traj),'-', color='brown', label='STEREOA (as seen from Earth)',  linewidth=1.5)
    #print(stereo_coord)
    if 'BEPI' in selected_sc:
        #if date_obs_enc17 >= min_date_bepi:
            ax.plot(*coord_to_polar(bepi_coord),'v', markersize=10 ,label='BepiColombo', color='violet',alpha=0.6)
        #if date_obs_enc17 >= min_date_bepi_traj:
            #ax.plot(*coord_to_polar(bepi_coord_traj.transform_to(earth_coord)), label='BepiColombo  -1/+1 day', color='violet', linestyle='dashed')
        #ax.plot(*coord_to_polar(bepi_coord_traj),'-', color='violet', label='BepiColombo (as seen from Earth)')
    if parse_time(date_obs_enc17) > min_date_solo:
        if 'SOLO' in selected_sc:    
            # if date_obs_enc17 >= min_date_solo:
                ax.plot(*coord_to_polar(solo_coord),'v', markersize=10 ,label='SolO', color='black',alpha=0.6)
            # if date_obs_enc17 >= min_date_solo_traj:
            #    ax.plot(*coord_to_polar(solo_coord_traj.transform_to(earth_coord)),label='SoLO -5/+5 day', color='black', linestyle='dashed',  linewidth=1.5)

    # Visualizziamo gli overlap, se esistono
    if 'STA' and 'PSP' in selected_sc  and 'STA HI' and 'WISPR' in selected_his:
        #if date_obs_enc17 >= min_date_psp:
            if (start_date2 in psp_set) and (start_date2 in stereo_set):    
                if not overlap_wispr_stereo.is_empty:
                    x, y = overlap_wispr_stereo.exterior.xy        
                    ax.fill(np.arctan2(y, x), np.hypot(x, y), color='y', alpha=.15, label='Overlap WISPR & Stereo-HI') #red
                    date_overlap_wispr_stereohi.append(start_date2)
    if parse_time(date_obs_enc17) > min_date_solo:
        if 'STA' and 'SOLO' in selected_sc and 'STA HI' and 'SOLO HI' in selected_his:
            #if date_obs_enc17 >= min_date_solo:
                if (start_date2 in stereo_set) and (start_date2 in solo_set):
                    if not overlap_solo_stereo.is_empty:
                        x, y = overlap_solo_stereo.exterior.xy        
                        ax.fill(np.arctan2(y, x), np.hypot(x, y), color='y', alpha=.15, label='Overlap Stereo-HI & SolO-HI') #blue
                        date_overlap_stereohi_solohi.append(start_date2)

        if 'PSP' and 'SOLO' in selected_sc and 'WISPR' and 'SOLO HI' in selected_his:
            if (start_date2 in psp_set) and (start_date2 in solo_set):
                if not overlap_wispr_solo.is_empty:
                    x, y = overlap_wispr_solo.exterior.xy
                    ax.fill(np.arctan2(y, x), np.hypot(x, y), color='y', alpha=.15, label='Overlap WISPR & SolO-HI') #green
                    date_overlap_wispr_solohi.append(start_date2)
        if 'STA' and 'SOLO' and 'PSP' in selected_sc  and 'WISPR' and 'STA HI' and 'SOLO HI' in selected_his:                
            #if (start_date2 >= min_date_psp) and (start_date2 >= min_date_solo):            
                if (start_date2 in stereo_set) and (start_date2 in solo_set) and (start_date2 in psp_set) :
                        if not overlap_wispr_stereo_solo.is_empty:
                            x, y = overlap_wispr_stereo_solo.exterior.xy            
                            ax.fill(np.arctan2(y, x), np.hypot(x, y), color='green', alpha=.2, label='Overlap WISPR, Stereo-HI & SolO-HI')#orange
                            date_overlap_all.append(start_date2)
                    
    #plot_hi_geo=True




    if plot_hi_geo:   
        lamda=30
        #check for active CME indices from HIGeoCAT (with the lists produced above in this notebook)
        #check where time is identical to frame time
        #date_obs_enc17 = pd.to_datetime(date_obs_enc17, format='%Y-%m-%d %H:%M:%S')

        hc_time_num,hc_r,hc_lat,hc_lon,hc_id = read_higeocat()
        print(hc_time_num[0])
        print(mdates.date2num(date_obs_enc17))
        cmeind=np.where(hc_time_num == mdates.date2num(date_obs_enc17)) #frame_time_num+k*res_in_days)
        #print(cmeind)
        #plot all active CME circles
        print("hi_geo True")
        print("np.size(cmeind):", np.size(cmeind))

        for p in range(0,np.size(cmeind)):
            #print("size:", np.size(cmeind))
            #print("2-abs(hc_lat[cmeind[0][p]]/90): ", 2-abs(hc_lat[cmeind[0][p]]/90))
            #central d
            dire=np.array([np.cos(hc_lon[cmeind[0][p]]*np.pi/180),np.sin(hc_lon[cmeind[0][p]]*np.pi/180)])*hc_r[cmeind[0][p]]

            #points on circle, correct for longitude
            circ_ang = ((np.arange(111)*2-20)*np.pi/180)-(hc_lon[cmeind[0][p]]*np.pi/180)
            
            #these equations are from moestl and davies 2013
            xc = 0+dire[0]/(1+np.sin(lamda*np.pi/180)) + (hc_r[cmeind[0][p]]*np.sin(lamda*np.pi/180)/(1+np.sin(lamda*np.pi/180)))*np.sin(circ_ang)
            yc = 0+dire[1]/(1+np.sin(lamda*np.pi/180)) + (hc_r[cmeind[0][p]]*np.sin(lamda*np.pi/180)/(1+np.sin(lamda*np.pi/180)))*np.cos(circ_ang)

            #now convert to polar coordinates
            rcirc=np.sqrt(xc**2+yc**2)
            longcirc=np.arctan2(yc,xc)
        
            #else:
            ax.plot(longcirc,rcirc, c='tab:orange', ls='-', alpha=0.7, lw=2.0) 
            #print("cme helcats plotted")
            #plt.figtext(0.85, 0.90,'WP3 Catalogue (HELCATS) - SSEF30', fontsize=fsize, ha='right',color='tab:orange')
    
    if plot_donki:    
        hc_time_num1, hc_r1, hc_lat1, hc_lon1, hc_id1, a1_ell, b1_ell, c1_ell = read_donki()
        print(hc_time_num1[0])
        print(mdates.date2num(date_obs_enc17))
        cmeind1=np.where(hc_time_num1 == mdates.date2num(date_obs_enc17))
        print("DONKI True")
        print("np.size(cmeind1):", np.size(cmeind1))
        
        for p in range(0,np.size(cmeind1)):
            #print("size:", np.size(cmeind1))
            t = ((np.arange(201)-10)*np.pi/180)-(hc_lon1[cmeind1[0][p]]*np.pi/180)
            t1 = ((np.arange(201)-10)*np.pi/180)
            
            longcirc1 = []
            rcirc1 = []
            for i in range(3):

                xc1 = c1_ell[i][cmeind1[0][p]]*np.cos(hc_lon1[cmeind1[0][p]]*np.pi/180)+((a1_ell[i][cmeind1[0][p]]*b1_ell[i][cmeind1[0][p]])/np.sqrt((b1_ell[i][cmeind1[0][p]]*np.cos(t1))**2+(a1_ell[i][cmeind1[0][p]]*np.sin(t1))**2))*np.sin(t)
                yc1 = c1_ell[i][cmeind1[0][p]]*np.sin(hc_lon1[cmeind1[0][p]]*np.pi/180)+((a1_ell[i][cmeind1[0][p]]*b1_ell[i][cmeind1[0][p]])/np.sqrt((b1_ell[i][cmeind1[0][p]]*np.cos(t1))**2+(a1_ell[i][cmeind1[0][p]]*np.sin(t1))**2))*np.cos(t)

                longcirc1.append(np.arctan2(yc1, xc1))
                rcirc1.append(np.sqrt(xc1**2+yc1**2))

            ax.plot(longcirc1[0],rcirc1[0], color='tab:blue', ls='-', alpha=0.5, lw=2.0) #2-abs(hc_lat1[cmeind1[0][p]]/100)
            #print("cme donki plotted")
            ax.fill_between(longcirc1[2], rcirc1[2], rcirc1[1], color='tab:blue', alpha=.08)  #comment not to have the error
            #plt.figtext(0.02, 0.080,'DONKI (CCMC) - ELEvo', fontsize=fsize, ha='right',color='tab:blue')
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
            #plt.figtext(0.02, 0.060-p*0.02,f"CME {p+1}", fontsize=fsize, ha='right',color='tab:brown')

    
    ax.set_title(start_date2)

    create_custom_legend(ax)
    plt.tight_layout()

    return fig



with col2:


    st.markdown(
            """
            <h1 style='font-size: 40px;'>Welcome to Cor-HI Explorer!</h1>
            <l3 style='font-size: 20px;'>
            This app enables you to explore spacecraft constellations, visualize instrument fields of view, check data availability for coronagraphs and heliospheric imagers, and propagate either a catalog of CMEs or your own CME data. We hope this tool helps you explore heliospheric events with ease!

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
        
    start_date_t = datetime.strptime(st.session_state["t_start2"], "%Y-%m-%d %H:%M:%S")
    end_date_t = datetime.strptime(st.session_state["t_end2"], "%Y-%m-%d %H:%M:%S")
    # Calcola la differenza tra le due date
    delta = end_date_t - start_date_t
    if "t_end2" not in st.session_state:
        st.session_state["intervals_lenght"] = delta.total_seconds() / (30 * 60)

    if time_cadence == "30 min":
        st.session_state["intervals_lenght"] = delta.total_seconds() / (30 * 60)
    elif time_cadence == "1 hrs":
        st.session_state["intervals_lenght"] = delta.total_seconds() / (1 * 60 * 60)  # 1 hour in seconds
    elif time_cadence == "2 hrs":
        st.session_state["intervals_lenght"] = delta.total_seconds() / (2 * 60 * 60)  # 2 hours in seconds
    elif time_cadence == "6 hrs":
        st.session_state["intervals_lenght"] = delta.total_seconds() / (6 * 60 * 60)  # 6 hours in seconds
    elif time_cadence == "12 hrs":
        st.session_state["intervals_lenght"] = delta.total_seconds() / (12 * 60 * 60)  # 12 hours in seconds

    #print(intervals_30_min)   
    if st.button('Generate the plots'):
        st.session_state.plot_files.clear()  # Clear previous plots
        start_time_make_frame = time.time() 
        figures = []
        paths_to_fig = []
        loading_message = st.empty()
        # Display a single statement (this will remain constant throughout the process)

        progress_bar = st.progress(0)
        for interval in range(int(st.session_state["intervals_lenght"])+1):
            title = datetime.strptime(st.session_state["t_start2"], "%Y-%m-%d %H:%M:%S")  + timedelta(hours= cad * interval)     
            try:
                # Create the plot
                fig = make_frame(interval)  # Replace with your actual plotting function     

                file_path = os.path.join('images', f"{title}_sc_constellation.png")      
                paths_to_fig.append(file_path)       
                # Save the figure to the file and add to plot_files               
                fig.savefig(file_path)
                plt.close(fig)
                st.session_state.plot_files.append(file_path)
                
                progress_bar.progress((interval + 1) / (int(st.session_state["intervals_lenght"]) + 1))
            except Exception as e:
                        st.error(f"Error generating plot for {title}: {e}")
        total_time = np.round((time.time() - start_time_make_frame), 2)
        loading_message.markdown(f"<p style='color: green; font-size: 14px;'>Plot generation completed in {total_time} seconds</p>", unsafe_allow_html=True)
        #print('time make frame in minutes: ',np.round((time.time()-start_time_make_frame)/60))
        #print(paths_to_fig)
        print("Making animation...")
        gif_buffer = create_gif_animation(paths_to_fig, duration=200)  # Adjust duration for speed

        # Display the GIF animation in Streamlit
        st.image(gif_buffer)
        st.warning("Archive data is updated monthly. Last update: September 30, 2024.")
       # with tempfile.NamedTemporaryFile(suffix=".mp4") as tmpfile:
       #     ani.save(tmpfile.name, writer="ffmpeg")
        #    st.video(tmpfile.name)

        # Extract the base64 video content
        #video_base64 = video_html.split("base64,")[1].split('"')[0]

        # Wrap the video in custom HTML with controls
        #custom_html = f"""
        #<div style="display: flex; justify-content: center;">
        #    <video width="600" height="400" controls id="video-player" style="border:1px solid black;">
        #        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        #        Your browser does not support the video tag.
        #    </video>
        #</div>

        #<script>
        #    const videoPlayer = document.getElementById('video-player');

        #    // Add keyboard shortcuts for video controls
        #    document.addEventListener('keydown', function(event) {{
        #        if (event.code === 'Space') {{
        #            if (videoPlayer.paused) {{
        #                videoPlayer.play();
        #            }} else {{
        #                videoPlayer.pause();
        #            }}
        #        }} else if (event.code === 'ArrowRight') {{
        #            videoPlayer.currentTime += 1;  // Go forward 1 second
        #        }} else if (event.code === 'ArrowLeft') {{
        #            videoPlayer.currentTime -= 1;  // Go backward 1 second
        #        }}
        #    }});
        #</script>
        #"""

        # Display the video in Streamlit
        #st.components.v1.html(custom_html, height=700)


        # Display in Streamlit
        #st.components.v1.html(video_html, height=400)
        #output_dir = 'animations'
        #os.makedirs(output_dir, exist_ok=True)

        # Save the animation as MP4 and GIF
        #mp4_file_path = os.path.join(output_dir, 'animation.mp4')
        #gif_file_path = os.path.join(output_dir, 'animation.gif')

        # Save the MP4 animation
        #ani.save(mp4_file_path, writer='ffmpeg')

        # Save the GIF animation
        #ani.save(gif_file_path, writer='imagemagick')

        # Display the animation in Streamlit
        #st.video(ani)
        #if st.session_state.plot_files:
        #    selected_plot = st.slider('Select Plot', 0, len(st.session_state.plot_files) - 1, 0)
        #    image_path = st.session_state.plot_files[selected_plot]
        #    st.image(image_path, use_column_width=True) #caption=f"Plot for {st.session_state.dates_stereohi_fov2[selected_plot].strftime('%Y-%m-%dT%H-%M-%S')}", use_column_width=True)
        
        # Create download buttons for the saved files
#        with open(gif_file_path, "rb") as f:
#            gif_data = f.read()

 #       with open(mp4_file_path, "rb") as f:
#            mp4_data = f.read()

#        st.download_button(
#                label="Download as GIF",
#                data=gif_data,
#            file_name='animation.gif',
#                mime='image/gif'
#            )

#        st.download_button(
#            label="Download as MP4",
#            data=mp4_data,
#                file_name='animation.mp4',
#                mime='video/mp4'
#            )
