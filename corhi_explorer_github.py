# S/C constellation with heliospheric imagers FoV overlaps and CME catalogs.

#Author: Greta M. Cappello (University of Graz, Austria)

#Updates:
#  August 30, 2024 : 1st Working version code.
#  Jan 15, 2025 : Update catalogs and observation dates of insturments. Update logo.

#Comments: This code is used to obtain the fov of the three heliospheric imagers (HI) available to date (SoloHI, WISPR, STEREO-A HI). It generates a S/C constellation with the FoV of the HIs, only for observational periods., Otherwise only the s/c location is plotted, since the event may have hit the detector. In addition, the overlap of the FoV is highlighted to show when 2 or 3 HIs may have seen the same event. It is also possible to obtain an observational and a FoV overlap calendar. Once the overlap calendar is done, how many CMEs are entering the overlap? Are there catalog-recorded events which are occuring during the fovs overlap? This tool wants to be a way to connect the observations available from 3 different s/c at different view point. 

import streamlit as st
import gdown
import time
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
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
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

from shapely.geometry import Polygon, GeometryCollection, MultiPoint, Point

from shapely.geometry.polygon import orient


from matplotlib.markers import MarkerStyle
import multiprocessing
from multiprocessing import Pool
import subprocess
from matplotlib.animation import FuncAnimation
from matplotlib import image as mpimg


path_local_greta = './'
overview_path = path_local_greta
path_to_logo = path_local_greta
st.set_page_config(page_title="CORHI-X",page_icon=path_to_logo+"corhiX_pictogram.png", layout="wide")


def reader_txt(file_path):
    times_obs = []
    with open(file_path, 'r') as f:
        for line in f:
            date_str = line.strip()
            date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            times_obs.append(date_obj)
    return times_obs


def download_from_gd(file_data_url, data_url):
            time.sleep(1)
            #status.text(f"Downloading file {file_data_url}/7...")
            if os.path.exists(file_data_url):
                os.remove(file_data_url)
            if not os.path.exists(file_data_url):
                # If it does not exist, download the file
                #st.write(f"Downloading the file {file_data_url}...")
                gdown.download(data_url, file_data_url, quiet=False,fuzzy=True)





with st.spinner('Starting CORHI-X....'):
    @st.cache_data()
    def starters():
        url_donki = 'https://drive.google.com/file/d/1pPlbsvjE6GaE2I6gGWcEC5Axls04cQmm/view?usp=sharing'
        url_higeo = 'https://drive.google.com/file/d/17RKgjSjk96O7RpcTd2MMs3584rec67tZ/view?usp=sharing'
        url_C2 = 'https://drive.google.com/file/d/1lhMrhCXpJNS1FOlIPLcSrcbnTMLpspSR/view?usp=sharing'
        url_cor1 = 'https://drive.google.com/file/d/1wTRUgwWqtkKbjLgW52WZxZW3T1jvb4Cy/view?usp=sharing'
        url_metis = 'https://drive.google.com/file/d/1GogqQFdtTTIrcLWDmdUROXBZo54jbDFq/view?usp=sharing'
        url_hi1A = 'https://drive.google.com/file/d/1W9XGlIUI4cuyIQGKb6Xizi0oZP9L1zOI/view?usp=sharing'
        url_solohi ='https://drive.google.com/file/d/1gB40XdR2Vr3M9K9pD_iHLXwvZwDAb_tt/view?usp=sharing'
        url_wispr = 'https://drive.google.com/file/d/14r2Vid2-OHs5oJDuzc1VvtYNVFEtTWCK/view?usp=sharing'

        kinematic_donki_file_f = path_local_greta + "donki_kinematics_2019_now.p"
        kinematic_higeo_file_f = path_local_greta + "higeocat_kinematics.p"
        file_date_c2 = path_local_greta + "c2_custom_intervals.txt"
        file_date_cor1 = path_local_greta + "cor1_custom_intervals.txt"
        file_date_metis = path_local_greta + "metis_custom_intervals.txt"
        file_date_hi1A = path_local_greta + "hi1A_custom_intervals.txt"
        file_date_solohi= path_local_greta + "solohi_custom_intervals.txt"
        file_date_wispr = path_local_greta + "wispr_custom_intervals.txt"

        
            #else:
                #st.write(f"Folder {file_data_url} already exists. No need to download.")


        download_from_gd(kinematic_donki_file_f, url_donki)
        download_from_gd(kinematic_higeo_file_f, url_higeo)
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



        # HIs
        times_wispr_obs_f = reader_txt(path_wispr_dates)
        times_solohi_obs_f = reader_txt(path_solohi_dates)
        times_hi1A_obs_f = reader_txt(path_hi1A_dates)

        # CORs
        times_metis_obs_f = reader_txt(path_metis_dates)
        times_cor1_obs_f = reader_txt(path_cor1_dates)
        times_c2_obs_f = reader_txt(path_c2_dates)

        return kinematic_donki_file_f, kinematic_higeo_file_f, times_wispr_obs_f, times_solohi_obs_f, times_hi1A_obs_f, times_metis_obs_f, times_cor1_obs_f, times_c2_obs_f
    @st.cache_data()
    def read_donki(file_d):
        [hc_time_num1_date, hc_r1_func, hc_lat1_func, hc_lon1_func, hc_id1_func, a1_ell_func, b1_ell_func, c1_ell_func]=pickle.load(open(file_d, "rb")) 
        hc_time_num11 = mdates.date2num(hc_time_num1_date)
        return hc_time_num11, hc_r1_func, hc_lat1_func, hc_lon1_func, hc_id1_func, a1_ell_func, b1_ell_func, c1_ell_func

    @st.cache_data()
    def read_higeocat(file_e):
        [hc_time_num_func,hc_r_func,hc_lat_func,hc_lon_func,hc_id_func]=pickle.load(open(file_e, "rb"))
        hc_time_num22 = mdates.date2num(hc_time_num_func)
        return hc_time_num22,hc_r_func,hc_lat_func,hc_lon_func,hc_id_func



col1, col2 = st.columns([1, 2])

with col1:
    kinematic_donki_file, kinematic_higeo_file, times_wispr_obs, times_solohi_obs, times_hi1A_obs, times_metis_obs, times_cor1_obs, times_c2_obs = starters()
    hc_time_num1, hc_r1, hc_lat1, hc_lon1, hc_id1, a1_ell, b1_ell, c1_ell = read_donki(kinematic_donki_file)
    hc_time_num,hc_r,hc_lat,hc_lon,hc_id = read_higeocat(kinematic_higeo_file)
    
    def write_file_cme():
        if not st.session_state.data:
            st.warning("No CME data available.")
            return

        # Convert session state data into a DataFrame
        df = pd.DataFrame(st.session_state.data)
        st.write("Dataframe User CMEs:", df)

        start_time = time.time()
        used = 1  # Adjust based on your machine
        #print('Using multiprocessing, nr of cores', mp.cpu_count(), ', nr of processes used: ', used)

        # Create the pool for multiprocessing
        pool = mp.get_context('fork').Pool(processes=used)

        # Map the worker function onto the parameters
        results = pool.map(cme_kinematics, range(len(st.session_state.data)))

        # Close the pool and wait for the work to finish
        pool.close()
        pool.join()

        #for j in range(len(st.session_state.data)):
            #cme_kinematics(j)



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
        halfwidth = np.deg2rad(st.session_state.data[i]['half angle'])+ np.arcsin(st.session_state.data[i]['kappa'])
        print(f"Half angle CME{i}",st.session_state.data[i]['half angle'])
        print(f"k CME{i}",st.session_state.data[i]['kappa'])
        print(f"delta CME{i}",np.rad2deg(np.arcsin(st.session_state.data[i]['kappa'])))
        print(f"half-width CME{i}", st.session_state.data[i]['half angle'] + np.rad2deg(np.arcsin(st.session_state.data[i]['kappa'])))
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
    #st.image(path_to_logo+"logo_corhix_white_border.png" , width=400)

    query_params = {}
    for key in st.query_params.keys():
        query_params[key] = st.query_params.get_all(key)
    
    print("TEST DATE URL:", datetime(query_params["date"][0]))

        
    #st.header("🔍 **Select the interval of time**")
    st.markdown("<h4 style='color: magenta;'>🔍 Select the interval of time</h4>", unsafe_allow_html=True)

    # Set up date selector
    try:
        default_date = datetime.strptime(query_params["date"][0], "%Y-%m-%d").date()
    except (KeyError, ValueError):
        default_date = datetime(2023, 10, 1).date()  # fallback default
    
    selected_date = st.date_input(
        "Select Initial Date:",
        default_date,
        help= "Select the initial date, starting from Jan. 2019, that you would like to use for your analysis. Either you write it in the format YYYY/MM/DD or you select it using the pop-up calendar.")

    # Set up 30-minute intervals as options
    time_options = [(datetime.min + timedelta(hours=h, minutes=m)).strftime("%H:%M") 
                    for h in range(24) for m in (0, 30)]
 
    selected_time = st.selectbox("Select Initial Time:", time_options, help= "Select the initial time you would like to use for your analysis. Either you write it in the format HH:00 (or HH:30) or you select it using the pending menu. Only times at 30 mins cadence are accepted.")

    
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
        interval_choice = st.radio("Select:", ["+1 Day", "+5 Days", "+20 Days", "Add Hours"], help = "Select the interval of time you would like to explore starting from the initial date/time you selected in the previous input box. If you  would like to visualize the plot for a single time-stamp, we recommend to use 'Add Hours' with an input equal to 1.")

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
        
        if datetime.strptime(st.session_state["t_start2"], "%Y-%m-%d %H:%M:%S").year > 2019:
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
            value="6 hrs",  # Default selection,
            help = 'Define the cadence you  would like to have between the plots that will be produced for the interval of time you selected in previous input boxes.'
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
                "Let me select S/C and FoV"), 
                help = "Two options are available for the visualization of the data. You can either plot the FoVs of all instruments simultaneously by selecting the 'Plot all S/C and all instruments FoV' option, or select specific spacecraft and separately plot the FoVs of their coronagraphs and heliospheric imagers using the 'Let me select S/C and FoV' option. Note that only when data is available in the archive of the instruments the FoV is shown.")

    if option == "Plot all S/C and all instruments' FoV":
        selected_sc = ["SOHO", "STA", "PSP", "SOLO", "BEPI"]
        selected_coronagraphs = ["C2-C3", "COR1-COR2", "METIS"]
        selected_his = ["WISPR", "STA HI", "SOLO HI"]

    elif option =="Let me select S/C and FoV":

        st.markdown("<h4 style='color: magenta;'>Select spacecraft </h4>", unsafe_allow_html=True)
        sc_options = ["SOHO", "STA", "PSP", "SOLO", "BEPI"]
        selected_sc = st.multiselect("Select spacecraft:", sc_options)

        st.markdown("<h4 style='color: magenta;'>Show FoVs coronagraphs</h4>", unsafe_allow_html=True)
        coronagraph_options = []
        if "SOHO" in selected_sc:
            coronagraph_options.append("C2-C3")
        if "STA" in selected_sc: 
            coronagraph_options.append("COR1-COR2")
        if "SOLO" in selected_sc: 
            coronagraph_options.append("METIS")
        selected_coronagraphs = st.multiselect("Select coronagraphs:", coronagraph_options)
    
        st.markdown("<h4 style='color: magenta;'>Show FoVs HIs</h4>", unsafe_allow_html=True)
        his_options = []
        if "PSP" in selected_sc:
            his_options.append("WISPR")
        if "STA" in selected_sc: 
            his_options.append("STA HI")
        if "SOLO" in selected_sc: 
            his_options.append("SOLO HI")
        selected_his = st.multiselect("Select HIs:", his_options)

    overlap_fov = st.checkbox("Overlap FoVs", help = "Check the box 'Overlap FoVs' in order to visualize the shaded areas in yellow and green, showing respectively the overlapping FoVs of two or three heliospheric imagers. Otherwise, if 'Overlap FoVs' is not selected just the FoVs will be plotted when the data is available in the archives.")
    lines_draw = st.checkbox("Draw connecting lines S/C-Sun", help = "The option 'Draw connecting lines S/C-Sun' allows to show a line connecting each spacecraft to the Sun. Note: It is not a connectivity tool, it is just a geometrical line to highlight the plane of the sky of the coronagraphs.")


    st.markdown("<h4 style='color: magenta;'>Select Catalog (optional)</h4>", unsafe_allow_html=True, help = "CORHI-X not only allows us to plot FoV overlaps for data available in the online archives but also to visualize how many transient events may have entered those FoVs.The CME information is taken from already existing catalogs or is defined by the user.")
    plot_hi_geo = st.checkbox("Plot HI-GEO/SSEF catalog", help = 'The HIGeoCAT catalogue (Barnes et al. 2019) derives the kinematics of the CMEs from STA/HI-A observations using geometric fitting techniques (Davies et al. 2013). For the visualization, the CMEs are propagated linearly outward as semicircles with a half-angle of 30°. Note that due to the FoV of STA/HI-A, HiGeoCAT has limited coverage of CME detection compared to DONKI. ')
    plot_donki = st.checkbox("Plot DONKI/ELEvo catalog", help = 'The Space Weather Database Of Notifications, Knowledge, Information (DONKI) catalog is provided by the Moon to Mars (M2M) Space Weather Analysis Office and hosted by the Community Coordinated Modeling Center (CCMC). The kinematic properties of the CMEs given in DONKI are derived from coronagraph observations using the CME Analysis Tool of the Space Weather Prediction Center (SWPC CAT, Millward et. al 2013). The ELliptical Evolution Model (ELEvo; Möstl et al. 2015) is used to visualize the propagation of the CMEs through the Heliosphere. The ELEvo model assumes an elliptical front for the CMEs and includes a simple drag-based model (Vrsnak et al. 2013).')
    plot_cme = st.checkbox("Plot user CMEs", help = 'You can insert up to 6 user CMEs that are then propagated radially outward using a simple drag-based model (Vrsnak et al. 2013). For each CME the user is requested to insert the Graduated Cylindrical Shell (GCS) model parameters, see Thernisien et al. (2006, 2009, 2011).')

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
                k = st.number_input(f"Enter k for CME {i + 1}", min_value=0.0, max_value=1.0, key=f"k_{i}", value = 0.30, help="κ is the aspect ratio of the GCS modelled flux-rope, as described in Thernisien et al.(2006, 2009, 2011). This parameter sets the rate of expansion versus the height of the CME, so the structure expands in a self-similar way.")
                alpha = st.number_input(f"Enter alpha for CME {i + 1}", min_value=0.0, max_value=75.0, key=f"alpha_{i}", value = 45.0, help="The half-angle, α, is the angle between the axis of the leg and the y-axis.")
                longitude = st.number_input(f"Enter HGS longitude (-180°, 180°) for CME {i + 1} ", min_value=-180.0, max_value=180.0, key=f"longitude_{i}", value = 120.0, help="Heliographic Stonyhurst (HGS) longitude of the CME source location.")
                v = st.number_input(f"Enter speed (km/s) for CME {i + 1}", min_value=50.0, max_value=3000.0, key=f"v_{i}", value = 900.0, help = "Initial speed of the CME in km/s at 21.5 Rsun.")
                t_0 = st.text_input(f"Enter time at 21.5 Rsun(YYYY-MM-DD H:M:S) for CME {i + 1}", st.session_state["t_start2"], help = "Time at which the CME reaches 21.5 solar radii. Format: YYYY-MM-DD HH:MM:SS.")

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
                        'half angle': params[1], 
                        'kappa':params[0]
                    }
                    for i, params in enumerate(cme_params_list)
                ]
                st.write("CME parameters submitted! Calculating kinematics...")

        
                write_file_cme()

            else:
                #st.write("No data collected. Be sure to submit the parameters!")
                st.warning("No data collected. Be sure to submit the parameters!")
# Play/Pause functionality




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




def coord_to_polar(coord):
    return coord.lon.to_value('rad'), coord.radius.to_value('AU')

def create_custom_legend(ax):
    # Define Field of View (FoV) lines
    fov_legend = []
    if "METIS" in selected_coronagraphs:
        fov_legend.append(mlines.Line2D([], [], color='orange', linewidth=1, label='METIS FoV'))
    if "C2-C3" in selected_coronagraphs:
        fov_legend.append(mlines.Line2D([], [], color='green', linewidth=1, label='C3 FoV'))
    if "COR1-COR2" in selected_coronagraphs:
        fov_legend.append( mlines.Line2D([], [], color='magenta', linewidth=1, label='COR2 FoV'))
    if "SOLO HI" in selected_his:
        fov_legend.append(mlines.Line2D([], [], color='black', linewidth=1, label='SolO-HI FoV'))
    if "WISPR" in selected_his:
        fov_legend.append(mlines.Line2D([], [], color='blue', linewidth=1, label='WISPR-I FoV'))
    if "STA HI" in selected_his:
        fov_legend.append (mlines.Line2D([], [], color='brown', linewidth=1, label='STEREO-A HI FoV'))
    
    # Define objects with markers
    object_legend = []
    object_legend.append(mlines.Line2D([], [], color='yellow', marker='.', markersize=7, linestyle='None', label='Sun'))
    object_legend.append(mlines.Line2D([], [], color='skyblue', marker='o', markersize=7, linestyle='None', label='Earth', alpha=0.6))
    if "PSP" in selected_sc:
        object_legend.append(mlines.Line2D([], [], color='blue', marker='v', markersize=7, linestyle='None', label='PSP', alpha=0.6))
    if "STA" in selected_sc:
        object_legend.append(mlines.Line2D([], [], color='brown', marker='v', markersize=7, linestyle='None', label='STEREOA', alpha=0.6))
    if "SOHO" in selected_sc:   
        object_legend.append(mlines.Line2D([], [], color='green', marker='v', markersize=7, linestyle='None', label='SOHO', alpha=0.6))
    if "BEPI" in selected_sc: 
        object_legend.append(mlines.Line2D([], [], color='violet', marker='v', markersize=7, linestyle='None', label='BepiColombo', alpha=0.6))
    if "SOLO" in selected_sc: 
         object_legend.append(mlines.Line2D([], [], color='black', marker='v', markersize=7, linestyle='None', label='SolO', alpha=0.6))


    custom_legend_items = []
    if plot_hi_geo:
        custom_legend_items.append(mlines.Line2D([0], [0], color='tab:orange', lw=1, label='HI-GEO/SSEF'))
    if plot_donki:
       custom_legend_items.append(mlines.Line2D([0], [0], color='tab:blue', lw=1, label='DONKI/ELEvo'))
    if plot_cme:
        custom_legend_items.append(mlines.Line2D([0], [0], color='tab:brown', lw=1, label='CME'))

    overlap_legend = [] 
    # Define overlap area
    if overlap_fov:
        overlap_legend.append(mlines.Line2D([], [], color='y', alpha=0.15, linewidth=8, label='Overlap HI FoVs'))
    
    
    # Combine all legend items
    all_legend_items = fov_legend + object_legend + overlap_legend + custom_legend_items
    
    # Add the legend to the plot
    #ax.legend(handles=all_legend_items, loc='upper left', title='Legend')
    ax.legend(
        handles=all_legend_items, 
        loc='lower center', 
        bbox_to_anchor=(0.8, -0.2),  # Position at the bottom right
        borderaxespad=0,
        fontsize=7,       # Smaller text size
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

def get_coordinates(spacecraft, start_time_sc, end_time_sc, minimum_time, cadence = '30m'):
    # Parse start and end times to ensure datetime format
    start_time_sc = parse_time(start_time_sc)
    end_time_sc = parse_time(end_time_sc)
    minimum_time = parse_time(minimum_time)

    # Proceed to query if start time is within acceptable range
    start_time_query = max(start_time_sc, minimum_time)

    sc_coordinates = get_horizons_coord(spacecraft,
                                            {'start': start_time_query.strftime("%Y-%m-%d %H:%M:%S"),
                                            'stop': end_time_sc.strftime("%Y-%m-%d %H:%M:%S"),
                                            'step': cadence})
    return sc_coordinates


    
min_date_solo = datetime(2020, 2, 10, 10, 0, 0)
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
    print(len(psp_coord_func))
    if start_time_sc < min_date_solo and end_time_sc < min_date_solo:
        solo_coord_func = np.ones(len(psp_coord_func))
        print(solo_coord_func)
    elif start_time_sc < min_date_solo and end_time_sc >= min_date_solo:
        solo_coord_func2= get_coordinates("Solar Orbiter", min_date_solo, end_time_sc, min_date_solo,cadence = '30m')
        n = len(psp_coord_func)-len(solo_coord_func2)
        solo_coord_func = np.concatenate((np.ones(n), solo_coord_func2))    
        print(solo_coord_func)
    elif start_time_sc >= min_date_solo:
        solo_coord_func= get_coordinates("Solar Orbiter", start_time_sc, end_time_sc, min_date_solo,cadence = '30m')
        print(solo_coord_func)
    bepi_coord_func = get_coordinates("BepiColombo", start_time_sc, end_time_sc, min_date_bepi, cadence = '30m')
    soho_coord_func = get_coordinates("SOHO", start_time_sc, end_time_sc, min_date_soho, cadence = '30m')
    sta_coord_func = get_coordinates("STEREO-A", start_time_sc, end_time_sc, min_date_stereo, cadence = '30m')
    julian_date = psp_coord_func.obstime.jd
    gregorian_datetime = Time(julian_date, format='jd').to_datetime()
    print(gregorian_datetime)

    return psp_coord_func, solo_coord_func, bepi_coord_func, soho_coord_func, sta_coord_func



def fov_to_polygon(angles, radii):
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return Polygon(np.column_stack((x, y)))

def circle_creator():
    radius_au = 1.0  # Raggio in AU
    num_points = 100  # Numero di punti per approssimare il cerchio
    angles = np.linspace(0, 2 * np.pi, num_points)
    x_circle = radius_au * np.cos(angles)
    y_circle = radius_au * np.sin(angles)
    return Polygon(zip(x_circle, y_circle))

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
    fig= plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='polar')
    ax.set_xticks(np.pi/180. * np.linspace(0,  360, 12, endpoint=False))

    ax.set_ylim(0,1.1)

    initial_datatime = datetime.strptime(st.session_state["t_start2"], "%Y-%m-%d %H:%M:%S")
    final_datatime = datetime.strptime(st.session_state["t_end2"], "%Y-%m-%d %H:%M:%S")
    #print(initial_datatime + timedelta(hours=1 * ind))
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

    
    date_obs_enc17 = start_date2


    if ind == 0:
        j = ind
    else:
        if time_cadence == "30 min":
            j =  ind
        if time_cadence == "1 hrs":
            j =  ind * 2
        if time_cadence == "2 hrs":
            j =  ind * 4
        if time_cadence == "6 hrs":
            j =  ind * 12
        if time_cadence == "12 hrs":
            j =  ind * 24

    #if initial_datatime >= min_date_psp or final_datatime>=min_date_psp:
    #    psp_coord_array = get_coordinates(initial_datatime, final_datatime)
    #    psp_coord = psp_coord_array[ind]
    #    ax.plot(*coord_to_polar(psp_coord),'v',label='PSP', color='blue',alpha=0.6)
    

#!!!A condition must be placed to consider times before 2019, easily put a condition in start time
#*****************************
#POSITION SPACECRAFT 
#*****************************
    psp_coord_array, solo_coord_array, bepi_coord_array, soho_coord_array, sta_coord_array = get_all_coordinates(initial_datatime,final_datatime)

    #psp_coord_array, solo_coord_array, bepi_coord_array, soho_coord_array, sta_coord_array = get_all_coordinates(initial_datatime,final_datatime)
    psp_coord = psp_coord_array[j]
    julian_date = psp_coord.obstime.jd
    gregorian_datetime = Time(julian_date, format='jd').to_datetime()
    
    #print(start_date2)
    print(f"Gregorian datetime: {gregorian_datetime}")
    #print(psp_coord)
    
    r=psp_coord.radius
    #solo_coord = solo_coord_array[ind]
    solo_coord = solo_coord_array[j]
    #print(psp_coord_array)
    #print(solo_coord_array)
    bepi_coord = bepi_coord_array[j]
    soho_coord = soho_coord_array[j]
    stereo_coord = sta_coord_array[j]
    
    print("psp coords:", psp_coord)
    print(len(psp_coord_array))
    print("soho coords:", soho_coord)
    print("stereo coords:", stereo_coord)
    print("solo coords:", solo_coord)
    print(len(solo_coord_array))
    print("bepi coords:", bepi_coord)
    #sun_coord = (0, 0, 0)

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
    if parse_time(date_obs_enc17) >= min_date_solo:
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
                fov_plotter_cori(solo_coord, 3.1, 'METIS', 'orange')

    if 'C2-C3' in selected_coronagraphs:
        if (start_date2 in c2_set):
            fov_plotter_cori(soho_coord, 8.16, 'C3', 'green')
    
    if 'COR1-COR2' in selected_coronagraphs:
        if (start_date2 in cor1_set):
            fov_plotter_cori(stereo_coord, 4.16, 'COR2', 'magenta')

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


    # Crea il cerchio come un oggetto Polygon
    circle_1AUcut = circle_creator()

    # Funzione di utilità per convertire i contorni del FoV in un poligono Shapely
    if 'WISPR' in selected_his and 'PSP' in selected_sc:
        if date_obs_enc17 >= min_date_psp:
            if (start_date2 in psp_set):
            # Convertiamo i FoV in poligoni
                polygon_wispr_i = circle_1AUcut.intersection(fov_to_polygon(np.concatenate((fov1_angles, fov2_angles_outer[::-1])), np.concatenate((fov1_ra, fov2_ra_outer[::-1]))))
            #polygon_wispr_o = fov_to_polygon(np.concatenate((fov1_angles_outer, fov2_angles_outer[::-1])), np.concatenate((fov1_ra_outer, fov2_ra_outer[::-1])))
    
    if 'STA HI' in selected_his and 'STA' in selected_sc:    
        if (start_date2 in stereo_set):
            polygon_stereo_hi = circle_1AUcut.intersection(fov_to_polygon(np.concatenate((fov1_angles_stAi, fov2_angles_stAi[::-1])), np.concatenate((fov1_ra_stAi, fov2_ra_stAi[::-1]))))

    if 'SOLO HI' in selected_his and 'SOLO' in selected_sc:    
        if date_obs_enc17 >= min_date_solo:
            if (start_date2 in solo_set):
                polygon_solo_hi = circle_1AUcut.intersection(fov_to_polygon(np.concatenate((fov1_angles_shi, fov2_angles_shi[::-1])), np.concatenate((fov1_ra_shi, fov2_ra_shi[::-1]))))

    if 'WISPR' and 'STA HI' in selected_his and 'PSP' and 'STA' in selected_sc:
        if date_obs_enc17 >= min_date_psp:
        # Calcoliamo le intersezioni tra i poligoni
            if (start_date2 in psp_set) and (start_date2 in stereo_set):
                overlap_wispr_stereo = circle_1AUcut.intersection(polygon_wispr_i.intersection(polygon_stereo_hi))
    else: 
        overlap_wispr_stereo = GeometryCollection()

    if 'SOLO HI' and 'STA HI' in selected_his and 'SOLO' and 'STA' in selected_sc:
        if date_obs_enc17 >= min_date_solo:
        # Calcoliamo le intersezioni tra i poligoni
            if (start_date2 in solo_set) and (start_date2 in stereo_set):
                overlap_solo_stereo = circle_1AUcut.intersection(polygon_solo_hi.intersection(polygon_stereo_hi))
    else: 
        overlap_solo_stereo = GeometryCollection()    #it gives an empty intersection when i de-select e.g. sta


    if 'WISPR' and 'STA HI' and 'SOLO HI' in selected_his and 'PSP' and 'STA' and 'SOLO' in selected_sc:               
        if date_obs_enc17 >= min_date_solo:
            if (start_date2 in psp_set) and (start_date2 in stereo_set) and (start_date2 in solo_set):   
                overlap_wispr_stereo_solo = circle_1AUcut.intersection(overlap_wispr_stereo.intersection(polygon_solo_hi))
    else: 
            overlap_wispr_stereo_solo = GeometryCollection()      

    if date_obs_enc17 >= min_date_solo:
        if 'WISPR' and 'SOLO HI' in selected_his and 'PSP' and 'SOLO' in selected_sc:
            if (start_date2 in psp_set) and (start_date2 in solo_set):
                overlap_wispr_solo = circle_1AUcut.intersection(polygon_wispr_i.intersection(polygon_solo_hi))
    else: 
        overlap_wispr_solo = GeometryCollection()      
    
    ax.plot(0, 0, marker=".",markersize=10, label='Sun', color='yellow')
    ax.plot(*coord_to_polar(earth_coord), 'o', markersize=10 ,label='Earth', color='skyblue',alpha=0.6)
    if 'PSP' in selected_sc:
        if date_obs_enc17 >= min_date_psp:
            ax.plot(*coord_to_polar(psp_coord),'v', markersize=10 ,label='PSP', color='blue',alpha=0.6)
            if lines_draw:
                psp_polar_coords = coord_to_polar(psp_coord)
                ax.plot([0,  psp_polar_coords[0]], [0,  psp_polar_coords[1]], color='blue', linestyle='--', alpha=0.5)
        #if date_obs_enc17 >= min_date_psp_traj:
        #    ax.plot(*coord_to_polar(psp_coord_traj.transform_to(earth_coord)),label='PSP -5/+5 day', color='blue', linestyle='solid',  linewidth=1.5)
    if 'STA' in selected_sc:
        ax.plot(*coord_to_polar(stereo_coord),'v', markersize=10 ,label='STEREOA', color='brown',alpha=0.6)
        if lines_draw:
            stereo_polar_coords = coord_to_polar(stereo_coord)
            ax.plot([0,  stereo_polar_coords[0]], [0,  stereo_polar_coords[1]], color='brown', linestyle='--', alpha=0.5)
        #ax.plot(*coord_to_polar(stereo_coord_traj.transform_to(earth_coord)),label='STEREO A -1/+1 day', color='brown', linestyle='dashed',  linewidth=1.5)
    if 'SOHO' in selected_sc:
            ax.plot(*coord_to_polar(soho_coord),'v', markersize=10 ,label='SOHO', color='green',alpha=0.6)
            if lines_draw:
                soho_polar_coords = coord_to_polar(soho_coord)
                ax.plot([0,  soho_polar_coords[0]], [0,  soho_polar_coords[1]], color='green', linestyle='--', alpha=0.5)
        #    ax.plot(*coord_to_polar(soho_coord_traj.transform_to(earth_coord)),label='SOHO A -1/+1 day', color='green', linestyle='dashed',  linewidth=1.5)
    
    #ax.plot(*coord_to_polar(stereo_coord_traj),'-', color='brown', label='STEREOA (as seen from Earth)',  linewidth=1.5)
    #print(stereo_coord)
    if 'BEPI' in selected_sc:
        #if date_obs_enc17 >= min_date_bepi:
            ax.plot(*coord_to_polar(bepi_coord),'v', markersize=10 ,label='BepiColombo', color='violet',alpha=0.6)
            if lines_draw:
                bepi_polar_coords = coord_to_polar(bepi_coord)
                ax.plot([0,  bepi_polar_coords[0]], [0,  bepi_polar_coords[1]], color='violet', linestyle='--', alpha=0.5)

        #if date_obs_enc17 >= min_date_bepi_traj:
            #ax.plot(*coord_to_polar(bepi_coord_traj.transform_to(earth_coord)), label='BepiColombo  -1/+1 day', color='violet', linestyle='dashed')
        #ax.plot(*coord_to_polar(bepi_coord_traj),'-', color='violet', label='BepiColombo (as seen from Earth)')
    if parse_time(date_obs_enc17) >= min_date_solo:
        if 'SOLO' in selected_sc:    
            # if date_obs_enc17 >= min_date_solo:
                ax.plot(*coord_to_polar(solo_coord),'v', markersize=10 ,label='SolO', color='black',alpha=0.6)
                if lines_draw:
                    solo_polar_coords = coord_to_polar(solo_coord)
                    ax.plot([0,  solo_polar_coords[0]], [0,  solo_polar_coords[1]], color='black', linestyle='--', alpha=0.5)
    
    #points = []
    #id_points = []
    #cmeind1 = np.where(hc_time_num1 == mdates.date2num(date_obs_enc17))
    #if len(cmeind1[0]) != 0: #if I have lenght of the indexes != 0, I get the r and lon at that index
    #    r=hc_r1[0][cmeind1[0]]
    #    lon=hc_lon1[cmeind1[0]] 
    #    for p in range(len(cmeind1[0])):
    #            points.append(Point(r[p] * np.cos(np.radians(lon[p])), r[p] * np.sin(np.radians(lon[p]))))
    #            id_points.append(hc_id1[cmeind1[0][p]])
    #    print(points)
    #    print(id_points)
    #    for i_p, point in enumerate(points):          
    #        ax.plot(lon[i_p]*np.pi/180,r[i_p],'v', markersize=5 ,label='point', color='green',alpha=0.6) 

    # Visualizziamo gli overlap, se esistono
    if 'STA' and 'PSP' in selected_sc  and 'STA HI' and 'WISPR' in selected_his:
        #if date_obs_enc17 >= min_date_psp:
            if (start_date2 in psp_set) and (start_date2 in stereo_set):    
                if not overlap_wispr_stereo.is_empty:
                    x, y = overlap_wispr_stereo.exterior.xy  
                    if overlap_fov:        
                        ax.fill(np.arctan2(y, x), np.hypot(x, y), color='y', alpha=.15, label='Overlap WISPR & Stereo-HI') #red
                        date_overlap_wispr_stereohi.append(start_date2)
                        #for i_p, point in enumerate(points):    
                            #if overlap_wispr_stereo.contains(point):
                                #ax.plot(lon[i_p]*np.pi/180,r[i_p],'v', markersize=5 ,label='point', color='red',alpha=0.6)
                            #ax.plot(lon[i_p]+90,r[i_p],'v', markersize=5 ,label='point', color='red',alpha=0.6)
                        
    if parse_time(date_obs_enc17) >= min_date_solo:
        if 'STA' and 'SOLO' in selected_sc and 'STA HI' and 'SOLO HI' in selected_his:
            #if date_obs_enc17 >= min_date_solo:
                if (start_date2 in stereo_set) and (start_date2 in solo_set):
                    if not overlap_solo_stereo.is_empty:
                        x, y = overlap_solo_stereo.exterior.xy  
                        if overlap_fov:        
                            ax.fill(np.arctan2(y, x), np.hypot(x, y), color='y', alpha=.15, label='Overlap Stereo-HI & SolO-HI') #blue
                        date_overlap_stereohi_solohi.append(start_date2)

        if 'PSP' and 'SOLO' in selected_sc and 'WISPR' and 'SOLO HI' in selected_his:
            if (start_date2 in psp_set) and (start_date2 in solo_set):
                if not overlap_wispr_solo.is_empty:
                    x, y = overlap_wispr_solo.exterior.xy
                    if overlap_fov:  
                        ax.fill(np.arctan2(y, x), np.hypot(x, y), color='y', alpha=.15, label='Overlap WISPR & SolO-HI') #green
                    date_overlap_wispr_solohi.append(start_date2)
        if 'STA' and 'SOLO' and 'PSP' in selected_sc  and 'WISPR' and 'STA HI' and 'SOLO HI' in selected_his:                
            #if (start_date2 >= min_date_psp) and (start_date2 >= min_date_solo):            
                if (start_date2 in stereo_set) and (start_date2 in solo_set) and (start_date2 in psp_set) :
                        if not overlap_wispr_stereo_solo.is_empty:
                            x, y = overlap_wispr_stereo_solo.exterior.xy  
                            if overlap_fov:          
                                ax.fill(np.arctan2(y, x), np.hypot(x, y), color='green', alpha=.2, label='Overlap WISPR, Stereo-HI & SolO-HI')#orange
                            date_overlap_all.append(start_date2)
                            
                    
    #plot_hi_geo=True




    if plot_hi_geo:   
        lamda=30
        #check for active CME indices from HIGeoCAT (with the lists produced above in this notebook)
        #check where time is identical to frame time
        #date_obs_enc17 = pd.to_datetime(date_obs_enc17, format='%Y-%m-%d %H:%M:%S')

        
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
            ax.plot(longcirc,rcirc, c='tab:orange', ls='-',alpha=0.5, lw=2.0) 
            #print("cme helcats plotted")
            #plt.figtext(0.85, 0.90,'WP3 Catalogue (HELCATS) - SSEF30', fontsize=fsize, ha='right',color='tab:orange')
    
    if plot_donki:    
        #hc_time_num1, hc_r1, hc_lat1, hc_lon1, hc_id1, a1_ell, b1_ell, c1_ell = read_donki()
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
            #ax.fill_between(longcirc1[2], rcirc1[2], rcirc1[1], color='tab:blue', alpha=.08)  #comment not to have the error
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
            #ax.fill_between(longcirc1[2], rcirc1[2], rcirc1[1], color='tab:brown', alpha=.08)
            #plt.figtext(0.02, 0.060-p*0.02,f"CME {p+1}", fontsize=fsize, ha='right',color='tab:brown')

    
    ax.set_title(start_date2, va='bottom', y=1.05)

    create_custom_legend(ax)
    plt.tight_layout()
    logo = mpimg.imread(path_to_logo+"logo_corhix_white_border.png")
    #imagebox = OffsetImage(logo, zoom=0.07)  # Make it smaller
    #ab = AnnotationBbox(imagebox, xy=(0.2, -0.1), xycoords='axes fraction', frameon=False)  # Move slightly up
    imagebox = OffsetImage(logo, zoom=0.025)
    ab = AnnotationBbox(imagebox, xy=(0.90, 0.03), xycoords='axes fraction', frameon=False)  # Move slightly up
    ax.add_artist(ab)
    file_path = os.path.join(temp_dir_path, f"{title}_sc_constellation.png")      
    #paths_to_fig.append(file_path)       
    # Save the figure to the file and add to plot_files               
    fig.savefig(file_path)
    plt.close(fig)
    st.session_state.paths_to_fig.append(file_path)
    
    

    return fig



with col2:

    
    st.markdown(
            """
            <h1 style='font-size: 40px;'>Welcome to CORHI-X!</h1>
            <l3 style='font-size: 20px;'>
            This app allows you to explore spacecraft constellations, visualize instrument fields of view, check data availability for coronagraphs and heliospheric imagers (displaying the FoV only when data is available in the instrument’s online archives), and propagate either a CME catalog or your own CME data. We hope this tool makes it easier for you to study heliospheric events!

            """, 
            unsafe_allow_html=True
        )
    
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = st.session_state.temp_dir.name
    
    if "paths_to_fig" not in st.session_state:
        st.session_state.paths_to_fig = []

    

    plot_files = []

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
    if 'n_intervals' not in st.session_state:
        st.session_state['n_intervals'] = int(delta.total_seconds() / (30 * 60))
    #print(intervals_30_min) 
    # 

    if st.button('Generate the plots'):
            st.session_state.paths_to_fig = []  # Clear previous plots
            st.session_state.temp_dir = tempfile.TemporaryDirectory()
            start_time_make_frame = time.time() 
            print("before:")
            print(st.session_state.temp_dir.name)
            print(st.session_state.paths_to_fig)
            loading_message = st.empty()
            # Display a single statement (this will remain constant throughout the process)
            figures = []
            paths_to_fig = []
            progress_bar = st.progress(0)
            m=0
            for interval in range(int(st.session_state["intervals_lenght"])+1):
                title = datetime.strptime(st.session_state["t_start2"], "%Y-%m-%d %H:%M:%S")  + timedelta(hours= cad * interval)     
                try:
                    # Create the plot
                    fig = make_frame(interval)  # Replace with your actual plotting function     

                    progress_bar.progress((interval + 1) / (int(st.session_state["intervals_lenght"]) + 1))
                except Exception as e:
                            st.error(f"Error generating plot for {title}: {e}")
            total_time = np.round((time.time() - start_time_make_frame), 2)
            print("after:")
            #print(st.session_state.temp_dir.name)
            print(st.session_state.paths_to_fig)

            loading_message.markdown(f"<p style='color: green; font-size: 14px;'>Plot generation completed in {total_time} seconds</p>", unsafe_allow_html=True)
            #print('time make frame in minutes: ',np.round((time.time()-start_time_make_frame)/60))
            #print(paths_to_fig)
            if "gif_buffer" not in st.session_state:
                st.session_state.gif_buffer = None

            print("Making animation...")

            #  Display the GIF using fragment: this helps to show the gif also when it get downloaded
            @st.fragment
            def gif_display():
                st.image(st.session_state.gif_buffer)

                #  Persistent download button (does NOT cause re-run issues)
                st.download_button(
                    label="Download GIF",
                    data=st.session_state.gif_buffer,
                    file_name="animation.gif",
                    mime="image/gif",
                    key="download_gif"
                )

            st.session_state.gif_buffer  = create_gif_animation(st.session_state.paths_to_fig, duration=200)  # Adjust duration for speed
            gif_display()
            st.warning("Note: Observation dates for each instrument are updated monthly (Last update: " + str(times_cor1_obs[-1])+ "). If an instrument's data is not yet available in its archive, its FoV may not be visible. Check below for the latest published data, at the date of the montly update, for each instrument. ")
            st.success('''
                       📄 **Citation:** Please cite the following paper if you use CORHI-X in your publication.
                       Cappello, G.M., Temmer, M., Weiler, E., Liberatore, A., Moestl, C., Amerstorfer, T. (2025).
                       CORHI-X: a Python tool to investigate heliospheric events through multiple observation angles and heliocentric distances. *Frontiers in Astronomy and Space Sciences* 12. [doi:10.3389/fspas.2025.1571024]{https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2025.1571024/full}
            ''')
            doi = "10.5281/zenodo.14800582."  # Replace with your actual DOI
            zenodo_url = f"https://zenodo.org/records/14800583"
            st.markdown(f"📄 **Zenodo repository:** [DOI: {doi}]({zenodo_url})")
            st.markdown(f"💻 **GitHub repository:** [gretacappello/CORHI_X](https://github.com/gretacappello/CORHI_X)")
            st.markdown(f"🌐 **DONKI Catalog :** [Link to DONKI](https://kauai.ccmc.gsfc.nasa.gov/DONKI/)")
            st.markdown(f"🌐 **HI-Geo Catalog:** [Link to HI-Geo](https://www.helcats-fp7.eu/catalogues/wp3_cat.html)")
            st.markdown(f"🚀 **PSP/WISPR Data (data released till: " + str(times_wispr_obs[-1]) + ":** [Download](https://wispr.nrl.navy.mil/wisprdata/)")
            st.markdown(f"🚀 **Solo/Solo-HI Data (data released till: " + str(times_solohi_obs[-1]) + ":** [Download](https://solohi.nrl.navy.mil/so_data/)")
            st.markdown(f"🚀 **STA/HI-A (data released till: " + str(times_hi1A_obs[-1]) + ":** [Download](https://secchi.nrl.navy.mil/get_data)")
            st.markdown(f"🚀 **SOHO/C2/C3 Data (data released till: " + str(times_c2_obs[-1]) + ":** [Download](https://lasco-www.nrl.navy.mil/index.php?p=get_data)")     
            st.markdown(f"🚀 **Solo/Metis Data (data released till: " + str(times_metis_obs[-1]) + ":** [Download](https://soar.esac.esa.int/soar/#search)"+ " or if you cannot find the available data in SOAR, please contact the team at metis@inaf.it to request them.")       
            st.markdown(f"🚀 **STA/COR1-COR2 Data (data released till: " + str(times_cor1_obs[-1]) + ":** [Download](https://secchi.nrl.navy.mil/get_data)")
            
     
