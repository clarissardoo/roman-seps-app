from flask import Flask,render_template_string,request
import numpy as np
import pandas as pd
from astropy.time import Time
from radvel.basis import Basis
from radvel.utils import Msini
from orbitize.basis import tp_to_tau
from orbitize.kepler import calc_orbit
from astropy import units as u
import matplotlib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import io
import base64
from pathlib import Path
import os


def weighted_percentile(data,weights,percentile):
    """Compute weighted percentile given posteriors and ln-like weights from posteriors sampled"""
    result=np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        sorted_indices=np.argsort(data[i,:])
        sorted_data=data[i,sorted_indices]
        sorted_weights=weights[sorted_indices]
        cumsum=np.cumsum(sorted_weights)
        cutoff=percentile/100.0
        idx=np.searchsorted(cumsum,cutoff)
        if idx>=len(sorted_data):
            idx=len(sorted_data)-1
        result[i]=sorted_data[idx]
    return result


def compute_sep(
        df,epochs,basis,m0,m0_err,plx,plx_err,n_planets=1,pl_num=1,override_inc=None,override_lan=None
):
    """
    Computes a sky-projected angular separation posterior given a
    RadVel-computed DataFrame
    Adapted from Sarah Blunt's compute_sep function to include override inclnations.

        Args:
        df (pd.DataFrame): Radvel-computed posterior (in any orbital basis)
        epochs (np.array of astropy.time.Time): epochs at which to compute
            separations
        basis (str): basis string of input posterior (see
            radvel.basis.BASIS_NAMES` for the full list of possibilities).
        m0 (float): median of primary mass distribution (assumed Gaussian).
        m0_err (float): 1sigma error of primary mass distribution
            (assumed Gaussian).
        plx (float): median of parallax distribution (assumed Gaussian).
        plx_err: 1sigma error of parallax distribution (assumed Gaussian).
        n_planets (int): total number of planets in RadVel posterior
        pl_num (int): planet number used in RadVel fits (e.g. a RadVel label of
            'per1' implies `pl_num` == 1)
        override_inc (float, optional): override inclination (int or None)
        override_lan (str, optional): override longitude of ascending node (int or None)

    Example:

        >> df = pandas.read_csv('sample_radvel_chains.csv.bz2', index_col=0)
        >> epochs = astropy.time.Time([2022, 2024], format='decimalyear')
        >> seps, df_orb = compute_sep(
               df, epochs, 'per tc secosw sesinw k', 0.82, 0.02, 312.22, 0.47
           )

    Returns:
        tuple of:
            np.array of size (len(epochs) x len(df)): sky-projected angular
                separations [mas] at each input epoch
            pd.DataFrame: corresponding orbital posterior in orbitize basis
    """
    myBasis=Basis(basis,n_planets)
    df=myBasis.to_synth(df)
    chain_len=len(df)
    tau_ref_epoch=58849

    # convert RadVel posteriors -> orbitize posteriors
    m_st=np.random.normal(m0,m0_err,size=chain_len)
    semiamp=df['k{}'.format(pl_num)].values
    per_day=df['per{}'.format(pl_num)].values
    period_yr=per_day/365.25
    ecc=df['e{}'.format(pl_num)].values
    msini=(
            Msini(semiamp,per_day,m_st,ecc,Msini_units='Earth')*
            (u.M_earth/u.M_sun).to('')
    )

    if override_inc is not None:
        # Numerical value provided
        inc=np.full(chain_len,np.radians(override_inc))
    else:
        # None - use random cos(i) sampling
        cosi=(2.*np.random.random(size=chain_len))-1.
        inc=np.arccos(cosi)

    m_pl=msini/np.sin(inc)
    mtot=m_st+m_pl
    sma=(period_yr**2*mtot)**(1/3)
    omega_st_rad=df['w{}'.format(pl_num)].values
    omega_pl_rad=omega_st_rad+np.pi
    parallax=np.random.normal(plx,plx_err,size=chain_len)

    if override_lan is not None:
        lan=np.full(chain_len,np.radians(override_lan))
    else:
        lan=np.random.random_sample(size=chain_len)*2.*np.pi

    tp_mjd=df['tp{}'.format(pl_num)].values-2400000.5
    tau=tp_to_tau(tp_mjd,tau_ref_epoch,period_yr)

    # compute projected separation in mas
    raoff,deoff,vz=calc_orbit(
        epochs.mjd,sma,ecc,inc,
        omega_pl_rad,lan,tau,
        parallax,mtot,tau_ref_epoch=tau_ref_epoch
    )
    seps=np.sqrt(raoff**2+deoff**2)

    # Compute 3D positions for phase angle calculation
    n_epochs=len(epochs)
    x_mas=np.zeros((n_epochs,chain_len))
    y_mas=np.zeros((n_epochs,chain_len))
    z_mas=np.zeros((n_epochs,chain_len))

    # Thiele-Innes constants
    A=sma*(np.cos(omega_pl_rad)*np.cos(lan)-np.sin(omega_pl_rad)*np.sin(lan)*np.cos(inc))
    B=sma*(np.cos(omega_pl_rad)*np.sin(lan)+np.sin(omega_pl_rad)*np.cos(lan)*np.cos(inc))
    F=sma*(-np.sin(omega_pl_rad)*np.cos(lan)-np.cos(omega_pl_rad)*np.sin(lan)*np.cos(inc))
    G=sma*(-np.sin(omega_pl_rad)*np.sin(lan)+np.cos(omega_pl_rad)*np.cos(lan)*np.cos(inc))
    C=sma*np.sin(omega_pl_rad)*np.sin(inc)
    H=sma*np.cos(omega_pl_rad)*np.sin(inc)

    for i in range(n_epochs):
        # Mean anomaly
        n_motion=2*np.pi/per_day
        M=n_motion*(epochs.mjd[i]-tp_mjd)

        # Eccentric anomaly
        EA=M+ecc*np.sin(M)+ecc**2*np.sin(2*M)/2
        for _ in range(20):
            err=EA-ecc*np.sin(EA)-M
            if np.all(np.abs(err)<1e-15):
                break
            EA=EA-err/(1-ecc*np.cos(EA))

        # Position in orbital plane
        X=np.cos(EA)-ecc
        Y=np.sqrt(1-ecc**2)*np.sin(EA)

        # 3D position in AU
        x_au=(B*X+G*Y)
        y_au=(A*X+F*Y)
        z_au=(C*X+H*Y)

        # Convert to mas
        x_mas[i,:]=x_au*parallax
        y_mas[i,:]=y_au*parallax
        z_mas[i,:]=z_au*parallax

    # 3D orbital radius
    r_mas=np.sqrt(x_mas**2+y_mas**2+z_mas**2)

    return seps,raoff,deoff,z_mas,r_mas


def weighted_std(data,weights):
    """Compute weighted standard deviation along axis 1"""
    if data.ndim==1:
        mean=np.average(data,weights=weights)
        variance=np.average((data-mean)**2,weights=weights)
        return np.sqrt(variance)
    else:
        mean=np.average(data,axis=1,weights=weights)
        variance=np.average((data-mean[:,np.newaxis])**2,axis=1,weights=weights)
        return np.sqrt(variance)


def get_median_orbit_indices(raoff_2d,deoff_2d,weights):
    """
    Find the orbit sample closest to the weighted median trajectory.
    Returns the index of that sample.
    """
    # Compute weighted median RA and Dec at each epoch
    med_ra=np.zeros(raoff_2d.shape[0])
    med_dec=np.zeros(raoff_2d.shape[0])

    for i in range(raoff_2d.shape[0]):
        # Weighted median for RA
        sorted_idx_ra=np.argsort(raoff_2d[i,:])
        sorted_weights_ra=weights[sorted_idx_ra]
        cumsum_ra=np.cumsum(sorted_weights_ra)
        median_idx_ra=np.searchsorted(cumsum_ra,0.5)
        if median_idx_ra>=len(sorted_idx_ra):
            median_idx_ra=len(sorted_idx_ra)-1
        med_ra[i]=raoff_2d[i,sorted_idx_ra[median_idx_ra]]

        # Weighted median for Dec
        sorted_idx_dec=np.argsort(deoff_2d[i,:])
        sorted_weights_dec=weights[sorted_idx_dec]
        cumsum_dec=np.cumsum(sorted_weights_dec)
        median_idx_dec=np.searchsorted(cumsum_dec,0.5)
        if median_idx_dec>=len(sorted_idx_dec):
            median_idx_dec=len(sorted_idx_dec)-1
        med_dec[i]=deoff_2d[i,sorted_idx_dec[median_idx_dec]]

    # Find the orbit that has minimum distance to this median trajectory
    distances=np.zeros(raoff_2d.shape[1])
    for j in range(raoff_2d.shape[1]):
        distances[j]=np.sum(np.sqrt((raoff_2d[:,j]-med_ra)**2+(deoff_2d[:,j]-med_dec)**2))

    return np.argmin(distances)


def compute_sep_skyplane(
        df,epochs,basis,m0,m0_err,plx,plx_err,
        n_planets=1,pl_num=1,override_inc=None
):
    """
    Computes sky-plane projected separation using Thiele-Innes with Ω=0 to match orbit_getpoints
    SMA bug fixed.
    """

    myBasis=Basis(basis,n_planets)
    df=myBasis.to_synth(df)
    chain_len=len(df)

    # Stellar mass distribution
    m_st=np.random.normal(m0,m0_err,size=chain_len)

    # Orbital params
    per_day=df[f'per{pl_num}'].values
    ecc=df[f'e{pl_num}'].values
    semiamp=df[f'k{pl_num}'].values
    omega_star=df[f'w{pl_num}'].values
    omega_pl=omega_star+np.pi

    # Calculate semi-major axis assuming edge-on (i=90°) - this was the bug btw D:
    period_yr=per_day/365.25
    msini=(
            Msini(semiamp,per_day,m_st,ecc,Msini_units='Earth')
            *(u.M_earth/u.M_sun).to('')
    )
    m_pl=msini  # At i=90 sin(i)=1
    mtot=m_st+m_pl
    sma=(period_yr**2*mtot)**(1/3)

    # Sample inclination
    if override_inc is None:
        cosi=2*np.random.rand(chain_len)-1
        inc=np.arccos(cosi)
    else:
        inc=np.full(chain_len,np.radians(override_inc))

    # Parallax (mas)
    parallax=np.random.normal(plx,plx_err,size=chain_len)

    # Thiele-Innes constants with Ω=0
    omega=np.zeros(chain_len)  # Ω = 0
    w=omega_pl

    A=sma*(np.cos(w)*np.cos(omega)-np.sin(w)*np.sin(omega)*np.cos(inc))
    B=sma*(np.cos(w)*np.sin(omega)+np.sin(w)*np.cos(omega)*np.cos(inc))
    F=sma*(-np.sin(w)*np.cos(omega)-np.cos(w)*np.sin(omega)*np.cos(inc))
    G=sma*(-np.sin(w)*np.sin(omega)+np.cos(w)*np.cos(omega)*np.cos(inc))

    # Time of periastron
    tp_mjd=df[f'tp{pl_num}'].values-2400000.5

    # Initialize output arrays
    n_epochs=len(epochs)
    X_mas=np.zeros((n_epochs,chain_len))
    Y_mas=np.zeros((n_epochs,chain_len))

    # Calculate positions for each epoch
    for i in range(n_epochs):
        # Mean anomaly
        n_motion=2*np.pi/per_day
        M=n_motion*(epochs.mjd[i]-tp_mjd)

        # Eccentric anomaly
        EA=M+ecc*np.sin(M)+ecc**2*np.sin(2*M)/2
        for _ in range(20):
            err=EA-ecc*np.sin(EA)-M
            if np.all(np.abs(err)<1e-15):
                break
            EA=EA-err/(1-ecc*np.cos(EA))

        # Position in orbital plane
        X=np.cos(EA)-ecc
        Y=np.sqrt(1-ecc**2)*np.sin(EA)

        # Apply Thiele-Innes transformation
        xpos_au=B*X+G*Y  # cos(i) factor
        ypos_au=A*X+F*Y  # stays full size

        # SWAP: X_mas gets the full-size coordinate, Y_mas gets the deprojected one
        #this keeps top/bottom idea for "favorable" consistent.
        X_mas[i,:]=ypos_au*parallax  # Full size
        Y_mas[i,:]=xpos_au*parallax  # Deprojected by cos(i)

    # Calculate separations
    seps=np.sqrt(X_mas**2+Y_mas**2)

    return seps,X_mas,Y_mas


base_path=Path("orbit_fits")

# Display names for prettier UI (now includes planet letters)
display_names={
    "47_UMa_c":"47 UMa c",
    "47_UMa_b":"47 UMa b",
    "47_UMa_d":"47 UMa d",
    "55_Cnc_d":"55 Cancri d",
    "eps_Eri_b":"Eps Eri b",
    "HD_87883_b":"HD 87883 b",
    "HD_114783_c":"HD 114783 c",
    "HD_134987_c":"HD 134987 c",
    "HD_154345_b":"HD 154345 b",
    "HD_160691_c":"HD 160691 c",
    "HD_190360_b":"HD 190360 b",
    "HD_217107_c":"HD 217107 c",
    "pi_Men_b":"Pi Men b",
    "ups_And_d":"Ups And d",
    "HD_192310_c":"HD 192310 c",
}

orbit_params={
    "47_UMa_c":{
        "star":"47_UMa",'pl_letter':'c',
        "basis":"per tc secosw sesinw k",
        "m0":1.0051917028549999,"m0_err":0.0468882076437500,
        "plx":72.0070,"plx_err":0.0974,
        "n_planets":3,"pl_num":2,"g_mag":4.866588,
    },
    "47_UMa_b":{
        "star":"47_UMa",'pl_letter':'b',
        "basis":"per tc secosw sesinw k",
        "m0":1.0051917028549999,"m0_err":0.0468882076437500,
        "plx":72.0070,"plx_err":0.0974,
        "n_planets":3,"pl_num":1,"g_mag":4.866588,
    },
    "47_UMa_d":{
        "star":"47_UMa",'pl_letter':'d',
        "basis":"per tc secosw sesinw k",
        "m0":1.0051917028549999,"m0_err":0.0468882076437500,
        "plx":72.0070,"plx_err":0.0974,
        "n_planets":3,"pl_num":3,"g_mag":4.866588,
    },
    "55_Cnc_d":{
        'star':"55_Cnc",'pl_letter':'d',
        "basis":"per tc secosw sesinw k",
        "m0":0.905,"m0_err":0.015,
        "plx":79.4482,"plx_err":0.0429,
        "n_planets":5,"pl_num":3,"g_mag":5.732681,
    },
    "eps_Eri_b":{
        'star':'eps_Eri','pl_letter':'b',
        "basis":"per tc secosw sesinw k",
        "m0":0.82,"m0_err":0.02,
        "plx":310.5773,"plx_err":0.1355,
        "n_planets":1,"pl_num":1,"g_mag":3.465752,
        "inc_mean":78.810,"inc_sig":29.340,
    },
    "HD_87883_b":{
        'star':'HD_87883','pl_letter':'b',
        "basis":"per tc secosw sesinw k",
        "m0":0.810,"m0_err":0.091,
        "plx":54.6678,"plx_err":0.0295,
        "n_planets":1,"pl_num":1,"g_mag":7.286231,
        "inc_mean":25.45,"inc_sig":1.61,
    },
    "HD_114783_c":{
        'star':'HD_114783','pl_letter':'c',
        "basis":"per tc secosw sesinw k",
        "m0":0.90,"m0_err":0.04,
        "plx":47.5529,"plx_err":0.0291,
        "n_planets":2,"pl_num":2,"g_mag":7.330857,
        "inc_mean":159,"inc_sig":6,
    },
    "HD_134987_c":{
        'star':'HD_134987','pl_letter':'c',
        "basis":"per tc secosw sesinw k",
        "m0":1.0926444945650000,"m0_err":0.0474835459017250,
        "plx":38.1946,"plx_err":0.0370,
        "n_planets":2,"pl_num":2,"g_mag":6.302472,
    },
    "HD_154345_b":{
        'star':'HD_154345','pl_letter':'b',
        "basis":"per tc secosw sesinw k",
        "m0":0.88,"m0_err":0.09,
        "plx":54.7359,"plx_err":0.0176,
        "n_planets":1,"pl_num":1,"g_mag":6.583667,
        "inc_mean":69,"inc_sig":13,
        'pl_letter':'b',
    },
    "HD_160691_c":{
        'star':'HD_160691','pl_letter':'c',
        "basis":"per tc secosw sesinw k",
        "m0":1.13,"m0_err":0.02,
        "plx":64.082,"plx_err":0.120162,
        "n_planets":4,"pl_num":4,"g_mag":4.942752,
    },
    "HD_190360_b":{
        'star':'HD_190360','pl_letter':'b',
        "basis":"per tc secosw sesinw k",
        "m0":1.0,"m0_err":0.1,
        "plx":62.4865,"plx_err":0.0354,
        "n_planets":2,"pl_num":1,"g_mag":5.552787,
        "inc_mean":80.2,"inc_sig":23.2,
    },
    "HD_217107_c":{
        'star':'HD_217107','pl_letter':'c',
        "basis":"per tc secosw sesinw k",
        "m0":1.05963082882500,"m0_err":0.04470613802572,
        "plx":49.7846,"plx_err":0.0263,
        "n_planets":2,"pl_num":2,"g_mag":5.996743,
        "inc_mean":89.3,"inc_sig":9.0,
    },
    "pi_Men_b":{
        'star':'pi_Men','pl_letter':'b',
        "basis":"per tc secosw sesinw k",
        "m0":1.10,"m0_err":0.14,
        "plx":54.6825,"plx_err":0.0354,
        "n_planets":1,"pl_num":1,"g_mag":5.511580,
        "inc_mean":54.436,"inc_sig":5.945,
    },
    "ups_And_d":{
        'star':'ups_And','pl_letter':'d',
        "basis":"per tc secosw sesinw k",
        "m0":1.29419667430000,"m0_err":0.04122482369025,
        "plx":74.1940,"plx_err":0.2083,
        "n_planets":3,"pl_num":3,"g_mag":3.966133,
        "inc_mean":23.758,"inc_sig":1.316,
    },
    "HD_192310_c":{
        'star':'HD_192310','pl_letter':'c',
        "basis":"per tc secosw sesinw k",
        "m0":0.84432448757250,"m0_err":0.02820926681885,
        "plx":113.4872,"plx_err":0.0516,
        "n_planets":2,"pl_num":2,"g_mag":5.481350,
    },
}

# Load posteriors - files are organized by star name, not planet name
posterior_cache={}
for planet_key in orbit_params.keys():
    params=orbit_params[planet_key]
    star_name=params['star']

    # Look for posterior files in orbit_fits/{star_name}/ directory
    star_dir=base_path/star_name
    files=list(star_dir.glob("*.csv.bz2"))

    if files:
        if len(files)>1:
            print(f"Warning: Multiple posterior files found for {star_name}, using first one")
        posterior_cache[planet_key]=pd.read_csv(files[0])
    else:
        print(f"Warning: No posterior data found for {planet_key} in {star_dir}")
        posterior_cache[planet_key]=None

# Flask App
app=Flask(__name__)

HTML="""
<!doctype html>
<html>
<head>
<title>Planet Orbital Visualization Tool</title>
<style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    .form-group { margin-bottom: 15px; }
    label { display: inline-block; width: 250px; font-weight: bold; }
    input, select { width: 200px; padding: 5px; }
    input[type="submit"] { width: auto; padding: 10px 30px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
    input[type="submit"]:hover { background-color: #45a049; }
    img { max-width: 100%; height: auto; margin-top: 20px; }
</style>
</head>
<body>

<h2>Planet Orbital Visualization Tool</h2>
<p>Visualize orbital trajectories and visibility for RV-detected planets</p>

<form method="post">

  <div class="form-group">
    <label>Planet:</label>
    <select name="planet">
      {% for name in planets %}
        <option value="{{name}}" {% if name == selected_planet %}selected{% endif %}>{{display_names[name]}}</option>
      {% endfor %}
    </select>
  </div>

  <div class="form-group">
    <label>Start date:</label>
    <input type="text" name="start_date" value="{{start_date or '2027-01-01'}}" placeholder="2027-01-01">
  </div>

  <div class="form-group">
    <label>End date:</label>
    <input type="text" name="end_date" value="{{end_date or '2028-06-01'}}" placeholder="2028-06-01">
  </div>

  <div class="form-group">
    <label>Inclination (degrees or "unknown"):</label>
    <input type="text" name="inclination" value="{{inclination or 'unknown'}}" placeholder="unknown">
  </div>

  <div class="form-group">
    <label>Longitude of Ascending Node Ω (degrees or "random"):</label>
    <input type="text" name="lan" value="{{lan or 'random'}}" placeholder="random">
  </div>

  <div class="form-group">
    <label>Number of samples (default 10000 or "all"):</label>
    <input type="text" name="nsamp" value="{{nsamp or '10000'}}" placeholder="10000">
  </div>

  <input type="submit" value="Generate Plots">
</form>

{% if plot_img %}
    <h3>Results</h3>
    <img src="data:image/png;base64,{{ plot_img }}" alt="Orbital Plots">
{% endif %}

{% if error %}
    <p style="color: red;"><b>Error:</b> {{ error }}</p>
{% endif %}

</body>
</html>
"""


@app.route("/",methods=["GET","POST"])
def index():
    if request.method=="GET":
        return render_template_string(
            HTML,planets=orbit_params.keys(),display_names=display_names,plot_img=None,error=None,
            selected_planet=None,start_date=None,end_date=None,
            inclination=None,lan=None,nsamp=None
        )

    # Get form data
    planet=request.form["planet"]
    start_date_str=request.form["start_date"]
    end_date_str=request.form["end_date"]
    inc_str=request.form.get("inclination","unknown").strip()
    lan_str=request.form.get("lan","random").strip()
    nsamp_str=request.form.get("nsamp","10000").strip()

    # parse parameters
    try:
        if nsamp_str.lower()=="all":
            nsamp=None  # Use all samples
        else:
            nsamp=int(nsamp_str) if nsamp_str else 10000

        # Handle inclination: "unknown" or numerical value
        if inc_str.lower()=="unknown":
            override_inc="unknown"  # Store as string
        else:
            override_inc=float(inc_str)  # Store as number

        override_lan=None if lan_str.lower()=="random" else float(lan_str)
    except ValueError as e:
        return render_template_string(
            HTML,planets=orbit_params.keys(),display_names=display_names,plot_img=None,
            error=f"Invalid input: {e}",
            selected_planet=planet,start_date=start_date_str,end_date=end_date_str,
            inclination=inc_str,lan=lan_str,nsamp=nsamp_str
        )

    if posterior_cache[planet] is None:
        return render_template_string(
            HTML,planets=orbit_params.keys(),display_names=display_names,plot_img=None,
            error=f"No posterior data found for {planet}",
            selected_planet=planet,start_date=start_date_str,end_date=end_date_str,
            inclination=inc_str,lan=lan_str,nsamp=nsamp_str
        )

    df=posterior_cache[planet]
    params=orbit_params[planet]

    # Dates
    try:
        t_start=Time(start_date_str)
        t_end=Time(end_date_str)
    except:
        return render_template_string(
            HTML,planets=orbit_params.keys(),display_names=display_names,plot_img=None,
            error="Invalid date format. Try: 2026-06-01",
            selected_planet=planet,start_date=start_date_str,end_date=end_date_str,
            inclination=inc_str,lan=lan_str,nsamp=nsamp_str
        )

    if t_end<=t_start:
        return render_template_string(
            HTML,planets=orbit_params.keys(),display_names=display_names,plot_img=None,
            error="End date must be after start date.",
            selected_planet=planet,start_date=start_date_str,end_date=end_date_str,
            inclination=inc_str,lan=lan_str,nsamp=nsamp_str
        )

    #sample from posteriors
    if nsamp is None:
        df_sample=df  # Use all samples
    else:
        df_sample=df.sample(nsamp,replace=True)

    myBasis=Basis(params["basis"],params["n_planets"])

    # Now create synth from sampled data
    df_synth=myBasis.to_synth(df_sample)

    # Get lnlike for weighting the posteriors
    lnlike=df_synth["lnprobability"].values

    # Create normalized weights from log-likelihoods
    weights=np.exp(lnlike-np.max(lnlike))
    weights=weights/np.sum(weights)

    epochs_sep=Time(np.linspace(t_start.mjd,t_end.mjd,40),format="mjd")

    # Convert to datetime objects for plotting
    from datetime import datetime
    # Convert epochs to decimal year for datetime conversion
    dates_sep_decimal=epochs_sep.decimalyear

    # Convert to datetime objects for plotting
    from datetime import datetime
    dates_sep=[datetime(int(t),1,1)+(datetime(int(t)+1,1,1)-datetime(int(t),1,1))*(t-int(t))
               for t in dates_sep_decimal]
    epochs_2d=Time(np.linspace(t_start.mjd,t_end.mjd,100),format="mjd")

    # Prepare override_inc for compute_sep and sky-plane
    # For "unknown": RA/Dec uses random cos(i), sky-plane uses face-on
    if override_inc=="unknown":
        override_inc_for_compute=None  # Random cos(i) for RA/Dec and visibility
        override_inc_for_skyplane=0.01  # Near face-on for sky-plane plot only
    else:
        # Numerical value - use same for both
        override_inc_for_compute=override_inc
        override_inc_for_skyplane=override_inc

    # Compute RA/DEC seps
    seps,_,_,z_mas,r_mas=compute_sep(
        df_sample,epochs_sep,
        params["basis"],params["m0"],params["m0_err"],
        params["plx"],params["plx_err"],
        params["n_planets"],params["pl_num"],
        override_inc=override_inc_for_compute,
        override_lan=override_lan
    )

    # Compute weighted statistics
    med_sep=weighted_percentile(seps,weights,50)
    low_sep=weighted_percentile(seps,weights,16)
    high_sep=weighted_percentile(seps,weights,84)

    # Visibility fractions w/ weighted percentiles
    IWA=155
    OWA=436
    IWAw=450
    OWAw=1300

    visible_narrow=(seps>=IWA)&(seps<=OWA)
    visible_wide=(seps>=IWAw)&(seps<=OWAw)

    visible_frac_narrow=np.sum(visible_narrow*weights[None,:],axis=1)/np.sum(weights)*100
    visible_frac_wide=np.sum(visible_wide*weights[None,:],axis=1)/np.sum(weights)*100

    # Phase angle calculations
    phase_angle_rad=np.arccos(np.clip(z_mas/r_mas,-1,1))  # Clip to handle numerical errors
    phase_angle_deg=np.degrees(phase_angle_rad)

    # Lambert phase function
    lambert_phase=(np.sin(phase_angle_rad)+(np.pi-phase_angle_rad)*np.cos(phase_angle_rad))/np.pi

    # Compute weighted statistics for phase angles
    med_phase=weighted_percentile(phase_angle_deg,weights,50)
    low_phase=weighted_percentile(phase_angle_deg,weights,16)
    high_phase=weighted_percentile(phase_angle_deg,weights,84)

    med_lambert=weighted_percentile(lambert_phase,weights,50)
    low_lambert=weighted_percentile(lambert_phase,weights,16)
    high_lambert=weighted_percentile(lambert_phase,weights,84)

    # compute 2d ra/dec plot
    seps_2d,raoff_2d,deoff_2d,z_mas_2d,r_mas_2d=compute_sep(
        df_synth,epochs_2d,
        params["basis"],params["m0"],params["m0_err"],
        params["plx"],params["plx_err"],
        params["n_planets"],params["pl_num"],
        override_inc=override_inc_for_compute,
        override_lan=override_lan
    )

    # plane of sky
    seps_2d_sky,raoff_2d_sky,deoff_2d_sky=compute_sep_skyplane(
        df_synth,epochs_2d,
        params["basis"],params["m0"],params["m0_err"],
        params["plx"],params["plx_err"],
        params["n_planets"],params["pl_num"],
        override_inc=override_inc_for_skyplane  # Use skyplane-specific inclination
    )

    # =========================================
    # CREATE PLOT WITH PLASMA COLORS (5 subplots)
    # UPDATED: Smaller right plots, no text overlays on middle plot
    # =========================================
    fig=plt.figure(figsize=(32,14))
    gs=fig.add_gridspec(3,5,height_ratios=[1.5,0.5,0.5],width_ratios=[1.5,1.5,0.8,0.8,0.1],hspace=0.35,wspace=0.3)

    # Plasma colormap colors
    cm=plt.cm.plasma
    c_iwa_narrow=cm(0.85)  # bright yellow-orange
    c_iwa_wide=cm(0.5)  # magenta
    c_orbit_light=cm(0.2)  # dark purple
    c_star=cm(0.0)  # dark purple/blue
    c_median=cm(0.6)  # orange
    c_fill=cm(0.2)  # dark purple
    c_samples=cm(0.15)  # very dark purple

    # PLOT 1: 2D ORBIT (ra/dec)
    ax1=fig.add_subplot(gs[:,0])
    inc_display='unknown' if override_inc=='unknown' else f'{override_inc}°'
    title_2d=f"{display_names[planet]}: Orbital Trajectory \n (i={inc_display}"
    if override_lan is not None:
        title_2d+=f", Ω={override_lan}°)"
    else:
        title_2d+=", Ω=random)"
    ax1.set_title(title_2d,fontsize=18)
    ax1.set_xlabel("RA Offset [mas]",fontsize=18)
    ax1.set_ylabel("Dec Offset [mas]",fontsize=18)

    theta=np.linspace(0,2*np.pi,100)

    if planet in ["eps_Eri_b","47_UMa_d"]:
        # For large-separation planets, show wide FOV rings
        ax1.plot(IWAw*np.cos(theta),IWAw*np.sin(theta),color=c_iwa_wide,lw=4,linestyle='--',label='IWA/OWA (Wide)')
        ax1.plot(OWAw*np.cos(theta),OWAw*np.sin(theta),color=c_iwa_wide,lw=4,linestyle='--')
        ax1.plot(IWA*np.cos(theta),IWA*np.sin(theta),color=c_iwa_narrow,lw=4,linestyle='--',label='IWA/OWA (Narrow)')
        ax1.plot(OWA*np.cos(theta),OWA*np.sin(theta),color=c_iwa_narrow,lw=4,linestyle='--')
    else:
        # For other planets, show narrow FOV rings
        ax1.plot(IWA*np.cos(theta),IWA*np.sin(theta),color=c_iwa_narrow,lw=4,linestyle='--',label='IWA/OWA (Narrow)')
        ax1.plot(OWA*np.cos(theta),OWA*np.sin(theta),color=c_iwa_narrow,lw=4,linestyle='--')

    n_samples=min(100,raoff_2d.shape[1])
    sample_indices=np.random.choice(raoff_2d.shape[1],n_samples,replace=False)
    for i in sample_indices:
        ax1.plot(raoff_2d[:,i],deoff_2d[:,i],'-',color=c_orbit_light,alpha=0.3,linewidth=0.5)

    # Add highlighted median orbit with date markers
    median_idx_radec=get_median_orbit_indices(raoff_2d,deoff_2d,weights)

    # Plot the median orbit in bright color
    ax1.plot(raoff_2d[:,median_idx_radec],deoff_2d[:,median_idx_radec],'-',
             color=c_median,linewidth=3,alpha=0.9,zorder=10,label='Median orbit')

    # Add date markers at specific dates
    marker_dates=['2027-01-01','2027-06-01', '2028-01-01', '2028-06-01']
    marker_times=Time(marker_dates)

    # Find indices in epochs_2d closest to marker dates
    for marker_time,marker_date_str in zip(marker_times,marker_dates):
        # Find closest epoch
        time_diffs=np.abs(epochs_2d.mjd-marker_time.mjd)
        closest_idx=np.argmin(time_diffs)

        # Get position
        ra_pos=raoff_2d[closest_idx,median_idx_radec]
        dec_pos=deoff_2d[closest_idx,median_idx_radec]

        # Plot marker
        ax1.plot(ra_pos,dec_pos,'o',color='cyan',markersize=10,
                 markeredgecolor='white',markeredgewidth=2,zorder=12)

        # Add text label with date
        ax1.annotate(marker_date_str,
                     xy=(ra_pos,dec_pos),
                     xytext=(10,10),textcoords='offset points',
                     fontsize=10,fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5',facecolor='white',
                               edgecolor='cyan',alpha=0.9),
                     arrowprops=dict(arrowstyle='->',color='cyan',lw=2),
                     zorder=13)

    ax1.plot(0,0,'*',color=c_star,markersize=20,label='Star',zorder=15)

    # Set axis limits based on FOV ranges (stable, not data-dependent)
    padding=0.05

    if planet in ["eps_Eri_b","47_UMa_d"]:
        # For large-separation planets, set limits based on wide FOV
        limit=OWAw*(1+padding)
    else:
        # For other planets, set limits based on narrow FOV
        limit=OWA*(1+padding)

    ax1.set_xlim(-limit,limit)
    ax1.set_ylim(-limit,limit)
    ax1.invert_xaxis()
    ax1.set_aspect('equal')
    ax1.tick_params(axis='both',which='major',labelsize=14)
    ax1.legend(loc='best',fontsize=12)

    # PLOT 4: sky plane orbit
    ax4=fig.add_subplot(gs[:,1])
    title_sky=f"{display_names[planet]}: Orbital Plane View (i={inc_display})"
    if override_inc=="unknown":
        title_sky+="\nNote: Near face-on view (i≈0.01°)\nlarger than projected separation above"
    ax4.set_title(title_sky,fontsize=18)

    # Determine plot style based on inclination type
    if override_inc=="unknown":
        # Unknown inclination: use vertical lines (face-on view for orbit, i ≈ 1.5°)
        ax4.set_xlabel("Sky-plane offset along line of nodes [mas]",fontsize=18)
        ax4.set_ylabel("Offset perpendicular to line of nodes (in orbit plane) [mas]",fontsize=18)

        # Background shading (NO TEXT) - will be set after we determine limits
        # IWA/OWA vertical lines
        ax4.axvline(IWA,color=c_iwa_narrow,linestyle='--',linewidth=4,label='IWA/OWA (Narrow)')
        ax4.axvline(-IWA,color=c_iwa_narrow,linestyle='--',linewidth=4)
        ax4.axvline(OWA,color=c_iwa_narrow,linestyle='--',linewidth=4)
        ax4.axvline(-OWA,color=c_iwa_narrow,linestyle='--',linewidth=4)
        ax4.axvline(IWAw,color=c_iwa_wide,linestyle='--',linewidth=4,label='IWA/OWA (Wide)')
        ax4.axvline(-IWAw,color=c_iwa_wide,linestyle='--',linewidth=4)
        ax4.axvline(OWAw,color=c_iwa_wide,linestyle='--',linewidth=4)
        ax4.axvline(-OWAw,color=c_iwa_wide,linestyle='--',linewidth=4)

    else:
        # Specific inclination: use circles
        ax4.set_xlabel("Projected separation along line of nodes [mas]",fontsize=18)
        ax4.set_ylabel("Projected separation perpendicular to line of nodes [mas]",fontsize=18)

        # Background shading (NO TEXT) - will be set after we determine limits
        # IWA/OWA circles
        theta=np.linspace(0,2*np.pi,100)
        ax4.plot(IWA*np.cos(theta),IWA*np.sin(theta),color=c_iwa_narrow,lw=4,linestyle='--',label='IWA/OWA (Narrow)')
        ax4.plot(OWA*np.cos(theta),OWA*np.sin(theta),color=c_iwa_narrow,lw=4,linestyle='--')
        ax4.plot(IWAw*np.cos(theta),IWAw*np.sin(theta),color=c_iwa_wide,lw=4,linestyle='--',label='IWA/OWA (Wide)')
        ax4.plot(OWAw*np.cos(theta),OWAw*np.sin(theta),color=c_iwa_wide,lw=4,linestyle='--')

    # Plot sample orbits
    n_samples_sky=min(100,raoff_2d_sky.shape[1])
    indices_sky=np.random.choice(raoff_2d_sky.shape[1],n_samples_sky,replace=False)
    for i in indices_sky:
        ax4.plot(raoff_2d_sky[:,i],deoff_2d_sky[:,i],'-',color=c_orbit_light,
                 alpha=0.3,linewidth=0.5)

    # Add highlighted median orbit with date markers
    median_idx=get_median_orbit_indices(raoff_2d_sky,deoff_2d_sky,weights)

    # Plot the median orbit in bright color
    ax4.plot(raoff_2d_sky[:,median_idx],deoff_2d_sky[:,median_idx],'-',
             color=c_median,linewidth=3,alpha=0.9,zorder=10,label='Median orbit')

    # Add date markers at specific dates
    for marker_time,marker_date_str in zip(marker_times,marker_dates):
        # Find closest epoch
        time_diffs=np.abs(epochs_2d.mjd-marker_time.mjd)
        closest_idx=np.argmin(time_diffs)

        # Get position
        ra_pos=raoff_2d_sky[closest_idx,median_idx]
        dec_pos=deoff_2d_sky[closest_idx,median_idx]

        # Plot marker
        ax4.plot(ra_pos,dec_pos,'o',color='cyan',markersize=10,
                 markeredgecolor='white',markeredgewidth=2,zorder=12)

        # Add text label with date
        ax4.annotate(marker_date_str,
                     xy=(ra_pos,dec_pos),
                     xytext=(10,10),textcoords='offset points',
                     fontsize=10,fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5',facecolor='white',
                               edgecolor='cyan',alpha=0.9),
                     arrowprops=dict(arrowstyle='->',color='cyan',lw=2),
                     zorder=13)

    ax4.plot(0,0,'*',color=c_star,markersize=20,label='Star',zorder=15)

    # Set axis limits based on actual data with some padding
    ra_sky_min,ra_sky_max=np.min(raoff_2d_sky),np.max(raoff_2d_sky)
    dec_sky_min,dec_sky_max=np.min(deoff_2d_sky),np.max(deoff_2d_sky)
    padding=0.2

    max_extent=max(abs(ra_sky_min),abs(ra_sky_max),abs(dec_sky_min),abs(dec_sky_max))
    max_extent=max(max_extent,OWA*1.1)  # Ensure OWA is visible

    ax4.set_xlim(-max_extent*(1+padding),max_extent*(1+padding))
    ax4.set_ylim(-max_extent*(1+padding),max_extent*(1+padding))

    # Add background shading now that limits are set
    ylims=ax4.get_ylim()
    ax4.axhspan(0,ylims[1],alpha=0.05,color='green',zorder=0)
    ax4.axhspan(ylims[0],0,alpha=0.05,color='red',zorder=0)

    # Add text labels for favorable/unfavorable regions
    ax4.text(0,ylims[1]*0.45,'Favorable - high illumination phase',
             ha='center',va='center',fontsize=12,color='darkgreen',
             alpha=0.7,fontweight='bold',style='italic')
    ax4.text(0,ylims[0]*0.45,'Unfavorable - low illumination phase',
             ha='center',va='center',fontsize=12,color='darkred',
             alpha=0.7,fontweight='bold',style='italic')

    ax4.set_aspect('equal')
    ax4.tick_params(axis='both',which='major',labelsize=14)
    ax4.legend(loc='best',fontsize=12)

    # Plot 2 - sep vs time
    ax2=fig.add_subplot(gs[0,2:4])
    min_sep_1sigma=np.min(low_sep)
    max_sep_1sigma=np.max(high_sep)

    sep_title=f"Separation vs Time (1σ Range: {min_sep_1sigma:.0f}-{max_sep_1sigma:.0f} mas)"
    if override_inc=="unknown":
        sep_title+="\nNote: Uses random cos(i) sampling for projected separation"
    ax2.set_title(sep_title,fontsize=14)
    ax2.set_ylabel("Separation [mas]",fontsize=14)
    ax2.tick_params(axis='both',which='major',labelsize=12)

    ax2.plot(dates_sep,med_sep,'-',color=c_median,linewidth=2,label='Median separation',marker='o',markersize=3)
    ax2.fill_between(dates_sep,low_sep,high_sep,color=c_fill,alpha=0.5,label='1σ interval')

    n_plot=min(20,seps.shape[1])
    idxs=np.random.choice(seps.shape[1],n_plot,replace=False)
    for i in idxs:
        ax2.plot(dates_sep,seps[:,i],color=c_samples,linewidth=1,alpha=0.15)

    # Always plot IWA/OWA narrow
    ax2.axhline(y=IWA,color=c_iwa_narrow,linestyle='--',linewidth=3,label='IWA/OWA (Narrow)')
    ax2.axhline(y=OWA,color=c_iwa_narrow,linestyle='--',linewidth=3)

    # Plot wide IWA/OWA for planets with large separations (Eps Eri b, 47 UMa d)
    if planet in ["eps_Eri_b","47_UMa_d"]:
        ax2.axhline(y=IWAw,color=c_iwa_wide,linestyle='--',linewidth=3,label='IWA/OWA (Wide)')
        ax2.axhline(y=OWAw,color=c_iwa_wide,linestyle='--',linewidth=3)

    # Set y-axis limits based on data with some padding
    y_min=max(0,np.min(low_sep)*0.8)  # 20% below min, but not below 0

    # Determine appropriate y_max based on which FOV rings are relevant
    if planet in ["eps_Eri_b","47_UMa_d"]:
        # For planets with large separations, always show the wide FOV range
        y_max=OWAw*1.15  # 15% padding above OWA wide
    else:
        # For others, show narrow FOV range
        y_max=OWA*1.15  # 15% padding above OWA narrow

    ax2.set_ylim(y_min,y_max)

    ax2.legend(loc='best',fontsize=11)

    # Plot 3 - vis fraction
    ax3=fig.add_subplot(gs[1,2:4],sharex=ax2)

    # Calculate min/max visibility for the window
    min_vis_narrow=np.min(visible_frac_narrow)
    max_vis_narrow=np.max(visible_frac_narrow)
    min_vis_wide=np.min(visible_frac_wide)
    max_vis_wide=np.max(visible_frac_wide)

    vis_title=f"Visibility Fraction\nNarrow: {min_vis_narrow:.1f}%-{max_vis_narrow:.1f}% | Wide: {min_vis_wide:.1f}%-{max_vis_wide:.1f}%"
    ax3.set_title(vis_title,fontsize=14)
    ax3.set_xlabel("Date (MM/YY)",fontsize=14)
    ax3.set_ylabel("Visible [%]",fontsize=14)
    ax3.tick_params(axis='both',which='major',labelsize=12)

    ax3.plot(dates_sep,visible_frac_narrow,color=c_iwa_narrow,linewidth=2,marker='o',markersize=3,
             label='Narrow (155-436 mas)')
    ax3.fill_between(dates_sep,0,visible_frac_narrow,color=c_iwa_narrow,alpha=0.2)
    ax3.plot(dates_sep,visible_frac_wide,color=c_iwa_wide,linewidth=2,marker='s',markersize=3,
             label='Wide (450-1300 mas)')
    ax3.fill_between(dates_sep,0,visible_frac_wide,color=c_iwa_wide,alpha=0.2)

    ax3.set_ylim([0,100])
    ax3.legend(loc='best',fontsize=11)

    # Plot 5 - phase angle
    ax5=fig.add_subplot(gs[2,2:4],sharex=ax2)
    ax5.set_title("Phase Angle & Lambert Phase",fontsize=14)
    ax5.set_xlabel("Date (MM/YY)",fontsize=14)
    ax5.set_ylabel("Phase Angle [°]",fontsize=14,color=c_median)
    ax5.tick_params(axis='both',which='major',labelsize=12)
    ax5.tick_params(axis='y',labelcolor=c_median)

    # Phase angle on left y-axis
    ax5.plot(dates_sep,med_phase,color=c_median,linewidth=2,marker='o',markersize=3,
             label='Phase Angle (median)')
    ax5.fill_between(dates_sep,low_phase,high_phase,color=c_median,alpha=0.2)

    # Lambert phase on right y-axis
    ax5_right=ax5.twinx()
    ax5_right.set_ylabel("Lambert Phase Function",fontsize=14,color=c_iwa_wide)
    ax5_right.tick_params(axis='y',labelcolor=c_iwa_wide,labelsize=12)
    ax5_right.plot(dates_sep,med_lambert,color=c_iwa_wide,linewidth=2,marker='s',markersize=3,
                   label='Lambert Phase (median)')
    ax5_right.fill_between(dates_sep,low_lambert,high_lambert,color=c_iwa_wide,alpha=0.2)
    ax5_right.set_ylim([0,1])

    # Combine legends
    lines1,labels1=ax5.get_legend_handles_labels()
    lines2,labels2=ax5_right.get_legend_handles_labels()
    ax5.legend(lines1+lines2,labels1+labels2,loc='best',fontsize=11)

    plt.suptitle(f"{display_names[planet]}: {start_date_str} → {end_date_str}",fontsize=18,y=0.98)
    plt.tight_layout()

    # Convert to base64
    buf=io.BytesIO()
    plt.savefig(buf,format='png',dpi=150,bbox_inches='tight')
    buf.seek(0)
    plot_b64=base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)

    return render_template_string(
        HTML,planets=orbit_params.keys(),display_names=display_names,plot_img=plot_b64,error=None,
        selected_planet=planet,start_date=start_date_str,end_date=end_date_str,
        inclination=inc_str,lan=lan_str,nsamp=nsamp_str
    )


if __name__=="__main__":
    app.run(host=os.getenv('HOST','0.0.0.0'),port=int(os.getenv('PORT',8080)),debug=False)