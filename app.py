from flask import Flask, render_template_string, request
import numpy as np
import pandas as pd
from astropy.time import Time
from radvel.basis import Basis
from radvel.utils import Msini
from orbitize.basis import tp_to_tau
from orbitize.kepler import calc_orbit
from astropy import units as u
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os


def compute_sep(
    df, epochs, basis, m0, m0_err, plx, plx_err, n_planets=1, pl_num=1
):
    """
    Computes a sky-projected angular separation posterior given a
    RadVel-computed DataFrame.

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

    myBasis = Basis(basis, n_planets)
    df = myBasis.to_synth(df)
    chain_len = len(df)
    tau_ref_epoch = 58849

    # convert RadVel posteriors -> orbitize posteriors
    m_st = np.random.normal(m0, m0_err, size = chain_len)
    semiamp = df['k{}'.format(pl_num)].values
    per_day = df['per{}'.format(pl_num)].values
    period_yr = per_day / 365.25
    ecc = df['e{}'.format(pl_num)].values
    msini = (
        Msini(semiamp, per_day, m_st, ecc, Msini_units='Earth') *
        (u.M_earth / u.M_sun).to('')
    )
    cosi = (2. * np.random.random(size = chain_len)) - 1.
    inc = np.arccos(cosi)
    m_pl = msini / np.sin(inc)
    mtot = m_st + m_pl
    sma = (period_yr**2 * mtot)**(1/3)
    omega_st_rad = df['w{}'.format(pl_num)].values
    omega_pl_rad = omega_st_rad + np.pi
    parallax = np.random.normal(plx, plx_err, size = chain_len)
    lan = np.random.random_sample(size = chain_len) * 2. * np.pi
    tp_mjd = df['tp{}'.format(pl_num)].values - 2400000.5
    tau = tp_to_tau(tp_mjd, tau_ref_epoch, period_yr)

    # compute projected separation in mas
    raoff, deoff, _ = calc_orbit(
        epochs.mjd, sma, ecc, inc,
        omega_pl_rad, lan, tau,
        parallax, mtot, tau_ref_epoch=tau_ref_epoch
    )
    seps = np.sqrt(raoff**2 + deoff**2)

    df_orb = pd.DataFrame(
        np.transpose([sma, ecc, inc, omega_pl_rad, lan, tau, parallax, m_st, m_pl]),
        columns=[
            'sma', 'ecc', 'inc_rad', 'omega_pl_rad', 'lan_rad', 'tau_58849',
            'plx', 'm_st', 'mp'
        ]
    )

    return seps

base_path = Path("orbit_fits")
orbit_params = {
    "47_UMa": {
        "basis": "per tc secosw sesinw k",
        "m0": 1.0051917028549999, "m0_err": 0.0468882076437500,
        "plx": 72.452800, "plx_err": 0.150701,
        "n_planets": 3, "pl_num": 2, "g_mag": 4.866588,
    },
    "55_Cnc": {
        "basis": "per tc secosw sesinw k",
        "m0": 0.905, "m0_err": 0.015,
        "plx": 79.4274000, "plx_err": 0.0776646,
        "n_planets": 5, "pl_num": 3, "g_mag": 5.732681,
    },
    "eps_Eri": {
        "basis": "per tc secosw sesinw k",
        "m0": 0.82, "m0_err": 0.02,
        "plx": 312.219000, "plx_err": 0.467348,
        "n_planets": 1, "pl_num": 1, "g_mag": 3.465752,
    },
    "HD_87883": {
        "basis": "per tc secosw sesinw k",
        "m0": 0.810, "m0_err": 0.091,
        "plx": 54.6421000, "plx_err": 0.0369056,
        "n_planets": 1, "pl_num": 1, "g_mag": 7.286231,
    },
    "HD_114783": {
        "basis": "per tc secosw sesinw k",
        "m0": 0.90, "m0_err": 0.04,
        "plx": 47.4482000, "plx_err": 0.0637202,
        "n_planets": 2, "pl_num": 2, "g_mag": 7.330857,
    },
    "HD_134987": {
        "basis": "per tc secosw sesinw k",
        "m0": 1.0926444945650000, "m0_err": 0.0474835459017250,
        "plx": 38.1678000, "plx_err": 0.0746519,
        "n_planets": 2, "pl_num": 2, "g_mag": 6.302472,
    },
    "HD_154345": {
        "basis": "per tc secosw sesinw k",
        "m0": 0.88, "m0_err": 0.09,
        "plx": 54.6636000, "plx_err": 0.0212277,
        "n_planets": 1, "pl_num": 1, "g_mag": 6.583667,
    },
    "HD_160691": {
        "basis": "per tc secosw sesinw k",
        "m0": 1.13, "m0_err": 0.02,
        "plx": 15.5981, "plx_err": 0.120162,
        "n_planets": 4, "pl_num": 4, "g_mag": 4.942752,
    },
    "HD_190360": {
        "basis": "per tc secosw sesinw k",
        "m0": 1.0, "m0_err": 0.1,
        "plx": 62.4443000, "plx_err": 0.0616881,
        "n_planets": 2, "pl_num": 1, "g_mag": 5.552787
    },
    "HD_217107": {
        "basis": "per tc secosw sesinw k",
        "m0": 1.05963082882500, "m0_err": 0.04470613802572,
        "plx": 49.8170000, "plx_err": 0.0573616,
        "n_planets": 2, "pl_num": 2, "g_mag": 5.996743
    },
    "pi_Men": {
        "basis": "per tc secosw sesinw k",
        "m0": 1.10, "m0_err": 0.14,
        "plx": 54.705200, "plx_err": 0.067131,
        "n_planets": 1, "pl_num": 1, "g_mag": 5.511580
    },
    "ups_And": {
        "basis": "per tc secosw sesinw k",
        "m0": 1.29419667430000, "m0_err": 0.04122482369025,
        "plx": 74.571100, "plx_err": 0.349118,
        "n_planets": 3, "pl_num": 3, "g_mag": 3.966133
    },
    "HD_192310": {
        "basis": "per tc secosw sesinw k",
        "m0": 0.84432448757250, "m0_err": 0.02820926681885,
        "plx": 113.648000, "plx_err": 0.118606,
        "n_planets": 2, "pl_num": 2, "g_mag": 5.481350
    },
}

posterior_cache = {}
for name in orbit_params.keys():
    files = list((base_path / name).glob("*.csv.bz2"))
    if files:
        posterior_cache[name] = pd.read_csv(files[0])
    else:
        posterior_cache[name] = None



# Flask App

app = Flask(__name__)

HTML = """
<!doctype html>
<title>Planet Separation Tool</title>

<h2>Planet Separation & Visibility Calculator</h2>

<form method="post">

  <label><b>Planet:</b></label>
  <select name="planet">
    {% for name in planets %}
      <option value="{{name}}">{{name}}</option>
    {% endfor %}
  </select>
  <br><br>

  <label><b>Start date:</b></label>
  <input type="text" name="start_date" placeholder="2030-01-01">
  <br><br>

  <label><b>End date:</b></label>
  <input type="text" name="end_date" placeholder="2035-01-01">
  <br><br>

  <label><b>Number of samples (default 2000):</b></label>
  <input type="text" name="nsamp" placeholder="2000">
  <br><br>

  <input type="submit" value="Plot">
</form>

{% if plot %}
    <h3>Results</h3>
    {{ plot | safe }}
{% endif %}
"""


# MAIN ROUTE

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(
            HTML, planets=orbit_params.keys(), plot=None
        )

    planet = request.form["planet"]
    start_date_str = request.form["start_date"]
    end_date_str   = request.form["end_date"]
    nsamp_str      = request.form["nsamp"]

    nsamp = int(nsamp_str.strip()) if nsamp_str.strip() else 200

    df = posterior_cache[planet].sample(nsamp, replace=True)
    params = orbit_params[planet]

    #  Dates
    try:
        t_start = Time(start_date_str)
        t_end   = Time(end_date_str)
    except:
        return "Invalid date format. Try: 2030-01-01"

    if t_end <= t_start:
        return "End date must be after start date."

    # Build epoch sampling
    epochs = Time(
        np.linspace(t_start.mjd, t_end.mjd, 40),
        format="mjd"
    )
    times = epochs.decimalyear

    # ------------------------
    # Compute separations
    # ------------------------
    seps = compute_sep(
        df, epochs,
        params["basis"], params["m0"], params["m0_err"],
        params["plx"], params["plx_err"],
        params["n_planets"], params["pl_num"]
    )

    # Stats
    med_sep = np.median(seps, axis=1)
    low_sep = np.percentile(seps, 16, axis=1)
    high_sep = np.percentile(seps, 84, axis=1)

    # Visibility fraction
    IWA = 155
    OWA = 436
    visible_frac = np.mean((seps >= IWA) & (seps <= OWA), axis=1) * 100

    # PLOTLY FIGURE

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4],
        subplot_titles=("Separation vs Time", "Visibility Fraction")
    )

    # Top panel
    fig.add_trace(
        go.Scatter(x=times, y=med_sep, mode="lines+markers",
                   name="Median separation", line=dict(color="blue")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=times, y=low_sep, line=dict(width=0), showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=times, y=high_sep,
                   fill="tonexty", line=dict(width=0),
                   fillcolor="rgba(0,0,255,0.2)", name="1σ interval"),
        row=1, col=1
    )

    # Sample posterior curves (20)
    n_plot = min(20, seps.shape[1])
    idxs = np.random.choice(seps.shape[1], n_plot, replace=False)
    for i in idxs:
        fig.add_trace(
            go.Scatter(x=times, y=seps[:, i], mode="lines",
                       line=dict(color="gray", width=1),
                       opacity=0.15, showlegend=False),
            row=1, col=1
        )

    # IWA / OWA horizontal lines
    fig.add_hline(y=IWA, line_color="pink", line_dash="dash", line_width = 5, row=1, col=1)
    fig.add_hline(y=OWA, line_color="pink", line_dash="dash",  line_width = 5, row=1, col=1)

    # Bottom Panel
    fig.add_trace(
        go.Scatter(x=times, y=visible_frac, mode="lines+markers",
                   line=dict(color="green"), name="Visible fraction"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=times, y=np.zeros_like(visible_frac),
            fill="tonexty", fillcolor="rgba(0,255,0,0.2)",
            line=dict(width=0), showlegend=False),
        row=2, col=1
    )

    # Layout
    fig.update_layout(
        height=800,
        title=f"{planet}: {start_date_str} → {end_date_str}",
        yaxis_title="Separation [mas]",
        xaxis2_title="Year",
        yaxis2_title="Visible [%]"
    )

    return render_template_string(
        HTML,
        planets=orbit_params.keys(),
        plot=fig.to_html(full_html=False),
    )

if __name__ == "__main__":
    app.run(host=os.getenv('HOST', '0.0.0.0'), port=int(os.getenv('PORT', 8080)), debug=False)
    # app.run()