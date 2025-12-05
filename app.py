from flask import Flask, render_template_string, request
import numpy as np
import pandas as pd
from astropy.time import Time
from radvel.basis import Basis
from radvel.utils import Msini
from orbitize.basis import tp_to_tau
from orbitize.kepler import calc_orbit
from astropy import units as u
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import io
import base64
from pathlib import Path
import os


def compute_sep(
        df, epochs, basis, m0, m0_err, plx, plx_err, n_planets=1, pl_num=1, override_inc=None, override_lan=None
):
    """
    Computes a sky-projected angular separation posterior given a
    RadVel-computed DataFrame.
    """
    myBasis = Basis(basis, n_planets)
    df = myBasis.to_synth(df)
    chain_len = len(df)
    tau_ref_epoch = 58849

    # convert RadVel posteriors -> orbitize posteriors
    m_st = np.random.normal(m0, m0_err, size=chain_len)
    semiamp = df['k{}'.format(pl_num)].values
    per_day = df['per{}'.format(pl_num)].values
    period_yr = per_day / 365.25
    ecc = df['e{}'.format(pl_num)].values
    msini = (
        Msini(semiamp, per_day, m_st, ecc, Msini_units='Earth') *
        (u.M_earth / u.M_sun).to('')
    )

    if override_inc is not None:
        inc = np.full(chain_len, np.radians(override_inc))
    else:
        cosi = (2. * np.random.random(size=chain_len)) - 1.
        inc = np.arccos(cosi)

    m_pl = msini / np.sin(inc)
    mtot = m_st + m_pl
    sma = (period_yr**2 * mtot)**(1/3)
    omega_st_rad = df['w{}'.format(pl_num)].values
    omega_pl_rad = omega_st_rad + np.pi
    parallax = np.random.normal(plx, plx_err, size=chain_len)

    if override_lan is not None:
        lan = np.full(chain_len, np.radians(override_lan))
    else:
        lan = np.random.random_sample(size=chain_len) * 2. * np.pi

    tp_mjd = df['tp{}'.format(pl_num)].values - 2400000.5
    tau = tp_to_tau(tp_mjd, tau_ref_epoch, period_yr)

    # compute projected separation in mas
    raoff, deoff, _ = calc_orbit(
        epochs.mjd, sma, ecc, inc,
        omega_pl_rad, lan, tau,
        parallax, mtot, tau_ref_epoch=tau_ref_epoch
    )
    seps = np.sqrt(raoff**2 + deoff**2)

    return seps, raoff, deoff


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
        "plx": 64.082, "plx_err": 0.120162,
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
        <option value="{{name}}" {% if name == selected_planet %}selected{% endif %}>{{name}}</option>
      {% endfor %}
    </select>
  </div>

  <div class="form-group">
    <label>Start date:</label>
    <input type="text" name="start_date" value="{{start_date or '2026-06-01'}}" placeholder="2026-06-01">
  </div>

  <div class="form-group">
    <label>End date:</label>
    <input type="text" name="end_date" value="{{end_date or '2031-06-01'}}" placeholder="2031-06-01">
  </div>

  <div class="form-group">
    <label>Inclination (degrees or "random"):</label>
    <input type="text" name="inclination" value="{{inclination or 'random'}}" placeholder="random">
  </div>

  <div class="form-group">
    <label>Longitude of Ascending Node Ω (degrees or "random"):</label>
    <input type="text" name="lan" value="{{lan or 'random'}}" placeholder="random">
  </div>

  <div class="form-group">
    <label>Number of samples (default 200):</label>
    <input type="text" name="nsamp" value="{{nsamp or '200'}}" placeholder="200">
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


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(
            HTML, planets=orbit_params.keys(), plot_img=None, error=None,
            selected_planet=None, start_date=None, end_date=None,
            inclination=None, lan=None, nsamp=None
        )

    # Get form data
    planet = request.form["planet"]
    start_date_str = request.form["start_date"]
    end_date_str = request.form["end_date"]
    inc_str = request.form.get("inclination", "random").strip()
    lan_str = request.form.get("lan", "random").strip()
    nsamp_str = request.form.get("nsamp", "200").strip()

    # Parse parameters
    try:
        nsamp = int(nsamp_str) if nsamp_str else 200
        override_inc = None if inc_str.lower() == "random" else float(inc_str)
        override_lan = None if lan_str.lower() == "random" else float(lan_str)
    except ValueError as e:
        return render_template_string(
            HTML, planets=orbit_params.keys(), plot_img=None,
            error=f"Invalid input: {e}",
            selected_planet=planet, start_date=start_date_str, end_date=end_date_str,
            inclination=inc_str, lan=lan_str, nsamp=nsamp_str
        )

    if posterior_cache[planet] is None:
        return render_template_string(
            HTML, planets=orbit_params.keys(), plot_img=None,
            error=f"No posterior data found for {planet}",
            selected_planet=planet, start_date=start_date_str, end_date=end_date_str,
            inclination=inc_str, lan=lan_str, nsamp=nsamp_str
        )

    df = posterior_cache[planet]
    params = orbit_params[planet]

    # Dates
    try:
        t_start = Time(start_date_str)
        t_end = Time(end_date_str)
    except:
        return render_template_string(
            HTML, planets=orbit_params.keys(), plot_img=None,
            error="Invalid date format. Try: 2026-06-01",
            selected_planet=planet, start_date=start_date_str, end_date=end_date_str,
            inclination=inc_str, lan=lan_str, nsamp=nsamp_str
        )

    if t_end <= t_start:
        return render_template_string(
            HTML, planets=orbit_params.keys(), plot_img=None,
            error="End date must be after start date.",
            selected_planet=planet, start_date=start_date_str, end_date=end_date_str,
            inclination=inc_str, lan=lan_str, nsamp=nsamp_str
        )

    # Sample from posterior
    df_sample = df.sample(nsamp, replace=True)
    myBasis = Basis(params["basis"], params["n_planets"])
    df_synth = myBasis.to_synth(df)

    # Build epoch sampling
    epochs_sep = Time(np.linspace(t_start.mjd, t_end.mjd, 40), format="mjd")
    times_sep = epochs_sep.decimalyear

    epochs_2d = Time(np.linspace(t_start.mjd, t_end.mjd, 100), format="mjd")

    # =========================================
    # COMPUTE SEPARATIONS
    # =========================================
    seps, _, _ = compute_sep(
        df_sample, epochs_sep,
        params["basis"], params["m0"], params["m0_err"],
        params["plx"], params["plx_err"],
        params["n_planets"], params["pl_num"],
        override_inc=override_inc,
        override_lan=override_lan
    )

    # Stats
    med_sep = np.median(seps, axis=1)
    low_sep = np.percentile(seps, 16, axis=1)
    high_sep = np.percentile(seps, 84, axis=1)

    # Visibility fractions
    IWA = 155
    OWA = 436
    IWAw = 450
    OWAw = 1300
    visible_frac_narrow = np.mean((seps >= IWA) & (seps <= OWA), axis=1) * 100
    visible_frac_wide = np.mean((seps >= IWAw) & (seps <= OWAw), axis=1) * 100

    # =========================================
    # COMPUTE FOR 2D ORBIT
    # =========================================
    seps_2d, raoff_2d, deoff_2d = compute_sep(
        df_synth, epochs_2d,
        params["basis"], params["m0"], params["m0_err"],
        params["plx"], params["plx_err"],
        params["n_planets"], params["pl_num"],
        override_inc=override_inc,
        override_lan=override_lan
    )

    best_idx = int(df_synth["lnprobability"].idxmax())

    # =========================================
    # CREATE PLOT WITH PLASMA COLORS
    # =========================================
    fig = plt.figure(figsize=(22, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.5], width_ratios=[1.5, 1, 1], hspace=0.3, wspace=0.3)

    # Plasma colormap colors
    cm = plt.cm.plasma
    c_iwa_narrow = cm(0.85)    # bright yellow-orange
    c_iwa_wide = cm(0.5)       # magenta
    c_orbit_light = cm(0.2)    # dark purple
    c_orbit_best = cm(0.95)    # bright yellow
    c_timestamp = cm(0.0)      # dark purple/blue
    c_start = cm(0.7)          # orange
    c_end = cm(0.3)            # purple
    c_star = cm(0.0)           # dark purple/blue
    c_median = cm(0.6)         # orange
    c_fill = cm(0.2)           # dark purple
    c_samples = cm(0.15)       # very dark purple

    # PLOT 1: 2D ORBIT
    ax1 = fig.add_subplot(gs[:, 0])
    title_2d = f"{planet}: Orbital Trajectory (i={'random' if override_inc is None else f'{override_inc}°'}"
    if override_lan is not None:
        title_2d += f", Ω={override_lan}°)"
    else:
        title_2d += ", Ω=random)"
    ax1.set_title(title_2d, fontsize=14)
    ax1.set_xlabel("RA Offset [mas]", fontsize=14)
    ax1.set_ylabel("Dec Offset [mas]", fontsize=14)

    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(IWA*np.cos(theta), IWA*np.sin(theta), color=c_iwa_narrow, lw=4, linestyle='--', label='IWA/OWA (Narrow)')
    ax1.plot(OWA*np.cos(theta), OWA*np.sin(theta), color=c_iwa_narrow, lw=4, linestyle='--')

    n_samples = min(100, raoff_2d.shape[1])
    sample_indices = np.random.choice(raoff_2d.shape[1], n_samples, replace=False)
    for i in sample_indices:
        ax1.plot(raoff_2d[:, i], deoff_2d[:, i], '-', color=c_orbit_light, alpha=0.3, linewidth=0.5)

    ax1.plot(raoff_2d[:, best_idx], deoff_2d[:, best_idx], '-', color=c_orbit_light, linewidth=5, label='Best-fit orbit', zorder=10)

    timestamp_dates = ['2027-01-01', '2027-06-01', '2028-01-01', '2030-01-01']
    timestamp_epochs = Time(timestamp_dates, format='iso')
    ra_interp = interp1d(epochs_2d.mjd, raoff_2d[:, best_idx], kind='cubic', fill_value='extrapolate')
    de_interp = interp1d(epochs_2d.mjd, deoff_2d[:, best_idx], kind='cubic', fill_value='extrapolate')
    raoff_ts = ra_interp(timestamp_epochs.mjd)
    deoff_ts = de_interp(timestamp_epochs.mjd)

    for i, date in enumerate(timestamp_dates):
        ax1.plot(raoff_ts[i], deoff_ts[i], 'o', color=c_timestamp, markersize=10, zorder=13)
        ax1.annotate(date, xy=(raoff_ts[i], deoff_ts[i]), xytext=(10, 10), textcoords='offset points',
                     fontsize=10, color=c_timestamp, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7), zorder=14)

    ax1.plot(raoff_2d[0, best_idx], deoff_2d[0, best_idx], 'o', color=c_start, markersize=12, label=f'Start ({start_date_str})', zorder=12)
    ax1.plot(raoff_2d[-1, best_idx], deoff_2d[-1, best_idx], 'o', color=c_end, markersize=12, label=f'End ({end_date_str})', zorder=12)
    ax1.plot(0, 0, '*', color=c_star, markersize=20, label='Star', zorder=15)

    ax1.set_aspect('equal')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    #ax1.legend(loc='best', fontsize=10)

    # PLOT 2: SEPARATION VS TIME
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.set_title("Separation vs Time", fontsize=14)
    ax2.set_ylabel("Separation [mas]", fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    ax2.plot(times_sep, med_sep, '-', color=c_median, linewidth=2, label='Median separation', marker='o', markersize=4)
    ax2.fill_between(times_sep, low_sep, high_sep, color=c_fill, alpha=0.5, label='1σ interval')

    n_plot = min(20, seps.shape[1])
    idxs = np.random.choice(seps.shape[1], n_plot, replace=False)
    for i in idxs:
        ax2.plot(times_sep, seps[:, i], color=c_samples, linewidth=1, alpha=0.15)

    ax2.axhline(y=IWA, color=c_iwa_narrow, linestyle='--', linewidth=4, label='IWA/OWA (Narrow)')
    ax2.axhline(y=OWA, color=c_iwa_narrow, linestyle='--', linewidth=4)
    ax2.axhline(y=IWAw, color=c_iwa_wide, linestyle='--', linewidth=4, label='IWA/OWA (Wide)')
    ax2.axhline(y=OWAw, color=c_iwa_wide, linestyle='--', linewidth=4)

    ax2.legend(loc='best', fontsize=10)

    # PLOT 3: VISIBILITY FRACTION
    ax3 = fig.add_subplot(gs[1, 1:], sharex=ax2)
    ax3.set_title("Visibility Fraction", fontsize=14)
    ax3.set_xlabel("Year", fontsize=14)
    ax3.set_ylabel("Visible [%]", fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=12)

    ax3.plot(times_sep, visible_frac_narrow, color=c_iwa_narrow, linewidth=2, marker='o', markersize=4, label='Narrow (155-436 mas)')
    ax3.fill_between(times_sep, 0, visible_frac_narrow, color=c_iwa_narrow, alpha=0.2)
    ax3.plot(times_sep, visible_frac_wide, color=c_iwa_wide, linewidth=2, marker='s', markersize=4, label='Wide (450-1300 mas)')
    ax3.fill_between(times_sep, 0, visible_frac_wide, color=c_iwa_wide, alpha=0.2)

    ax3.set_ylim([0, 100])
    ax3.legend(loc='best', fontsize=10)

    plt.suptitle(f"{planet}: {start_date_str} → {end_date_str}", fontsize=16, y=0.98)
    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plot_b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)

    return render_template_string(
        HTML, planets=orbit_params.keys(), plot_img=plot_b64, error=None,
        selected_planet=planet, start_date=start_date_str, end_date=end_date_str,
        inclination=inc_str, lan=lan_str, nsamp=nsamp_str
    )


if __name__ == "__main__":
    app.run(host=os.getenv('HOST', '0.0.0.0'), port=int(os.getenv('PORT', 8080)), debug=False)