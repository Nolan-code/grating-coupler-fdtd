import meep as mp
import numpy as np
import math
import matplotlib.pyplot as plt

# --- Fixed physical parameters ---
resolution = 150               # pixels/um
lam_cen = 1.55                  # center wavelength (um)
theta = math.radians(10)        # incidence angle
Si = mp.Medium(epsilon=12.0)     # silicon refractive index (as permittivity)
h = 0.22                          # waveguide height (um)
gN = 20                            # number of teeth
t_slab = 0.07                       # height of the base in the etched region (non-etched slab)
t_etch = h - t_slab                  # height of the teeth (etched depth)

dpml = 1.0             # thickness of the absorbing (PML) layer
margin_before = 8.0    # distance between the PML and the grating
dair = 2.0              # air height above and below the slab
Lwg = 8.0                # output waveguide length
d_standoff = 1.0          # height between the slab and the source
fcen = 1 / lam_cen         # center frequency
df = 0.05 * fcen            # pulsed source bandwidth
beam_kdir = mp.Vector3(math.sin(theta), -math.cos(theta), 0)   # source propagation direction vector


def snap_to_grid(value, resolution):
    """Round a physical dimension to the nearest multiple of one pixel, to avoid
    subpixel-averaging artifacts (see the paper's numerical validation chapter)."""
    px = 1 / resolution
    return round(value / px) * px


def run_grating(gp_test, gdc_test):
    """Build and run one FDTD simulation of the grating coupler for given (gp, gdc),
    using the real tilted Gaussian beam source. Returns the coupled power and the
    geometry parameters needed to run a matching normalization simulation."""
    gp_test = snap_to_grid(gp_test, resolution)
    t_slab_snap = snap_to_grid(t_slab, resolution)
    t_etch_snap = snap_to_grid(t_etch, resolution)
    tooth_w = snap_to_grid(gp_test * gdc_test, resolution)

    Lgrating = gN * gp_test
    sx = dpml + margin_before + Lgrating + Lwg + dpml
    sy = dpml + dair + h + dair + dpml
    cell_size = mp.Vector3(sx, sy, 0)
    x0 = snap_to_grid(-0.5 * sx + dpml + margin_before, resolution)
    x_wg_start = x0 + Lgrating
    x_focus = x0 + Lgrating / 2

    boundary_layers = [mp.PML(dpml)]

    # continuous non-etched base, spanning the grating + output waveguide
    geometry = [mp.Block(
        size=mp.Vector3(Lgrating + Lwg + dpml, t_slab_snap, mp.inf),
        center=mp.Vector3(x0 + (Lgrating + Lwg + dpml) / 2, -h/2 + t_slab_snap/2, 0),
        material=Si,
    )]
    # individual grating teeth
    for i in range(gN):
        xc = x0 + i * gp_test + tooth_w / 2
        geometry.append(mp.Block(
            size=mp.Vector3(tooth_w, t_etch_snap, mp.inf),
            center=mp.Vector3(xc, -h/2 + t_slab_snap + t_etch_snap/2, 0),
            material=Si,
        ))
    # output waveguide, full thickness
    geometry.append(mp.Block(
        size=mp.Vector3(Lwg + dpml, t_etch_snap, mp.inf),
        center=mp.Vector3(x_wg_start + (Lwg + dpml) / 2, -h/2 + t_slab_snap + t_etch_snap/2, 0),
        material=Si,
    ))

    y_src = h / 2 + d_standoff
    sources = [mp.GaussianBeam2DSource(
        src=mp.GaussianSource(fcen, fwidth=df),
        center=mp.Vector3(x_focus, y_src, 0),
        size=mp.Vector3(20, 0, 0),
        beam_x0=mp.Vector3(0, -d_standoff, 0),
        beam_kdir=beam_kdir,
        beam_w0=5.2,
        beam_E0=mp.Vector3(0, 0, 1),
    )]

    sim = mp.Simulation(resolution=resolution, cell_size=cell_size,
                         boundary_layers=boundary_layers, geometry=geometry, sources=sources)

    mon_pt = mp.Vector3(x_wg_start + 0.7 * Lwg, 0, 0)
    flux = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=mon_pt, size=mp.Vector3(0, sy - 2*dpml, 0)))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt, 1e-9))
    res = sim.get_eigenmode_coefficients(flux, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y)
    coupled = abs(res.alpha[0, 0, 0]) ** 2
    sim.reset_meep()
    return coupled, x_focus, y_src, sx, sy


def get_incident_power(x_focus, y_src, sx, sy):
    """Normalization run: same source, no grating geometry, to measure the total
    incident power crossing the plane where the grating surface would be."""
    sources_norm = [mp.GaussianBeam2DSource(
        src=mp.GaussianSource(fcen, fwidth=df),
        center=mp.Vector3(x_focus, y_src, 0),
        size=mp.Vector3(20, 0, 0),
        beam_x0=mp.Vector3(0, -d_standoff, 0),
        beam_kdir=beam_kdir,
        beam_w0=5.2,
        beam_E0=mp.Vector3(0, 0, 1),
    )]
    sim_norm = mp.Simulation(resolution=resolution, cell_size=mp.Vector3(sx, sy, 0),
                              boundary_layers=[mp.PML(dpml)], sources=sources_norm, geometry=[])
    norm_pt = mp.Vector3(x_focus, h/2, 0)
    norm_flux = sim_norm.add_flux(fcen, 0, 1, mp.FluxRegion(center=norm_pt, size=mp.Vector3(20, 0, 0)))
    sim_norm.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, norm_pt, 1e-9))
    return abs(mp.get_fluxes(norm_flux)[0])


# --- Sweep 1: grating period Lambda, gdc fixed at 0.5 (reproduces the paper's Table 5.1) ---
# Edit this list to test other period values:
gp_values = [0.50, 0.60, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.90]
gp_results = []

for gpv in gp_values:
    coupled, x_focus, y_src, sx, sy = run_grating(gpv, 0.50)
    if gpv == gp_values[0]:
        # incident_power only depends on the source, not on the grating geometry,
        # so it is computed once and reused for every point in both sweeps below.
        incident_power = get_incident_power(x_focus, y_src, sx, sy)
    eff_dB = 10 * np.log10(coupled / incident_power)
    gp_results.append(eff_dB)
    print(f"gp={gpv:.2f} um -> {eff_dB:.2f} dB", flush=True)

# --- Sweep 2: duty cycle, Lambda fixed at 0.72 um (reproduces the paper's Table 5.2) ---
# Edit this list to test other duty-cycle values:
gdc_values = [0.40, 0.50, 0.60, 0.65, 0.67, 0.69, 0.70, 0.71, 0.73, 0.75, 0.80]
gdc_results = []

for gdcv in gdc_values:
    coupled, x_focus, y_src, sx, sy = run_grating(0.72, gdcv)
    eff_dB = 10 * np.log10(coupled / incident_power)
    gdc_results.append(eff_dB)
    print(f"gdc={gdcv:.2f} -> {eff_dB:.2f} dB", flush=True)

# --- Save the raw data ---
np.savetxt("balayage_gp.csv", np.column_stack([gp_values, gp_results]), delimiter=",", header="gp,eff_dB")
np.savetxt("balayage_gdc.csv", np.column_stack([gdc_values, gdc_results]), delimiter=",", header="gdc,eff_dB")

# --- Plots ---
fig, ax = plt.subplots()
ax.plot(gp_values, gp_results, "o-")
ax.set_xlabel("Λ (µm)")
ax.set_ylabel("Efficiency (dB)")
ax.set_title("Grating period sweep")
plt.savefig("gp_r150.png", dpi=150, bbox_inches="tight")

fig, ax = plt.subplots()
ax.plot(gdc_values, gdc_results, "o-")
ax.set_xlabel("Duty cycle")
ax.set_ylabel("Efficiency (dB)")
ax.set_title("Duty cycle sweep")
plt.savefig("gdc_r150.png", dpi=150, bbox_inches="tight")
