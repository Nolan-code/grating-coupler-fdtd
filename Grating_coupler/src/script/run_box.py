import meep as mp
import numpy as np
import math

# --- Physical parameters ---
resolution = 150
lam_cen = 1.55
theta = math.radians(10)
fcen = 1 / lam_cen

Si = mp.Medium(epsilon=12.0)
Air = mp.Medium(index=1.0)
box_material = mp.Medium(index=1.44)   # buried oxide (BOX) reflector
n_BOX = 1.44
h = 0.22
t_slab = 0.07
t_etch = h - t_slab
gN = 20

# --- Domain ---
dpml = 1.0
margin_before = 8.0
dair = 2.0
Lwg = 8.0
sy = dpml + dair + h + dair + dpml

# --- Source ---
d_standoff = 1.0
y_src = h/2 + d_standoff
df = 0.05 * fcen
beam_kdir = mp.Vector3(math.sin(theta), -math.cos(theta), 0)

# --- Design region (only used here to keep the grid-sizing convention consistent
# with the adjoint-optimization scripts; not otherwise used in this file) ---
design_region_size_x = 20 * 0.72   # 14.4 um
design_region_size_y = h
design_region_size = mp.Vector3(design_region_size_x, design_region_size_y, 0)
nx_sim_grid = int(design_region_size.x * resolution) + 1
ny_sim_grid = int(design_region_size.y * resolution) + 1

# --- Fixed geometry ---
sx = dpml + margin_before + design_region_size_x + Lwg + dpml
cell_size = mp.Vector3(sx, sy, 0)
x0 = -0.5 * sx + dpml + margin_before
x_wg_start = x0 + design_region_size_x
x_focus = x0 + design_region_size_x / 2


def design_region_to_meshgrid(nx, ny):
    xcoord = np.linspace(-0.5 * design_region_size.x, +0.5 * design_region_size.x, nx)
    ycoord = np.linspace(-0.5 * design_region_size.y, +0.5 * design_region_size.y, ny)
    xv, yv = np.meshgrid(xcoord, ycoord, indexing="ij")
    return xv, yv


def teeth_weight(params):
    gp, gdc = params
    xv, yv = design_region_to_meshgrid(nx_sim_grid, ny_sim_grid)
    x_loc = xv - xv[0][0]
    x_mod = np.mod(x_loc, gp)
    weights = np.where(x_mod <= gdc * gp, 1, 0)
    return weights


def make_gaussian_source():
    return [mp.GaussianBeam2DSource(
        src=mp.GaussianSource(fcen, fwidth=df),
        center=mp.Vector3(x_focus, y_src, 0),
        size=mp.Vector3(20, 0, 0),
        beam_x0=mp.Vector3(0, -d_standoff, 0),
        beam_kdir=beam_kdir,
        beam_w0=5.2,
        beam_E0=mp.Vector3(0, 0, 1),
    )]


def run_grating_with_box(gp_test, gdc_test, box_thickness):
    """Build and run the grating coupler with a buried-oxide (BOX) reflector of the
    given thickness inserted below the non-etched base, and a silicon substrate
    filling the remainder of the bottom air margin."""
    gp_test = snap_to_grid(gp_test, resolution)
    box_thickness = snap_to_grid(box_thickness, resolution)
    t_slab_snap = snap_to_grid(t_slab, resolution)
    t_etch_snap = snap_to_grid(t_etch, resolution)
    tooth_w = snap_to_grid(gp_test * gdc_test, resolution)

    Lgrating = gN * gp_test
    sx = dpml + margin_before + Lgrating + Lwg + dpml
    sy_local = dpml + dair + h + dair + dpml
    cell_size = mp.Vector3(sx, sy_local, 0)
    x0 = snap_to_grid(-0.5 * sx + dpml + margin_before, resolution)
    x_wg_start = x0 + Lgrating
    x_focus = x0 + Lgrating / 2

    geometry = [mp.Block(
        size=mp.Vector3(Lgrating + Lwg + dpml, t_slab_snap, mp.inf),
        center=mp.Vector3(x0 + (Lgrating + Lwg + dpml) / 2, -h/2 + t_slab_snap/2, 0),
        material=Si,
    )]
    for i in range(gN):
        xc = x0 + i * gp_test + tooth_w / 2
        geometry.append(mp.Block(
            size=mp.Vector3(tooth_w, t_etch_snap, mp.inf),
            center=mp.Vector3(xc, -h/2 + t_slab_snap + t_etch_snap/2, 0),
            material=Si,
        ))
    geometry.append(mp.Block(
        size=mp.Vector3(Lwg + dpml, t_etch_snap, mp.inf),
        center=mp.Vector3(x_wg_start + (Lwg + dpml) / 2, -h/2 + t_slab_snap + t_etch_snap/2, 0),
        material=Si,
    ))

    # --- BOX reflector + silicon substrate, below the base ---
    geometry.append(mp.Block(
        size=mp.Vector3(Lgrating + Lwg + dpml, box_thickness, mp.inf),
        center=mp.Vector3(x0 + (Lgrating + Lwg + dpml) / 2, -h/2 - box_thickness/2, 0),
        material=box_material,
    ))
    remaining = dair - box_thickness   # fill the rest of the bottom air margin with Si substrate
    if remaining > 0:
        geometry.append(mp.Block(
            size=mp.Vector3(Lgrating + Lwg + dpml, remaining, mp.inf),
            center=mp.Vector3(x0 + (Lgrating + Lwg + dpml) / 2, -h/2 - box_thickness - remaining/2, 0),
            material=Si,
        ))

    y_src = h / 2 + d_standoff
    sources = [mp.GaussianBeam2DSource(
        src=mp.GaussianSource(fcen, fwidth=df),
        center=mp.Vector3(x_focus, y_src, 0), size=mp.Vector3(20, 0, 0),
        beam_x0=mp.Vector3(0, -d_standoff, 0), beam_kdir=beam_kdir,
        beam_w0=5.2, beam_E0=mp.Vector3(0, 0, 1),
    )]

    sim = mp.Simulation(resolution=resolution, cell_size=cell_size,
                         boundary_layers=[mp.PML(dpml)], geometry=geometry, sources=sources)
    mon_pt = mp.Vector3(x_wg_start + 0.7 * Lwg, 0, 0)
    flux = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=mon_pt, size=mp.Vector3(0, sy_local-2*dpml, 0)))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt, 1e-9))
    res = sim.get_eigenmode_coefficients(flux, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y)
    coupled = abs(res.alpha[0, 0, 0]) ** 2
    sim.reset_meep()
    return coupled


def get_incident_power(gp_test, gdc_test):
    """Normalization run (no geometry), using the grating's true gN*gp length so
    that the domain/source geometry matches run_grating_with_box exactly."""
    gp_test = snap_to_grid(gp_test, resolution)
    Lgrating = gN * gp_test
    sx_local = dpml + margin_before + Lgrating + Lwg + dpml
    sy_local = dpml + dair + h + dair + dpml
    x0_local = snap_to_grid(-0.5 * sx_local + dpml + margin_before, resolution)
    x_focus_local = x0_local + Lgrating / 2
    y_src_local = h/2 + d_standoff

    sources = [mp.GaussianBeam2DSource(
        src=mp.GaussianSource(fcen, fwidth=df),
        center=mp.Vector3(x_focus_local, y_src_local, 0), size=mp.Vector3(20, 0, 0),
        beam_x0=mp.Vector3(0, -d_standoff, 0), beam_kdir=beam_kdir,
        beam_w0=5.2, beam_E0=mp.Vector3(0, 0, 1),
    )]
    sim_norm = mp.Simulation(resolution=resolution, cell_size=mp.Vector3(sx_local, sy_local, 0),
                              boundary_layers=[mp.PML(dpml)], sources=sources, geometry=[])
    norm_pt = mp.Vector3(x_focus_local, h/2, 0)
    norm_flux = sim_norm.add_flux(fcen, 0, 1, mp.FluxRegion(center=norm_pt, size=mp.Vector3(20, 0, 0)))
    sim_norm.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, norm_pt, 1e-9))
    return abs(mp.get_fluxes(norm_flux)[0])


def snap_to_grid(value, resolution):
    """Round a physical dimension to the nearest multiple of one pixel."""
    px = 1 / resolution
    return round(value / px) * px


# --- Sweep, at the fixed optimal (gp, gdc) found by the gradient optimization ---
gp_opt, gdc_opt = 0.7596, 0.6819

# Edit this list to test other BOX thicknesses (values below reproduce the paper's Table 5.5):
box_values = [0.03, 0.04, 0.05, 0.06, 0.09, 0.12, 0.15, 0.20, 0.27, 0.350, 0.450, 0.600, 0.807, 1.000]

incident_power = get_incident_power(gp_opt, gdc_opt)
for bt in box_values:
    coupled = run_grating_with_box(gp_opt, gdc_opt, bt)
    eff_dB = 10 * np.log10(coupled / incident_power)
    print(f"box_thickness={bt:.3f} um -> {eff_dB:.2f} dB", flush=True)
