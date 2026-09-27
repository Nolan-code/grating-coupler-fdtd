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
h = 0.22
t_slab = 0.07
t_etch = h - t_slab

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

# --- Design region: FIXED size, identical to the one used during the adjoint optimization ---
design_region_size_x = 20 * 0.72   # 14.4 um
design_region_size_y = h
design_region_size = mp.Vector3(design_region_size_x, design_region_size_y, 0)
nx_sim_grid = int(design_region_size.x * resolution) + 1
ny_sim_grid = int(design_region_size.y * resolution) + 1

# --- Fixed geometry, consistent with the adjoint optimization's build_sim ---
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
    """Binary Si/air weight map for the design region, from params = [gp, gdc]."""
    gp, gdc = params
    xv, yv = design_region_to_meshgrid(nx_sim_grid, ny_sim_grid)
    x_loc = xv - xv[0][0]
    x_mod = np.mod(x_loc, gp)
    weights = np.where(x_mod <= gdc * gp, 1, 0)
    return weights


def make_gaussian_source():
    """The real tilted Gaussian beam source (fiber mode), used for the final,
    physically accurate efficiency figure reported in the paper."""
    return [mp.GaussianBeam2DSource(
        src=mp.GaussianSource(fcen, fwidth=df),
        center=mp.Vector3(x_focus, y_src, 0),
        size=mp.Vector3(20, 0, 0),
        beam_x0=mp.Vector3(0, -d_standoff, 0),
        beam_kdir=beam_kdir,
        beam_w0=5.2,
        beam_E0=mp.Vector3(0, 0, 1),
    )]


def get_incident_power():
    """Normalization run (no grating geometry), to measure the total power the
    fiber beam delivers at the height where the grating surface would be."""
    sim_norm = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=[mp.PML(dpml)],
        sources=make_gaussian_source(),
        geometry=[],
    )
    norm_pt = mp.Vector3(x_focus, h/2, 0)
    norm_flux = sim_norm.add_flux(fcen, 0, 1, mp.FluxRegion(center=norm_pt, size=mp.Vector3(20, 0, 0)))
    sim_norm.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, norm_pt, 1e-9))
    return abs(mp.get_fluxes(norm_flux)[0])


def build_sim_gaussian(params):
    """Build and run the full grating-coupler simulation for a given (gp, gdc),
    with the real Gaussian beam source, and return the coupled power."""
    weights_2d = teeth_weight(params)

    design_variables = mp.MaterialGrid(
        mp.Vector3(nx_sim_grid, ny_sim_grid), Air, Si,
        weights=weights_2d,
        do_averaging=False,
    )

    design_center = mp.Vector3(x0 + design_region_size_x/2, -h/2 + t_slab + t_etch/2, 0)

    geometry = [
        # continuous non-etched base
        mp.Block(
            size=mp.Vector3(design_region_size_x + Lwg + dpml, t_slab, mp.inf),
            center=mp.Vector3(x0 + (design_region_size_x+Lwg+dpml)/2, -h/2 + t_slab/2, 0),
            material=Si,
        ),
        # output waveguide, full thickness
        mp.Block(
            size=mp.Vector3(Lwg + dpml, t_etch, mp.inf),
            center=mp.Vector3(x_wg_start + (Lwg+dpml)/2, -h/2 + t_slab + t_etch/2, 0),
            material=Si,
        ),
        # design region holding the teeth pattern
        mp.Block(
            center=design_center,
            size=design_region_size,
            material=design_variables,
        ),
    ]

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=[mp.PML(dpml)],
        sources=make_gaussian_source(),
        geometry=geometry,
    )

    mon_pt = mp.Vector3(x_wg_start + 0.7 * Lwg, 0, 0)
    flux = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=mon_pt, size=mp.Vector3(0, sy - 2*dpml, 0)))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt, 1e-9))
    res = sim.get_eigenmode_coefficients(flux, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y)
    coupled = abs(res.alpha[0, 0, 0]) ** 2
    return coupled


# --- Step 1: normalization run ---
print("Computing incident_power...", flush=True)
incident_power = get_incident_power()
print("incident_power =", incident_power, flush=True)

# --- Step 2: main run, at the point found by the gradient optimization ---
# Edit these two values to check the efficiency at a different (gp, gdc) point:
x_opt = np.array([0.7596, 0.6819])   # gp, gdc

print("Running main simulation...", flush=True)
coupled_power = build_sim_gaussian(x_opt)

efficiency_dB = 10 * np.log10(coupled_power / incident_power)
print("coupled_power =", coupled_power)
print("efficiency =", efficiency_dB, "dB")
