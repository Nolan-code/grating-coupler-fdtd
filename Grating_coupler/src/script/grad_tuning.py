from autograd.extend import primitive, defvjp
from autograd import numpy as npa
from autograd import tensor_jacobian_product
import meep as mp
import meep.adjoint as mpa
import numpy as np
import math
import cmath
import nlopt

# --- Physical parameters ---
resolution = 150            # pixels/um
lam_cen = 1.55               # center wavelength (um)
theta = math.radians(10)     # fiber incidence angle w.r.t. normal (rad)
fcen = 1 / lam_cen            # center frequency corresponding to lam_cen (Meep units: c=1)
k0 = 2 * np.pi / lam_cen       # vacuum wavevector (2*pi/lambda)

Si = mp.Medium(epsilon=12.0)     # silicon material
Air = mp.Medium(index=1.0)       # air material
h = 0.22                          # total silicon slab thickness (um)
t_slab = 0.07                     # thickness of the non-etched base (um)
t_etch = h - t_slab                # thickness of the etched (grating) part (um)
gN = 5                              # number of teeth used ONLY for the (unused) design-region sizing convention below;
                                     # the design region itself is fixed (see design_region_size_x), not gN*gp

# --- Domain ---
dpml = 1.0                 # PML absorbing layer thickness (um)
margin_before = 8.0        # air margin before the grating (um)
dair = 2.0                 # air margin above/below the slab (um)
Lwg = 8.0                   # output waveguide length (um)
sy = dpml + dair + h + dair + dpml   # total simulation domain height (um)

# --- Source ---
d_standoff = 1.0            # vertical distance between the source line and the grating surface (um)
y_src = h/2 + d_standoff      # y position of the source
df = 0.05 * fcen               # pulsed source bandwidth

# --- Design region: FIXED size, matches the geometry used to produce the report's gradient results ---
design_region_resolution = 10 * resolution
design_region_size_x = 20 * 0.72   # 14.4 um (fixed length, independent of the gp being tested)
design_region_size_y = h
design_region_size = mp.Vector3(design_region_size_x, design_region_size_y, 0)
nx_sim_grid = int(design_region_size.x * resolution) + 1
ny_sim_grid = int(design_region_size.y * resolution) + 1

# --- Fixed geometry (used for the normalization run too) ---
sx_fixed = dpml + margin_before + design_region_size_x + Lwg + dpml
x0_fixed = -0.5 * sx_fixed + dpml + margin_before
x_focus_fixed = x0_fixed + design_region_size_x / 2


# --- Tilted-phase source (compatible with the adjoint solver) ---
# A plain mp.Source with a linear phase ramp reproduces a tilted-incidence plane wave
# (the same phase-matching physics as the tilted Gaussian beam), without triggering the
# adjoint/GaussianBeam2DSource incompatibility described in the report.
def tilted_phase(pos):
    return cmath.exp(1j * k0 * np.sin(theta) * pos.x)


# --- Helper functions building the periodic teeth pattern from (gp, gdc) ---
def design_region_to_meshgrid(nx: int, ny: int):
    xcoord = np.linspace(-0.5 * design_region_size.x, +0.5 * design_region_size.x, nx)
    ycoord = np.linspace(-0.5 * design_region_size.y, +0.5 * design_region_size.y, ny)
    xv, yv = np.meshgrid(xcoord, ycoord, indexing="ij")
    return xv, yv


@primitive
def teeth_weight(params):
    """Binary Si/air weight map for the design region, from params = [gp, gdc]."""
    gp, gdc = params
    xv, yv = design_region_to_meshgrid(nx_sim_grid, ny_sim_grid)
    x_loc = xv - xv[0][0]          # shift the origin to the start of the grid instead of its middle
    x_mod = np.mod(x_loc, gp)      # position within the current period
    weights = np.where(x_mod <= gdc * gp, 1, 0)
    return weights.flatten()


def teeth_weight_vjp(ans, params):
    """Finite-difference vector-Jacobian product, so autograd can differentiate teeth_weight
    with respect to (gp, gdc) even though it is a non-differentiable binary mask."""
    gp, gdc = params
    eps = 1.0 / resolution   # perturbation size

    w0 = teeth_weight([gp, gdc])
    w_dgp = teeth_weight([gp + eps, gdc])
    w_dgdc = teeth_weight([gp, gdc + eps])

    jac_gp = (w_dgp - w0) / eps      # d(rho)/d(gp) per pixel
    jac_gdc = (w_dgdc - w0) / eps    # d(rho)/d(gdc) per pixel

    def vjp(g):
        # g: adjoint gradient (per pixel)
        dF_dgp = np.tensordot(g, jac_gp, axes=1)
        dF_dgdc = np.tensordot(g, jac_gdc, axes=1)
        return np.array([dF_dgp, dF_dgdc])

    return vjp


defvjp(teeth_weight, teeth_weight_vjp)


# --- Build the Meep simulation + adjoint optimization problem for a given (gp, gdc) ---
def build_sim(params):
    gp, gdc = params

    sx_local = dpml + margin_before + design_region_size_x + Lwg + dpml
    cell_size_local = mp.Vector3(sx_local, sy, 0)
    x0_local = -0.5 * sx_local + dpml + margin_before
    x_wg_start_local = x0_local + design_region_size_x
    x_focus_local = x0_local + design_region_size_x / 2

    design_variables = mp.MaterialGrid(
        mp.Vector3(nx_sim_grid, ny_sim_grid), Air, Si,
        weights=np.ones((nx_sim_grid, ny_sim_grid)),   # placeholder; overwritten by teeth_weight(x) at each call
        do_averaging=False,
    )
    design_region = mpa.DesignRegion(
        design_variables,
        volume=mp.Volume(
            center=mp.Vector3(x0_local + design_region_size_x/2, -h/2 + t_slab + t_etch/2, 0),
            size=design_region_size,
        ),
    )

    sources = [mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(x_focus_local, y_src, 0),
        size=mp.Vector3(20, 0, 0),
        amp_func=tilted_phase,
    )]

    geometry = [
        # continuous non-etched base
        mp.Block(
            size=mp.Vector3(design_region_size_x + Lwg + dpml, t_slab, mp.inf),
            center=mp.Vector3(x0_local + (design_region_size_x+Lwg+dpml)/2, -h/2 + t_slab/2, 0),
            material=Si,
        ),
        # output waveguide (full thickness)
        mp.Block(
            size=mp.Vector3(Lwg + dpml, t_etch, mp.inf),
            center=mp.Vector3(x_wg_start_local + (Lwg+dpml)/2, -h/2 + t_slab + t_etch/2, 0),
            material=Si,
        ),
        # design region (replaces the individual teeth blocks)
        mp.Block(
            center=design_region.center,
            size=design_region.size,
            material=design_variables,
        ),
    ]

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size_local,
        boundary_layers=[mp.PML(dpml)],
        sources=sources,
        geometry=geometry,
    )

    mon_pt_local = mp.Vector3(x_wg_start_local + 0.7 * Lwg, 0, 0)

    obj_args = [mpa.EigenmodeCoefficient(
        sim, mp.Volume(center=mon_pt_local, size=mp.Vector3(0, sy - 2*dpml, 0)), mode=1,
        eig_parity=mp.ODD_Z + mp.EVEN_Y,
    )]

    def obj_func(mode_coeff):
        return npa.abs(mode_coeff) ** 2

    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=obj_func,
        objective_arguments=obj_args,
        design_regions=[design_region],
        frequencies=[fcen],
        decay_by=1e-9,
    )

    return opt


# --- Normalization run (same tilted-phase source, no grating geometry) ---
def get_incident_power(x_focus, y_src, sx, sy):
    sources_norm = [mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(x_focus, y_src, 0),
        size=mp.Vector3(20, 0, 0),
        amp_func=tilted_phase,
    )]
    sim_norm = mp.Simulation(
        resolution=resolution,
        cell_size=mp.Vector3(sx, sy, 0),
        boundary_layers=[mp.PML(dpml)],
        sources=sources_norm,
        geometry=[],
    )
    norm_pt = mp.Vector3(x_focus, h/2, 0)
    norm_flux = sim_norm.add_flux(fcen, 0, 1, mp.FluxRegion(center=norm_pt, size=mp.Vector3(20, 0, 0)))
    sim_norm.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, norm_pt, 1e-9))
    return abs(mp.get_fluxes(norm_flux)[0])


# --- Compute the incident power ONCE, before optimization starts ---
if mp.am_master():
    print("Computing incident_power...", flush=True)
incident_power = get_incident_power(x_focus_fixed, y_src, sx_fixed, sy)
if mp.am_master():
    print("incident_power =", incident_power, flush=True)


# --- Objective function passed to nlopt ---
def f(x, grad):
    if mp.am_master():
        print(f"--- Testing: gp={x[0]:.4f}, gdc={x[1]:.4f} ---", flush=True)
    weights = teeth_weight(x)
    opt = build_sim(x)
    val, grad_pixels = opt([weights], need_gradient=True)

    if grad.size > 0:
        grad_params = tensor_jacobian_product(teeth_weight, 0)(x, grad_pixels)
        grad[:] = grad_params

    if mp.am_master():
        efficiency_dB = 10 * np.log10(val[0] / incident_power)
        print(f"gp={x[0]:.4f}, gdc={x[1]:.4f} -> val={val[0]:.4f}, efficiency={efficiency_dB:.2f} dB", flush=True)

    return float(val[0])


# --- Optimization setup ---
# NOTE: bounds and starting point below match the run reported in the paper (Table 5.3).
# Feel free to edit them to explore a different region of the (gp, gdc) parameter space.
solver = nlopt.opt(nlopt.LD_MMA, 2)
solver.set_lower_bounds([0.65, 0.55])
solver.set_upper_bounds([0.80, 0.80])
solver.set_max_objective(f)
solver.set_initial_step([0.01, 0.01])
solver.set_ftol_rel(1e-4)     # stopping criterion (relative change in objective)
solver.set_maxeval(15)

x_init = np.array([0.7435, 0.6717])  # starting point (gp, gdc) -- edit to try a different one

x_opt = solver.optimize(x_init)
if mp.am_master():
    print("Final result: gp =", x_opt[0], " gdc =", x_opt[1])
    print("Return code:", solver.last_optimize_result())
