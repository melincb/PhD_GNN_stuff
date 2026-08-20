import argparse
import pickle
import numpy as np

import torch


parser = argparse.ArgumentParser()
parser.add_argument('--AE_version', help="Version of AE, e.g. 'v5'", type=str, required=True)
parser.add_argument('--VarDA_type', help="'prec_3D-Var' or 'inc_3D-Var'", type=str, default='prec_3D-Var')
parser.add_argument('--FWD_model', help="Which forward model do I use? Options: 'persistence', 'NNfwd'", type=str, required=False, default='persistence')
parser.add_argument("--obs_datetime", help="Date and hour of *observation* in format yyyy-mm-dd-hh or yyyymmddhh", type=str, required=False, default='2023-04-15-00')
parser.add_argument("--forecast_len", help="Number of 1-hourly forecast steps", type=int, required=False, default=24)
parser.add_argument('--init_lr', help='Initial learning rate for SGD optimizer when performing 3D-Var cost function minimization in latent space', type=float, default=0.5)
parser.add_argument('--custom_addon', help="Custom addon to output filenames", type=str, default='', required=False)
parser.add_argument('--pseudo_obs', help="Generate and assimilate pseudo observations", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--obs_inc', help="Observation increment for single observation experiment - if '0.0', regular experiment will be performed with obs. on manually defined grid", type=str, required=False, default='0.0')
parser.add_argument('--singobs_lat', help="Latitude in case of single observation experiment", type=float, required=False, default=np.nan)
parser.add_argument('--singobs_lon', help="Latitude in case of single observation experiment", type=float, required=False, default=np.nan)
parser.add_argument('--plot_singles', help="Plot some figures one by one (only if --plot)", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--obs_qty', help="Observed variable (passed e.g. as Z200,u200)", type=str, required=True)
parser.add_argument('--obs_std', help="Standard deviation of pseudo observations, arbitrary unit (passed in the same order as obs_qty, e.g. as 1.0,2.5). If set to 0.0 in a singobs experiment, the system will set obs_std to the same value as background std in physical space", type=str, required=True)
parser.add_argument('--savefig_dir', help='Directory for saving the figure (if specified, it will be saved to experiments/figures/{in_out_ch}ch/args.savefig_dir; otherwise it will be just saved to experiments/figures/{in_out_ch}ch/)', type=str, default='', required=False)
parser.add_argument('--plot_projection', help="Plotting projection (PlateCarree or NearsidePerspective)", type=str, default='PlateCarree', required=False)
parser.add_argument('--B_type', help="B-matrix version: climatological or ensemble", type=str, required=True)
parser.add_argument('--em_idx', help="Ensemble member index (in case of ensemble B)", type=int, default=0)

args = parser.parse_args()
if '-' not in args.obs_datetime:
    if len(args.obs_datetime) == 10:
        args.obs_datetime = f'{args.obs_datetime[0:4]}-{args.obs_datetime[4:6]}-{args.obs_datetime[6:8]}-{args.obs_datetime[8:10]}'
    else:
        raise AttributeError('Improper obs_datetime format. Should be yyyy-mm-dd-hh or yyyymmddhh.')

assert args.VarDA_type in ('prec_3D-Var')

deterministic = False#True #False


import sys
# print(sys.path)
sys.path.append('../..')
from general_ae_info import ae_props, date_to_dataidx, standardise_destandardise
AE_props = ae_props(args.AE_version)
sys.path.append(str(AE_props.BASE_PATH))
from utilities import clean_state_dict
import datasets, importlib
importlib.reload(datasets)

B_addon = ''
if args.B_type == 'ensemble':
    B_addon = 'ens_B/'
FIGS = f"{AE_props.vPATH}/DA/algorithm-single/experiments/figs/{AE_props.EXP_ID}/{B_addon}"
import os
if not os.path.exists(FIGS):
    os.makedirs(FIGS)

#%%

from interpolator import interpolate_irregular_grid, normalize_longitudes
if deterministic:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import time
import torch.nn as nn
import torch.nn.functional as F

import importlib
import custom_loss
importlib.reload(custom_loss)



import gc
from datetime import datetime, timedelta

tst = datetime.now()
print('Initiated computation', tst)


# Get the indices of the fields corresponding to each quantity
obs_qty_indices = {AE_props.reconstructed_variables[idx]:idx for idx in range(len(AE_props.reconstructed_variables))} # {'u200':0, 'v200':1, 'z200':2}
# Which quantities do we observe?
obs_qty = [q for q in args.obs_qty.split(',')]  # quantities
obs_qty_idx = [obs_qty_indices[q] for q in obs_qty] # quantities corresponding incides
# Get the standard deviations of the observations (value correspond to observed quantities, all observations of a certain quantity have same std)
obs_std = [float(s) for s in args.obs_std.split(',')]
# Get the observation departures (=observation increments) in case of single observation (location) experiment
obs_inc = [float(oi) for oi in args.obs_inc.split(',')]



# -----------------------------
# Device setup
# -----------------------------

free_mem = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
best_gpu = free_mem.index(max(free_mem))
device = torch.device(f"cuda:{best_gpu}" if torch.cuda.is_available() else "cpu")
print(f"Free memory on GPUs: {free_mem}")
print(f"Using device: {device}")


# -----------------------------
# AE setup
# -----------------------------
deterministic = False   # Force cuda to work in a strictly deterministic fashion (an order of magnitude slow-down; useful only for debugging purposes)
if deterministic:
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
importlib.reload(AE_props.modelslib)


if AE_props.use_pooling:
    graphs = []
    for lev in range(AE_props.N_SUBGRAPHS+1):
        graphs.append(torch.load(AE_props.GRAPH / f"{AE_props.EXP_ID}_edge_index_ae_{lev}.pt"))
    pooling_matrices, unpooling_matrices = torch.load(AE_props.GRAPH / f"{AE_props.EXP_ID}_pooling_matrices.pt")
    proper_level_connections = torch.load(AE_props.GRAPH / f"{AE_props.EXP_ID}_{AE_props.level_connections_type}.pt")

    graphs = [g.to(device) for g in graphs]
    pooling_matrices = [p.to(device) for p in pooling_matrices]
    unpooling_matrices = [u.to(device) for u in unpooling_matrices]

else:
    AE_edge_index = torch.load(AE_props.GRAPH / f"{AE_props.EXP_ID}_edge_index_ae.pt")
    AE_edge_index = AE_edge_index.to(device)



AE_model = AE_props.modelslib.ProgressiveGraphAutoencoder(
        in_dim=AE_props.IN_DIM, 
        encoder_hidden_dims=AE_props.HIDDEN_DIMS, 
        decoder_hidden_dims=AE_props.HIDDEN_DIMS[::-1], 
        latent_dim=AE_props.LATENT_DIM, 
        out_dim=AE_props.OUT_DIM,
        graphs = graphs, 
        level_connections = proper_level_connections,
        heads=AE_props.HEADS, 
        gat_dropout=AE_props.GAT_DROPOUT, feature_dropout=AE_props.FEATURE_DROPOUT, latent_dropout=AE_props.LATENT_DROPOUT, latent_noise_std=AE_props.LATENT_NOISE_STD,
        use_residuals=AE_props.use_residuals, use_residualsIO=AE_props.use_residualsIO).to(device)


AE_best_model = torch.load(AE_props.best_model_pth, map_location=device)  # use the latest or best
AE_model.load_state_dict(clean_state_dict(AE_best_model))
AE_model.eval()

AE_scalers_mean = np.load(AE_props.scalers_mean_pth)
AE_scalers_std  = np.load(AE_props.scalers_std_pth)

print('\nClimatological mean and std of observed variables:')
for q in obs_qty:
    print(q)
    print(AE_scalers_mean[q])
    print(AE_scalers_std[q])
    print()

AE_scalers_mean = torch.tensor([torch.from_numpy(AE_scalers_mean[rv]) for rv in AE_props.reconstructed_variables])
AE_scalers_std = torch.tensor([torch.from_numpy(AE_scalers_std[rv]) for rv in AE_props.reconstructed_variables])



# -----------------------------
# Forward model setup
# -----------------------------
accepted_FWD_models = ('persistence')
if args.FWD_model not in accepted_FWD_models:
    print('Unknown forward model:', args.FWD_model)
    print('Accepted forward models:', accepted_FWD_models)
    raise AttributeError # Unknown forward model
if args.FWD_model == 'persistence':
    pass
else:
    # set FWD model attributes here
    pass

FWD_end_datetime = datetime.strptime(args.obs_datetime, "%Y-%m-%d-%H")
FWD_end_date = (FWD_end_datetime.day, FWD_end_datetime.month, FWD_end_datetime.year)
FWD_end_time = FWD_end_datetime.hour
FWD_start_datetime = FWD_end_datetime - timedelta(hours=args.forecast_len)
FWD_start_date = (FWD_start_datetime.day, FWD_start_datetime.month, FWD_start_datetime.year)
FWD_start_time = FWD_start_datetime.hour



# ------------------------------------------------------
# Prepare background
# ------------------------------------------------------

if args.B_type != 'ensemble':
    # ------------------------------------------------------
    # Load data at the initial and end time of FWD model
    # ------------------------------------------------------
    indices_of_interest = range(date_to_dataidx(FWD_start_datetime), date_to_dataidx(FWD_end_datetime) + 1)
    print(indices_of_interest)

    import datasets
    importlib.reload(datasets)

    dataset_of_interest = datasets.ReallyLazyGraphDataset_v2(
        indices_of_interest, AE_props.DATA, \
        AE_props.time_varying_variables, AE_props.static_variables, \
        AE_props.reconstructed_variables)

    init_truth = dataset_of_interest[0].to(device)
    end_truth = dataset_of_interest[-1].to(device)

    if args.FWD_model == 'persistence':
        physical_bg = init_truth.x
    else:
        raise AttributeError(f'{args.FWD_model} not yet implemented in bg computation')
    
    # import cartopy.crs as ccrs
    # from matplotlib.tri import Triangulation
    # import matplotlib.pyplot as plt
    # grid_lats = np.load("../../../grid_lats.npy")   # -90 to 90
    # grid_lons = np.load("../../../grid_lons.npy")   # 0 to 360
    # grid_lons_plot = normalize_longitudes(grid_lons)    # -180 to 180
    # p1 = standardise_destandardise(physical_bg[:,:-4], AE_props, AE_scalers_mean, AE_scalers_std, 'destandardise', device).cpu().numpy()
    # #p1 = physical_bg.cpu().numpy()
    # print('shape p1', p1.shape)
    # tri = Triangulation(grid_lons_plot, grid_lats)
    # proj = ccrs.Robinson()
    # u700 = np.zeros(np.shape(grid_lats))
    # v700 = np.zeros(np.shape(grid_lats))
    # z700 = np.zeros(np.shape(grid_lats))
    # tot = 1
    # for i in range(len(AE_props.reconstructed_variables)):
    #     c = AE_props.reconstructed_variables[i]
    #     if c == 'u700':
    #         u700 = p1[:,i]
    #     elif c == 'v700':
    #         v700 = p1[:,i]
    #     elif c == 'z700':
    #         z700 = p1[:,i]
    #         print('yes')
    # plt.show()
    # import matplotlib
    # matplotlib.rcParams.update({"font.size": 16})
    # plt.figure(figsize=(12,6))
    # ax = plt.subplot(tot, 1, 1, projection=proj)
    # rf = 25
    # ax.quiver(grid_lons_plot[::rf], grid_lats[::rf], u700[::rf], v700[::rf], transform=ccrs.PlateCarree(), zorder=-1)
    # #tcf = ax.tricontourf(tri, z700, cmap='jet', vmin=np.amin(z700), vmax=np.amax(z700), levels=np.linspace(np.amin(z700), np.amax(z700), 10), transform=ccrs.PlateCarree())
    # tcf = ax.tricontourf(tri, np.sqrt(u700**2 + v700**2), cmap='hot', levels=np.arange(0,40.1,10), extend='max', transform=ccrs.PlateCarree())
    # plt.colorbar(tcf, label='Wind speed [m/s]', fraction=0.02, pad=0.04)
    # # ax.set_xlabel('Longitude')
    # # ax.set_ylabel('Latitude')
    # # ax.grid()
    # ax.coastlines()
    # # rf = 25
    # # ax.quiver(grid_lons_plot[::rf], grid_lats[::rf], u700[::rf], v700[::rf], transform=ccrs.PlateCarree())
    # #ax.quiver(np.array([-100]), np.array([-100]), np.array([1]), np.array([1]), transform=ccrs.PlateCarree())
    # ax.set_title(f'700 hPa background')

    # plt.savefig('bg.pdf')
    # plt.show()

    # raise AssertionError

else:
    # ------------------------------------------------------
    # Load the background
    # ------------------------------------------------------
    dataset_of_interest = datasets.ReallyLazyGraphDataset_v2(
        range(args.em_idx, args.em_idx+1), AE_props.ENSEMBLE_DATA/args.obs_datetime.replace('-',''), \
        AE_props.time_varying_variables, AE_props.static_variables, \
        AE_props.reconstructed_variables)
    physical_bg = dataset_of_interest[0].to(device).x





decoded_bg, latent_bg = AE_model(physical_bg, return_latent=True)
decoded_bg_dest = standardise_destandardise(decoded_bg, AE_props, AE_scalers_mean, AE_scalers_std, 'destandardise', device)

if args.B_type == 'climatological':
    B_matrix_savename = f'{AE_props.vPATH}/DA/B-matrix/matrices/{args.FWD_model}/{AE_props.EXP_ID}_climatological_prediction' +\
                        f'_len_{args.forecast_len}h'
    # if args.idx_reduction > 1:
    #     B_matrix_savename += f'_red_{args.idx_reduction}'
    B_matrix_savename += '_diag'
elif args.B_type == 'ensemble':
    B_matrix_savename = f'{AE_props.vPATH}/DA/B-matrix/matrices/EDA/{AE_props.EXP_ID}_{args.obs_datetime}_diag'
else:
    raise AttributeError('Unsupported B_type:', args.B_type)


B_matrix = torch.load(B_matrix_savename + '.pt', map_location='cpu')
B_matrix_sqrt = torch.sqrt(B_matrix)
B_matrix_sqrt_inv = 1 / torch.sqrt(B_matrix)

# ------------------------------------------------------
# Prepare observations
# ------------------------------------------------------
# grid_lats = torch.from_numpy(np.load(AE_props.DATA / "grid_lats.npy")).to(torch.float32).to(device)   # -90 to 90
# grid_lons = torch.from_numpy(np.load(AE_props.DATA / "grid_lons.npy")).to(torch.float32).to(device)   # 0 to 360
grid_lats = torch.from_numpy(np.load("../../../grid_lats.npy")).to(torch.float32).to(device)   # -90 to 90
grid_lons = torch.from_numpy(np.load("../../../grid_lons.npy")).to(torch.float32).to(device)   # 0 to 360
grid_lons_plot = normalize_longitudes(grid_lons.cpu()).numpy()    # -180 to 180

# Here I generate pseudo observations - the real world observations should be imported
if args.pseudo_obs:
    # Prepare observation locations
    if (args.singobs_lat is not np.nan) and (args.singobs_lon is not np.nan):
        # Single observation (location) experiment
        obs_lats = torch.tensor([args.singobs_lat]).to(device)
        obs_lons = torch.tensor([args.singobs_lon]).to(device)
    elif (args.singobs_lat is not np.nan) and (args.singobs_lon is not np.nan):
        raise AttributeError("Specified either singobs_lat or singobs_lon, but not both")
    else:
        # Define your own grid here
        # Here is an example for obs. at locations [(10.9°N, 1.5°W), (12.1°S, 110.1°E), (12.1°N, 110.1°W), (89.9°N, 50.0°W)]
        obs_lats = torch.from_numpy(np.array([10.9, -12.1, 12.1, 89.9])).to(torch.float32).to(device)
        obs_lons = torch.from_numpy(np.array([-1.5, 110.1, 110.1, -50.])).to(torch.float32).to(device)

    # Prepare observation values
    if args.obs_inc == '0.0':
        # Observations sampled at FWD_end_datetime
        # TBD, not included in thesis
        # end_truth = dataset_of_interest[-1].to(device)
        # print(type(end_truth))
        # print(type(end_truth.x))
        # print(end_truth.x.shape)
        # obs_values = interpolate_irregular_grid(grid_lats, grid_lons, end_truth, obs_lats, obs_lons)
        # Potential problem to be discussed: the indices of the observed variables in end_truth
        # (input vars = time varying data + static data, but not all time varying data = reconstructed data)
        obs_values = interpolate_irregular_grid(grid_lats, grid_lons, decoded_bg_dest[:,obs_qty_idx], obs_lats, obs_lons) # shape (N obs qty, N obs loc)
        print('Background values:', obs_values)
        # obs_values = obs_values + torch.tensor([[0 for iobsloc in range(obs_values.shape[1])] for oinc in obs_inc]).to(device)
        obs_vec = obs_values.reshape((obs_values.shape[0] * obs_values.shape[1], 1)) # shape (N obs qty * N obs loc, 1) = (N obs, 1)
        # raise AttributeError("Option obs_inc=0.0 not yet implemented")

    else:
        obs_values = interpolate_irregular_grid(grid_lats, grid_lons, decoded_bg_dest[:,obs_qty_idx], obs_lats, obs_lons) # shape (N obs qty, N obs loc)
        print('Background values:', obs_values)
        obs_values = obs_values + torch.tensor([[oinc for iobsloc in range(obs_values.shape[1])] for oinc in obs_inc]).to(device)
        obs_vec = obs_values.reshape((obs_values.shape[0] * obs_values.shape[1], 1)) # shape (N obs qty * N obs loc, 1) = (N obs, 1)
        # In case of diagonal R, I could leave obs_vec in the shape of obs_values, but I transform it here in "traditional" shape for tracking reasons
        # (so it will be easier to follow the protocol for real-world observations)
        print('Observed values:',obs_vec)

    # ------------------------------------------------------
    # R-matrix
    # ------------------------------------------------------
    # Assume uncorrelated errors - diagonal R
    # Similarly to Jb term, we can also have
    # Jo = torch.sum(torch.sum((y - Hx) * R * (y - Hx), axis=1)), R has shape (N obs qty * N obs loc, 1)
    R_matrix = torch.tensor([[ostd**2 for iobsloc in range(obs_values.shape[1])] for ostd in obs_std]).reshape((obs_values.shape[0] * obs_values.shape[1], 1))
    R_matrix_inv = 1 / R_matrix
    # shape of R_matrix before reshaping: (N obs qty, N obs loc)
    # after reshaping it needs to be (N obs qty * N obs loc, 1), otherwise we cannot use the superfast multiplication with "*" Jo term

    # print(R_matrix)



# ------------------------------------------------------
# 3D-Var
# ------------------------------------------------------

def latent3DVar_algorithm_preconditioned(
        latent_bg=latent_bg,    # torch tensor, shape (nnode, latent_dim)
        obs_vec=obs_vec,        # torch tensor, shape (nobs, 1)
        B_matrix_sqrt=B_matrix_sqrt,    # torch tensor, shape (nnode, latent_dim), elements correspond to full B diagonal
        obs_lats=obs_lats,  # torch tensor, shape (nobs,)
        obs_lons=obs_lons,  # torch tensor, shape (nobs,)
        obs_qty_idx=obs_qty_idx,    # list with observed qty indices - used for interpolation in H operator
        R_matrix_inv=R_matrix_inv,  # torch tensor, shape (nobs, 1)
        init_lr=args.init_lr,   # float
        max_num_steps=100,
        factor_lr=0.5,  # Factor by which the learning rate is reduced
        rtol_stop=0.01,   # Relative tolerance for convergence criterion
        minimum_lr=1e-4
):
    '''All torch tensors are float32'''
    # Though most of torch tensors are already on GPU, we resend them (for those that are already there this happens instantaneously)
    latent_bg = latent_bg.to(device)
    obs_vec = obs_vec.to(device)
    B_matrix_sqrt = B_matrix_sqrt.to(device)
    obs_lats = obs_lats.to(device)
    obs_lons = obs_lons.to(device)
    R_matrix_inv = R_matrix_inv.to(device)

    corrected_chi = torch.zeros(latent_bg.shape).to(device) # = B_matrix_sqrt_inv @ (latent_bg - latent_bg).clone()
    corrected_chi.requires_grad = True


    # ------------------------------------------------------
    # Prepare minimisation trackers and settings
    # ------------------------------------------------------

    # Setting the optimizer for stochastic gradient descend during 3D-Var
    optimizer = torch.optim.SGD([corrected_chi], lr=init_lr)
    # Variable to keep track of the best loss
    best_J = float('inf')
    # Minimisation step with the best loss
    best_J_step = 0
    # Latent state at the best step
    best_latent_vec = latent_bg.clone().detach() # = (B_matrix_sqrt @ corrected_chi_vec.clone().detach() + latent_bg)


    all_J = [] # Store the values of cost function at each minimisation step
    all_Jb = [] # Store the values of background term at each minimisation step
    all_Jo = [] # Store the values of observation term of cost function at each minimisation step
    all_grad_J = [] # Store the values of the cost function gradient's Euclidean norm at each minimisation step

    # Ending step of the minimisation process (retains this value if the convergence is not reached before that)
    ending_step = max_num_steps

    # ------------------------------------------------------
    # Preconditioned 3D-VAR cost function
    # ------------------------------------------------------
    def latent3DVar_cost():
        # ------------------------------------------------------
        # Background term
        # ------------------------------------------------------

        # If chi was a vector with shape (nnode*latent_dim, 1): J_b = 1/2 * chi**T @ chi
        # The formulation below is equivalent, faster, and works also for chi with shape (nnode, latent_dim)
        J_b = 1 /2 * torch.sum(torch.sum(corrected_chi * corrected_chi, axis=1))

        # ------------------------------------------------------
        # Observation term
        # ------------------------------------------------------
        # J_o = 1/2 * (y - H(S**-1(D(z))))**T R**-1 (y - H(S**-1(D(z))))

        # 1) S**-1(D(z)), z = zb + B_sqrt * chi
        decoded_lat_vec_dest = standardise_destandardise(
            input_fields = AE_model.decode(latent_bg + B_matrix_sqrt * corrected_chi),
            AE_props = AE_props,
            scalers_mean = AE_scalers_mean,
            scalers_std = AE_scalers_std,
            action = 'destandardise',
            device = device
        )

        # 2) Apply observation operator H
        HSDz = interpolate_irregular_grid(grid_lats, grid_lons, decoded_lat_vec_dest[:, obs_qty_idx], obs_lats, obs_lons).T
        # print('HSDz', HSDz)
        # input('....')

        # 3) Compute J_o with a similar speedup as J_b
        J_o = 1 / 2 * torch.sum(torch.sum((obs_vec - HSDz) * R_matrix_inv * (obs_vec - HSDz), axis=1))


        # ------------------------------------------------------
        # Compute sum of terms and return them
        # ------------------------------------------------------

        J = torch.add(J_b, J_o)

        return J, J_b, J_o


    # ------------------------------------------------------
    # Minimisation algorithm
    # ------------------------------------------------------
    for step in range(1, max_num_steps + 1):
        J, Jb, Jo = latent3DVar_cost()  # Get current values of the cost function

        all_J.append(J.item())  # Store current value of the cost function
        all_Jb.append(Jb.item())  # Store current value of the background term
        all_Jo.append(Jo.item())  # Store current value of the observation term
        previous_chi = corrected_chi.clone().detach()  # Store current preconditioned latent vector
        J.backward(retain_graph=True)  # Backpropagate to accumulate gradients
        grad_J = corrected_chi.grad  # Gradient of J with respect to corrected_chi
        norm_grad_J = torch.norm(grad_J, p=2)
        all_grad_J.append(norm_grad_J)  # Store the Euclidean norm of current gradient

        optimizer.step()  # Update corrected_chi according to grad_J and step size
        optimizer.zero_grad()  # Clear the gradients for the next iteration

        # Check for improvement in loss (for learning rate)
        if J < best_J:
            best_J = J
            best_grad_J = norm_grad_J
            best_J_step = step
            best_latent_vec = (B_matrix_sqrt * previous_chi + latent_bg)
        else:
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * factor_lr, minimum_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        # Check for improvement in gradient (for stopping criterion)
        if step >= 2:
            if all_grad_J[-1] / all_grad_J[0] < rtol_stop:
                ending_step = step
                break

        # Monitor the minimisation procedure
        # We print ensemble member index, minimisation step, cost function value,
        # the ratio between the cost function in this and the previous step, number of steps after the last update of the best latent vector
        if step == 1:
            print(f'\nInitial J {J.item():4f}, initial grad J {all_grad_J[0].item():4f}')
        if step > 1:
            print(f'Step {step}, J {J.item():4f}, ratio {all_J[-1] / all_J[-2]:4f},',
                  f'grad J {all_grad_J[-1].item():4f}, ratio {(all_grad_J[-1] / all_grad_J[0]).item():4f}')

    print(f'Ending step {ending_step}, ending J {J.item():4f}, best step {best_J_step},',
          f'best J {best_J.item():4f}, best grad J ratio {(best_grad_J / all_grad_J[0]).item():4f}')

    # This kind of output may be a bit clumsy, however, it has to be done this way in case of parallelization,
    # so we decided to do it the same way here for the sake of universality.
    return {'out_latent': best_latent_vec, 'best_J': best_J, 'all_J': all_J,
            'all_Jo': all_Jo, 'all_Jb': all_Jb, 'all_grad_J': all_grad_J}

if args.VarDA_type == 'prec_3D-Var':
    algorithm_output = latent3DVar_algorithm_preconditioned()
else:
    raise AttributeError(f'{args.VarDA_type} not yet added to the system')

decoded_ana_dest = standardise_destandardise(
    AE_model.decode(algorithm_output['out_latent']),
    AE_props,
    AE_scalers_mean,
    AE_scalers_std,
    action='destandardise',
    device=device
)


ana_inc = decoded_ana_dest - decoded_bg_dest
ana_inc_obs_loc = interpolate_irregular_grid(grid_lats, grid_lons, ana_inc[:, obs_qty_idx], obs_lats, obs_lons)

print('Ana. inc. at obs. loc.', ana_inc_obs_loc)

grid_lats = grid_lats.cpu()
grid_lons = grid_lons.cpu()
ana_inc = ana_inc.detach().cpu()

if args.custom_addon == 'o3np':
    zerolon = np.nonzero(grid_lons.numpy() == 0.)
    #print(ana_inc.shape)
    #print(zerolon)
    #print(ana_inc[zerolon].shape)
    np.savez(f'experiments/data/o3np/ana_inc_and_bg_at_0W_{args.obs_datetime}_{args.singobs_lat}N', 
             ana_inc=ana_inc[zerolon].numpy(), 
             bg=decoded_bg_dest[zerolon].detach().cpu().numpy())
    import time
    #time.sleep(50)
    raise AssertionError

elif args.custom_addon[:5] == 'ekman':
    idx = int(args.custom_addon[5:])
    ana_inc_at_obs_loc = ana_inc[idx]
    print(np.shape(ana_inc))
    print(ana_inc_at_obs_loc.shape)
    np.savez(f'experiments/data/ekman/obs_{args.obs_qty}_ana_inc_and_bg_at_{args.singobs_lat}_{args.singobs_lon}_{idx}_{args.obs_datetime}', 
             ana_inc=ana_inc_at_obs_loc.numpy(), 
             bg=decoded_bg_dest[idx].detach().cpu().numpy(),
             metadata=(args.singobs_lat, args.singobs_lon, idx))
    raise AssertionError

import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# from matplotlib.tri import Triangulation
# tri = Triangulation(grid_lons_plot, grid_lats)
# if args.singobs_lat is not np.nan:
#     # single obs experiment - orthographic projection
#     projection = ccrs.NearsidePerspective(
#                        central_longitude=args.singobs_lon,
#                        central_latitude=args.singobs_lat,
#                        satellite_height=4500000)
#     fig = plt.figure(figsize=(6, 6 * ana_inc.shape[1]))
# else:
#     projection = ccrs.PlateCarree()
#     fig = plt.figure(figsize=(12, 6 * ana_inc.shape[1]))
import cartopy.crs as ccrs
plt.figure(figsize=(6,5.5 * ana_inc.shape[1]))
if args.singobs_lon is not np.nan:
    proj = ccrs.NearsidePerspective(central_longitude=args.singobs_lon, central_latitude=args.singobs_lat, satellite_height=10000000)
    # proj = ccrs.Orthographic(central_longitude=args.singobs_lon, central_latitude=args.singobs_lat)
    # proj = ccrs.Robinson()
    print('Set NearsidePerspective projection')
else:
    proj = ccrs.Robinson()

for i in range(ana_inc.shape[1]): #range(3):#
    # ax = fig.add_subplot(ana_inc.shape[1], 1, i+1, projection=projection)
    # minmax = max(torch.amax(ana_inc[:, i]), abs(torch.amin(ana_inc[:, i])))
    # tcf = ax.tricontourf(tri, ana_inc[:, i], cmap='bwr', vmin=-minmax, vmax=minmax, levels=np.linspace(-minmax, minmax, 10), transform=ccrs.PlateCarree())
    # fig.colorbar(tcf, ax, label='unit')
    # ax.set_xlabel('Longitude')
    # ax.set_ylabel('Latitude')
    # ax.grid()
    # ax.set_title(f'{AE_props.reconstructed_variables[i]} increment')
    ax = plt.subplot(ana_inc.shape[1], 1, i+1, projection=proj)
    minmax = max(torch.amax(ana_inc[:, i]), abs(torch.amin(ana_inc[:, i])))

    xy = proj.transform_points(ccrs.PlateCarree(), grid_lons_plot, grid_lats)
    x = xy[:, 0]
    y = xy[:, 1]

    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]
    vals = ana_inc[:, i].detach().cpu().numpy()
    vals_valid = vals[mask]
    # print(np.nanmin(vals_valid), np.nanmax(vals_valid))

    from matplotlib.tri import Triangulation
    import matplotlib.ticker as mticker
    # triang = Triangulation(grid_lons_plot, grid_lats)
    triang = Triangulation(x_valid, y_valid)
    #tcf = ax.tricontourf(triang, ana_inc[:, i], cmap='bwr', vmin=-minmax, vmax=minmax, levels=np.linspace(-minmax, minmax, 10), transform=ccrs.PlateCarree())
    tcf = ax.tricontourf(triang, vals_valid, cmap='bwr', vmin=-minmax, vmax=minmax, levels=np.linspace(-minmax, minmax, 10))

    #tcf = plt.tricontourf(grid_lons_plot, grid_lats, ana_inc[:, i], cmap='bwr', vmin=-minmax, vmax=minmax, levels=np.linspace(-minmax, minmax, 10))
    
    # plt.scatter(grid_lons_plot, grid_lats, s=0.2, c='k')
    ax.scatter(obs_lons.cpu(), obs_lats.cpu(), c='gold', zorder=1000, transform=ccrs.PlateCarree())
    plt.colorbar(tcf, label='unit')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    gl = ax.gridlines()
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, step=30))
    gl.ylocator = mticker.FixedLocator(np.arange(-80, 80+1, step=20))
    ax.coastlines()
    ax.set_title(f'{AE_props.reconstructed_variables[i]} increment')



if args.singobs_lat is not np.nan:
    plt.savefig(f"{FIGS}/increments_{args.AE_version}_{args.VarDA_type}_singobs_{args.singobs_lat}_{args.singobs_lon}_{args.custom_addon}_{args.obs_datetime}_obs_{args.obs_qty}_obs_inc_{args.obs_inc}_obs_std_{args.obs_std}_ilr_{args.init_lr}.png", dpi=100)

else:
    plt.savefig(f"{FIGS}/increments_{args.AE_version}_{args.VarDA_type}_multiobs_{args.custom_addon}_{args.obs_datetime}_obs_{args.obs_qty}_obs_inc_{args.obs_inc}_obs_std_{args.obs_std}_ilr_{args.init_lr}.png", dpi=100)
