import argparse
import pickle
import numpy as np

import matplotlib as mpl

# mpl.rcParams['pdf.fonttype'] = 42   # TrueType
# mpl.rcParams['ps.fonttype'] = 42
#mpl.rcParams['pdf.fonttype'] = 3
mpl.rcParams["text.usetex"] = False

import torch


parser = argparse.ArgumentParser()
parser.add_argument('--AE_version', help="Version of AE, e.g. 'v5'", type=str, required=True)
parser.add_argument("--ERA5_datetime", help="Date and hour of ERA5 input in format yyyy-mm-dd-hh or yyyymmddhh", type=str, required=False, default='2023-04-15-00')
parser.add_argument('--savefig_dir', help='Directory for saving the figure (if specified, it will be saved to experiments/figures/{in_out_ch}ch/args.savefig_dir; otherwise it will be just saved to experiments/figures/{in_out_ch}ch/)', type=str, default='', required=False)
parser.add_argument("--plot_rec", help='', required=False, default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--fractional_save", help='', required=False, default=False, action=argparse.BooleanOptionalAction)

ERA5_datetime = 0

args = parser.parse_args()
if '-' not in args.ERA5_datetime:
    if len(args.ERA5_datetime) == 10:
        args.ERA5_datetime = f'{args.ERA5_datetime[0:4]}-{args.ERA5_datetime[4:6]}-{args.ERA5_datetime[6:8]}-{args.ERA5_datetime[8:10]}'
    else:
        raise AttributeError('Improper ERA5_datetime format. Should be yyyy-mm-dd-hh or yyyymmddhh.')



deterministic = False#True #False


import sys
# print(sys.path)
sys.path.append('DA/algorithm-single')
from general_ae_info import ae_props, date_to_dataidx, standardise_destandardise
AE_props = ae_props(args.AE_version)
sys.path.append(str(AE_props.BASE_PATH))
from utilities import clean_state_dict
import datasets, importlib
importlib.reload(datasets)



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






# -----------------------------
# Device setup
# -----------------------------
# if args.VarDA_type == 'inc_3D-Var':
#     device = 'cpu'
# else:
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
# import models10
importlib.reload(AE_props.modelslib)

import subgraphs
importlib.reload(subgraphs)

from subgraphs import build_multiscale_graph, build_hierarchical_graph_v4, \
    build_hierarchical_graph_v5, create_pooling_matrices_v2, build_hierarchical_graph_v6

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
# for q in obs_qty:
#     print(q)
#     print(AE_scalers_mean[q])
#     print(AE_scalers_std[q])
#     print()

AE_scalers_mean = torch.tensor([torch.from_numpy(AE_scalers_mean[rv]) for rv in AE_props.reconstructed_variables])
AE_scalers_std = torch.tensor([torch.from_numpy(AE_scalers_std[rv]) for rv in AE_props.reconstructed_variables])



the_datetime = datetime.strptime(args.ERA5_datetime, "%Y-%m-%d-%H")



# ------------------------------------------------------
# Load data
# ------------------------------------------------------
indices_of_interest = range(date_to_dataidx(the_datetime), date_to_dataidx(the_datetime) + 1)
print(indices_of_interest)

import datasets
importlib.reload(datasets)

dataset_of_interest = datasets.ReallyLazyGraphDataset_v2(
    indices_of_interest, AE_props.DATA, \
    AE_props.time_varying_variables, AE_props.static_variables, \
    AE_props.reconstructed_variables)

init_truth = dataset_of_interest[0].to(device)
end_truth = dataset_of_interest[-1].to(device)

physical_bg = init_truth.x

import cartopy.crs as ccrs
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
grid_lats = np.load("../grid_lats.npy")   # -90 to 90
grid_lons = np.load("../grid_lons.npy")   # 0 to 360
grid_lons_plot = normalize_longitudes(grid_lons)    # -180 to 180
if not args.plot_rec:
    p1 = standardise_destandardise(physical_bg[:,:-4], AE_props, AE_scalers_mean, AE_scalers_std, 'destandardise', device).cpu().numpy()
else:
    _, latent_bg = AE_model(physical_bg, return_latent=True)
    p1 = standardise_destandardise(
            input_fields = AE_model.decode(latent_bg),
            AE_props = AE_props,
            scalers_mean = AE_scalers_mean,
            scalers_std = AE_scalers_std,
            action = 'destandardise',
            device = device
        ).cpu().detach().numpy()
#p1 = physical_bg.cpu().numpy()
print('shape p1', p1.shape)
tri = Triangulation(grid_lons_plot, grid_lats)
proj = ccrs.Robinson()
import matplotlib
matplotlib.rcParams.update({"font.size": 20})
tot = 1


order = ['u', 'v', 'z', 't', 'q', 'o3', 'rest']
oo = 0
i1 = 0
if args.fractional_save:
    plt.figure(figsize=(6,5.5 * 18))
else:
    plt.figure(figsize=(6,5.5 * p1.shape[1]))
for i in range(p1.shape[1]):
    print(i)
    if args.fractional_save:
        ax = plt.subplot(18, 1, i1+1, projection=proj)
    else:
        ax = plt.subplot(p1.shape[1], 1, i+1, projection=proj)
    minmax = max(np.amax(p1[:, i]), abs(np.amin(p1[:, i])))

    xy = proj.transform_points(ccrs.PlateCarree(), grid_lons_plot, grid_lats)
    x = xy[:, 0]
    y = xy[:, 1]

    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]
    vals = p1[:, i]#.detach().cpu().numpy()
    vals_valid = vals[mask]
    # print(np.nanmin(vals_valid), np.nanmax(vals_valid))

    from matplotlib.tri import Triangulation
    import matplotlib.ticker as mticker
    # triang = Triangulation(grid_lons_plot, grid_lats)
    triang = Triangulation(x_valid, y_valid)
    span = np.amax(vals_valid) - np.amin(vals_valid)
    print(span)
    if span < 0.1:
        step=span/10
    if span < 2:
        step = None
    elif span < 16:
        step = 1
    elif span < 30:
        step = 2
    elif span < 80:
        step = 5
    elif span < 130:
        step = 10
    elif span < 200:
        step = 20
    elif span < 500:
        step=50
    elif span < 1500:
        step=100
    elif span < 3000:
        step=200
    elif span < 1000:
        step=500
    else:
        step=1000
    
    try:
        sh = 0.5
        if 'siconc' in AE_props.reconstructed_variables[i]:
            levs = np.arange(0,1.01,step=0.05)
            levs[0] = -0.05
            levs[-1] = 1.01
            tcf = ax.tricontourf(triang, vals_valid, cmap='CMRmap_r', levels=levs)
            plt.colorbar(tcf, shrink=sh, ticks=[-0.05, 0.25, 0.5, 0.75, 1.01], format=mticker.FixedFormatter(['0', '25', '50', '75', '100']))
        elif AE_props.reconstructed_variables[i] in ['v1000', 'v700', 'u1000', 'u700']:
            minmax = max(np.amax(vals_valid), -np.amin(vals_valid))
            step = 4
            #levs1 = np.arange(step * np.ceil(-misnmax / step), 0,step=step)
            levs2 = np.arange(step//2, step * np.floor(minmax / step)+step*0.9,step=step)
            levs = np.append(-levs2[::-1], levs2)
            filtered_levels = [l for l in levs if l%10 == 0]
            tcf = ax.tricontourf(triang, vals_valid, cmap='RdGy', levels=levs, extend='both')
            plt.colorbar(tcf, shrink=sh, ticks=filtered_levels)
        elif step:
            levs = np.arange(
                step * np.ceil(np.amin(vals_valid) / step),
                step * np.floor(np.amax(vals_valid) / step)+step/2,
                step
            )
            tcf = ax.tricontourf(triang, vals_valid, cmap='turbo', extend='both', levels=levs)
            if span > 20 and span < 80:
                filtered_levels = [l for l in levs if l%10 == 0]
                plt.colorbar(tcf, shrink=sh, ticks=filtered_levels)

            else:
                plt.colorbar(tcf, shrink=sh)
        else:
            tcf = ax.tricontourf(triang, vals_valid, cmap='turbo', extend='both', levels=8)
            plt.colorbar(tcf, shrink=sh)

    except:
        print('Failed, ', AE_props.reconstructed_variables[i])
    ax.coastlines()
    if AE_props.reconstructed_variables[i][0] in ('u', 'v'):
        unit = r'm/s'
    elif AE_props.reconstructed_variables[i][0] == 'z':
        unit = r'$\mathrm{m}^2/\mathrm{s}^2$'
    elif AE_props.reconstructed_variables[i][0] == 't' or AE_props.reconstructed_variables[i] in ('t2m', 'sst2', 'stl1'):
        unit = r'K'
    elif AE_props.reconstructed_variables[i][0] in ('o', 'q'):
        unit = r'kg/kg'
    elif AE_props.reconstructed_variables[i] == 'msl':
        unit = r'Pa'
    elif AE_props.reconstructed_variables[i] == 'sd':
        unit = r'm w.e.'
    elif AE_props.reconstructed_variables[i] == 'siconc2':
        unit = r'%'
    else:
        print('Unknown unit')
        unit = ''
    if not args.plot_rec:
        ax.set_title(f'True {AE_props.reconstructed_variables[i]} [{unit}]')
    else:
        ax.set_title(f'Reconstructed {AE_props.reconstructed_variables[i]} [{unit}]')

    #break
    i1 += 1
    if args.fractional_save:
        if AE_props.reconstructed_variables[i] in ['u1000', 'v1000', 'z1000', 't1000', 'q1000', 'o31000', 'stl1']:

            if not args.plot_rec:
                plt.savefig(f'figs/{args.AE_version}_truth_{order[oo]}_{args.ERA5_datetime}.pdf')
            else:
                plt.savefig(f'figs/{args.AE_version}_rec_{order[oo]}_{args.ERA5_datetime}.pdf')
            plt.cla()
            plt.clf()
                    
            plt.figure(figsize=(6,5.5 * 18))
            oo += 1
            i1 = 0
if not args.fractional_save:
    if not args.plot_rec:
        plt.savefig(f'figs/{args.AE_version}_truth_{args.ERA5_datetime}.pdf')
    else:
        plt.savefig(f'figs/{args.AE_version}_rec_{args.ERA5_datetime}.pdf')


