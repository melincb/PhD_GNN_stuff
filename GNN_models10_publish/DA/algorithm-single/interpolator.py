def normalize_longitudes(lons):
    # 0 - 360 -> -180 - +180
    # Use for: (1) plotting, (2) interpolation (observation longitudes are given as -180 - +180)
    return ((lons + 180) % 360) - 180

def interpolate_irregular_grid(grid_lats, grid_lons, grid_values, target_lats, target_lons, k=4):
    import torch
    # Normalize longitudes
    grid_lons = normalize_longitudes(grid_lons) # grid lons are initially 0 to 358.875
    # target_lons = normalize_longitudes(target_lons)   # Target lons are originally already -180 to 180

    # Convert to Cartesian
    def latlon_to_cartesian(lat, lon):
        lat_rad = torch.deg2rad(lat)
        lon_rad = torch.deg2rad(lon)
        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = torch.cos(lat_rad) * torch.sin(lon_rad)
        z = torch.sin(lat_rad)
        return torch.stack([x, y, z], dim=-1)  # shape: (..., 3)

    grid_xyz = latlon_to_cartesian(grid_lats, grid_lons)  # (M, 3)
    target_xyz = latlon_to_cartesian(target_lats, target_lons)  # (N, 3)

    def interpolate_knn(grid_xyz, grid_values, target_xyz, k=4, eps=1e-6):
        if grid_values.shape[0] == grid_xyz.shape[0]:
            grid_values = grid_values.T

        # grid_values: (F, M), grid_xyz: (M, 3), target_xyz: (T, 3)
        T = target_xyz.shape[0]
        F, M = grid_values.shape

        dists = torch.cdist(target_xyz, grid_xyz)  # (T, M)
        knn_dists, knn_indices = torch.topk(dists, k, dim=1, largest=False)  # (T, k)

        # Expand knn_indices for gather
        knn_indices_exp = knn_indices.unsqueeze(0).expand(F, -1, -1)  # (F, T, k)
        knn_values = torch.gather(grid_values.unsqueeze(1).expand(-1, T, -1), 2, knn_indices_exp)  # (F, T, k)

        weights = 1.0 / (knn_dists + eps)  # (T, k)
        weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize
        weights = weights.unsqueeze(0)  # (1, T, k)

        interpolated = (weights * knn_values).sum(dim=2)  # (F, T)

        return interpolated


    return interpolate_knn(grid_xyz, grid_values, target_xyz, k)

# import numpy as np
# import torch
# from pathlib import Path
# BASE_PATH = Path("/shared/mari/zaplotnikz/autoencoder")
# DATA = BASE_PATH / "data"
# grid_lats = torch.from_numpy(np.load(DATA / "grid_lats.npy"))
# grid_lons = torch.from_numpy(np.load(DATA / "grid_lons.npy"))
# grid_lons_plot = normalize_longitudes(grid_lons)
#
# grid_values1 = grid_lons / 360
# grid_values2 = grid_lats / 90
#
# grid_values = torch.stack((grid_values1, grid_values2))
#
# target_lons = torch.from_numpy(np.array([-1.5, 110.1, 110.1, -50.]))
# target_lats = torch.from_numpy(np.array([10.9, -12.1, 12.1, 89.9]))
#
# result = interpolate_irregular_grid(grid_lats, grid_lons, grid_values, target_lats, target_lons)
# print(result)
# print(max(grid_values2))
#
# import matplotlib.pyplot as plt
# plt.figure(1,figsize=(12, 6))
# tcf = plt.tricontourf(grid_lons_plot.numpy(), grid_lats.numpy(), grid_values1.numpy(), levels=np.arange(0.,1.01,0.10), cmap='coolwarm')
# plt.scatter(grid_lons_plot.numpy(), grid_lats.numpy(), s=0.2, c='k')
# plt.scatter(target_lons, target_lats, c='gold')
# plt.colorbar(tcf, label='m/s)')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Test')
# plt.savefig(f"interpolator_test1.png",dpi=300)
# plt.figure(2,figsize=(12, 6))
# tcf = plt.tricontourf(grid_lons_plot.numpy(), grid_lats.numpy(), grid_values2.numpy(), levels=np.arange(-1.,1.01,0.10), cmap='coolwarm')
# plt.scatter(grid_lons_plot.numpy(), grid_lats.numpy(), s=0.2, c='k')
# plt.scatter(target_lons, target_lats, c='gold')
# plt.colorbar(tcf, label='m/s)')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Test')
# plt.savefig(f"interpolator_test2.png",dpi=300)