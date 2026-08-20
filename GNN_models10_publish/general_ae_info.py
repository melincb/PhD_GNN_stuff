class ae_props:
    def __init__(self, version):
        import sys
        from pathlib import Path
        self.BASE_PATH = Path('') # INSERT Path("")
        # some important stuff from supporting_stuff should be placed in BASE_PATH
        # all manually set paths in other files (e.g. to grid_lats.npy and grid_lons.npy) should be modified according to new setup!
        sys.path.append(str(self.BASE_PATH))
        
        self.nt = 731   # only years 2023-2024
        self.nnode = 35718

        self.static_variables = ["lats", "lons", "lnsurfgeo", "lsm"] # normalized logarithm of surface geopotential
        # Atmospheric variables
        self.levels = [5,10,20,30,50,70,100,150,200,300,400,500,600,700,800,850,925,1000] # pressure levels in hPa
        self.variables = ["u", "v", "z", "t", "q","o3"] # zonal wind, meridional wind, geopotential, temperature, specific humidity
        combinations = [f"{var}{level}" for var in self.variables for level in self.levels]

        # Surface variables
        surface_variables = ["t2m", "msl", "sd", "siconc2","sst2","stl1"]

        self.time_varying_variables = combinations + surface_variables
        self.reconstructed_variables = combinations + surface_variables
        
        self.IN_DIM = len(self.time_varying_variables+self.static_variables)
        self.OUT_DIM = len(self.reconstructed_variables)
        
        self.n_train = 0# int(0.8 * self.nt)
        self.n_val = 0 #int(0.13 * self.nt)
        self.n_test = self.nt - self.n_train - self.n_val
        self.data_dt = 24  # timestepping in data (in hours)

        self.EXP_PREFIX = version
        self.EXP_ID = version
        self.vPATH = '' #INSERT PATH STRING
        self.DATA = self.BASE_PATH / "data"
        self.CHECKPOINTS = self.BASE_PATH / "checkpoints"
        self.WEIGHTS = self.BASE_PATH / "weights"
        self.GRAPH = self.BASE_PATH / "graph"
        self.best_model_pth = self.WEIGHTS / f"{self.EXP_PREFIX}_best_model.pt"

        self.scalers_mean_pth = self.DATA / "scalers_mean.npz"
        self.scalers_std_pth = self.DATA / "scalers_std.npz"

        if version == 'v63':
            import models10
            self.modelslib = models10
            
            self.latent_dim = 128
            self.LATENT_DIM = self.latent_dim

            self.hidden_dims = [312, 256, 192, 160, 144] # encoder order
            self.HIDDEN_DIMS = self.hidden_dims
            

            self.K_FINE = 12
            self.K_COARSE = 12 
            self.RES_RATIO = 4
            self.N_SUBGRAPHS = 6 
            self.use_pooling = True

            self.HEADS = 8
            self.GAT_DROPOUT = 0.1
            self.FEATURE_DROPOUT = 0.2
            self.LATENT_DROPOUT = 0.05
            self.LATENT_NOISE_STD = 0.03
            self.pooling_schedule = [1,3]
            self.unpooling_schedule = [2,4]

            nlatnode = self.nnode
            for i in range(len(self.pooling_schedule)):
                nlatnode = nlatnode // 4
            self.nlatnode = nlatnode         
            self.use_residuals = True
            self.use_residualsIO = True
            self.level_connections_type = 'level_connections_local'

        else:
            raise AttributeError(f'Version not recognised: {version}')
        
def date_to_dataidx(datestr):
    from datetime import datetime, timedelta
    if type(datestr) == str:
        if len(datestr) == len('2020-01-01-00'):
            # Remove hours - they are redundant for current dataset
            datestr = datestr[:-3]
        interest_datetime = datetime.strptime(datestr, '%Y-%m-%d')
    else:
        interest_datetime = datestr
    init_data_datetime = datetime(2023, 1, 1)   # data only for 2023-2024
    delta_days = (interest_datetime - init_data_datetime).days

    return delta_days


def standardise_destandardise(
        input_fields,
        AE_props,
        scalers_mean,   # Loading them every single time is much slower than passing them from main file
        scalers_std,
        action='standardise',
        device=None
):
    import torch    # Don't worry about importing torch multiple times -
                    # the first import indeed takes O(1s),
                    # but all subsequent imports take O(1e-6s)

    # all_mean = torch.tensor([torch.from_numpy(scalers_mean[rv]) for rv in AE_props.reconstructed_variables])
    # all_std = torch.tensor([torch.from_numpy(scalers_std[rv]) for rv in AE_props.reconstructed_variables])

    # print(type(scalers_mean))
    if type(scalers_mean) != torch.Tensor:
        all_mean = torch.tensor([torch.from_numpy(scalers_mean[rv]) for rv in AE_props.reconstructed_variables])
        all_std = torch.tensor([torch.from_numpy(scalers_std[rv]) for rv in AE_props.reconstructed_variables])
    else:
        all_mean = scalers_mean
        all_std = scalers_std

    if device:
        all_mean, all_std = all_mean.to(device), all_std.to(device)

    if action == 'standardise':
        return (input_fields - all_mean) / all_std
    elif action == 'destandardise':
        return input_fields * all_std + all_mean
    elif action == 'destandardise variance':
        return input_fields * all_std * all_std   # For Welford algorithm output
    else:
        raise ValueError(f'Unknown action: {action}')

