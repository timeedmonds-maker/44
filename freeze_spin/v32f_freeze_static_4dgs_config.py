# V32f freeze-only static-in-time configuration for hustvl/4DGaussians.
# One synchronized timestamp only. Spatial Gaussian optimisation remains active;
# temporal regularisers are disabled because there is no motion to learn.

ModelHiddenParams = dict(
    kplanes_config={
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 32,
        'resolution': [64, 64, 64, 1],
    },
    multires=[1, 2, 4],
    defor_depth=1,
    net_width=128,
    plane_tv_weight=0.00015,
    time_smoothness_weight=0.0,
    l1_time_planes=0.0,
    no_dx=False,
    no_grid=False,
    no_ds=False,
    no_dr=False,
    no_do=False,
    no_dshs=False,
    empty_voxel=False,
    static_mlp=False,
)

OptimizationParams = dict(
    dataloader=True,
    iterations=8000,
    coarse_iterations=1000,
    batch_size=1,
    lambda_dssim=0.20,
    lambda_lpips=0.01,
    position_lr_max_steps=7000,
    densify_from_iter=250,
    densify_until_iter=6000,
    densification_interval=100,
    pruning_from_iter=400,
    pruning_interval=100,
    opacity_reset_interval=2500,
    opacity_threshold_coarse=0.005,
    opacity_threshold_fine_init=0.005,
    opacity_threshold_fine_after=0.005,
    densify_grad_threshold_coarse=0.0002,
    densify_grad_threshold_fine_init=0.0002,
    densify_grad_threshold_after=0.00015,
)
