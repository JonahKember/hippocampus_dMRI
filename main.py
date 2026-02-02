import json
import preprocessing, fitting, surface_mapping

with open('config.json') as f:
    config = json.load(f)

preprocessing.create_B0_surface(config)
preprocessing.create_mask(config)
preprocessing.create_surface_normals(config)
preprocessing.create_surface_to_sphere_angles(config)

fitting.fit_DKT_params(config)
fitting.fit_DKT_metrics(config)
fitting.fit_directional_kurtosis(config)
fitting.fit_directional_diffusion(config)

surface_mapping.create_partial_volume_mask(config)
surface_mapping.run_surface_mapping(config)
