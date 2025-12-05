import yaml

def update_yaml_config(model_name, best_params, cfg_path=None):
    if cfg_path is None:
        if model_name == 'logistic':
            cfg_path = 'cfg/logreg.yml'
        elif model_name == 'xgboost':
            cfg_path = 'cfg/xgboost.yml'
        elif model_name == 'random_forest':
            cfg_path = 'cfg/random_forest.yml'
        else:
            cfg_path = 'cfg/{model_name}.yml'

    # Load existing YAML
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Overwrite old params with tuned ones
    cfg.update(best_params)

    # Write back to YAML
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"💾 Saved best parameters to {cfg_path}")