import yaml

def load_config(config_path, config_name:str):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config.get(config_name, config)