import yaml

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return repr(self.__dict__)

def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    return Config(yaml_data)

# Example usage
if __name__ == "__main__":
    yaml_path = "config.yaml"  # Path to your YAML file
    cfg = load_config(yaml_path)