
import trainer.loading as loading
import fixed_f_vanilla as vf
import display


import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    path = cfg.path
    setting = cfg.setting
    


# DATA_PATH = r"D:\task\research\codes\MultiSource\wsl\2\multi-source\data_set_2\\"
# MODEL_PATH = r"D:\task\research\codes\MultiSource\wsl\1\weight_set_2\\"
DATA_PATH = "/home/viki/Codes/MultiSource/2/multi-source/data_set_2/"
MODEL_PATH = "/home/viki/Codes/MultiSource/3/multi_source_exp/formula_test/weight/"

my_app()

N_TASK = 21

for id in range(N_TASK):

    data = loading.load_data(DATA_PATH, id)
    model_f, model_g = loading.load_model(MODEL_PATH, id)

    break


