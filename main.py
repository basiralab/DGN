import os
from model import DGN
from config import MODEL_PARAMS, CONFIG

if not os.path.exists('temp'):
    os.makedirs('temp')
if not os.path.exists('output'):
    os.makedirs('output')  


DGN.train_model(
                CONFIG["X"],
                model_params=MODEL_PARAMS,
                n_max_epochs=CONFIG["N_max_epochs"],
                n_folds=CONFIG["n_folds"],
                random_sample_size=CONFIG["random_sample_size"],
                early_stop=CONFIG["early_stop"],
                model_name=CONFIG["model_name"])