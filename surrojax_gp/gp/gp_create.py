import json
import jax.numpy as jnp
import inspect
import types
import re

from .gp_class import GP_Surrogate

FUN_NAME_REGEX = re.compile("def\s+(.+)\s*\(")
KERNEL_FUN_NAME = 'kernel'

def create_gp(kernel_func, x_train, y_train, constraints):
    # Pre-process x_train and y_train shapes:
    x_train, y_train = preprocess_x_and_y(x_train, y_train)
    # Create dictionary which stores all user-provided information:
    create_dict = {"kernel": kernel_func,
                   "x_train": x_train,
                   "y_train": y_train,
                   "constraints": constraints}
    # Create Gaussian process:
    GP = GP_Surrogate(create_dict)
    return GP

def preprocess_x_and_y(x_train, y_train):
    x_train = jnp.atleast_2d(x_train.squeeze())
    y_train = jnp.atleast_1d(y_train.squeeze())
    assert y_train.ndim==1 and x_train.ndim==2
    if x_train.shape[0] != y_train.size:
        x_train = x_train.T
    assert x_train.shape[0] == y_train.size
    return (x_train, y_train)

def load_gp(json_dir):
    # Attempt to load JSON file:
    try:
        with open(json_dir, "r") as json_file:
            loaded_json = json.load(json_file)
    except:
        print(f"Unable to load the file {json_dir}")
    # Convert relevant attributes to jax.numpy arrays:
    loaded_json = json_2_jnp(loaded_json)
    # Load kernel function specified in JSON file:
    local_dict = {}
    exec(loaded_json["kernel_fun"], globals(), local_dict)
    loaded_json["kernel"] = local_dict["kernel"]
    # Create Gaussian process from loaded data:
    GP = GP_Surrogate(loaded_json)
    return GP

# NB: Modules other than jax.numpy need to be imported INSIDE function:
def save_gp(GP_obj, save_dir, save_params=True, save_L_and_alpha=True):
    # Place information to be saved into a dictionary:
    fun_lines = inspect.getsourcelines(GP_obj.kernel)[0]
    # Replace function name with 'kernel' in first line - prevents errors during loading:
    fun_lines[0] = re.sub(FUN_NAME_REGEX, fun_lines[0], f"def {KERNEL_FUN_NAME}(", count=1)
    kernel_fun_str = ''.join(fun_lines)
    vals_2_save = {"kernel_fun": kernel_fun_str,
                   "x_train": GP_obj.x_train,
                   "y_train": GP_obj.y_train,
                   "constraints": GP_obj.constraints}
    if save_params:
        vals_2_save["params"] = GP_obj.params 
    if save_L_and_alpha:
        vals_2_save["L"] = GP_obj.L
        vals_2_save["alpha"] = GP_obj.alpha
    # Convert relevant values to JSON-saveable format:
    vals_2_save = jnp_2_json(vals_2_save)
    # Save dictionary as JSON file:
    if save_dir[-5:] != ".json":
        save_dir += ".json"
    with open(save_dir, 'w') as f:
        json.dump(vals_2_save, f, indent=4)

# Converts relevant GP attributes from Jax.numpy arrays (which cannot be saved into JSON files)
# to lists (which can be saved into JSON files):
def jnp_2_json(save_dict):
    for key in (set(save_dict.keys()) & {"x_train", "y_train", "L", "alpha"}):
        save_dict[key] = save_dict[key].tolist()
    if "params" in save_dict:
        save_dict["params"] = {key: value.tolist() for (key, value) in save_dict["params"].items()}
    return save_dict

# Converts relevant attributes into Jax.numpy arrays:
def json_2_jnp(json_dict):
    # Convert al
    for key in (set(json_dict.keys()) & {"x_train", "y_train", "L", "alpha"}):
        json_dict[key] = jnp.array(json_dict[key])
    if "params" in json_dict:
        json_dict["params"] = {key: jnp.array(value) for (key, value) in json_dict["params"].items()}
    return json_dict