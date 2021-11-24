import ujson as json
import torch
import os
import os.path
from datajuicer import cachable, get, format_template
import argparse
import random
import re
from Utils.utils import device

def standard_defaults():
    return {
        "seed":0,
        "batch_size":128,
        "weight_decay":5e-4,
        "momentum":0.9,
        "save_every":10,
        "lr":0.1,
        }

def help():
    return {
        "l2_weight_decay":"L2 weight decay term.",
        "seed":"Seed of experiments.",
        "batch_size":"Batch size of training.",
        "clipping_alpha":"Number of standard deviations used for clipping weights."
    }

launch_settings = {
    "direct":"mkdir -p Resources/Logs; python3 {code_file} {args} 2>&1 | tee Resources/Logs/{session_id}.log",
    "bsub":"module load CUDA/10.2 nvidia-cudnn/7.6.5 nvidia-nccl/2.5.6 ; mkdir -p Resources/Logs; bsub -o Resources/Logs/{session_id}.log -R \"rusage[ngpus_excl_p=1]\" -q prod.med \"python3 {code_file} {args}\""
}


def _dict_to_bash(dic):
    def _format(key, value):
        if type(value) is bool:
            if value==True:
                return f"-{key}"
            else:
                return ""
        else: return f"-{key}={value}"
    
    return  " ".join([_format(key, val) for key, val in dic.items()])


def mk_runner(architecture, env_vars):
    @cachable(
        dependencies=["model:"+key for key in architecture.default_hyperparameters().keys()], 
        saver = None,
        loader = architecture.loader,
        checker=architecture.checker,
        table_name=architecture.__name__
    )
    def runner(model):
        try:
            mode = get(model, "mode")
        except:
            mode = "direct"
        model["mode"] = mode
        def _format(key, value):
            if type(value) is bool:
                if value==True:
                    return f"-{key}"
                else:
                    return ""
            else: return f"-{key}={value}"

        model["args"] = _dict_to_bash({key:get(model,key) for key in list(architecture.default_hyperparameters().keys())+env_vars + ["session_id"]})
        
        command = format_template(model,launch_settings[mode])
        print(command)
        os.system(command)
        return None

    return runner

def _get_flags(default_dict, help_dict, arg_dict=None):
    parser = argparse.ArgumentParser()
    for key, value in default_dict.items():
        if type(value) is bool:
            parser.add_argument("-"+key, action="store_true",help=help_dict.get(key,""))
        else:
            parser.add_argument("-" + key,type=type(value),default=value,help=help_dict.get(key,""))
    parser.add_argument("-session_id", type=int, default = 0)
    if arg_dict is None:
        flags = parser.parse_args()
    else:
        dic = {key: get(arg_dict,key) for key in arg_dict if key in default_dict or key=="session_id"}
        string = _dict_to_bash(dic)
        flags = parser.parse_args(str.split(string))
    if flags.session_id==0:
        flags.session_id = random.randint(1000000000, 9999999999)
    
    return flags

def log(session_id, key, value, save_dir = None):
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Resources/TrainingResults/")
    file = os.path.join(save_dir, str(session_id) + ".json")
    exists = os.path.isfile(file)
    directory = os.path.dirname(file)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    if exists:
        data = open(file).read()
        d = json.loads(data)
    else:
        d = {}
    with open(file,'w+') as f:
        if key in d:
            d[key] += [value]
        else:
            d[key]=[value]
        out = re.sub('(?<!")Inf(?!")','"Inf"', json.dumps(d))
        out = re.sub('(?<!")NaN(?!")','"NaN"', out)
        f.write(out)

class Resnet:
    @staticmethod 
    def default_hyperparameters():
        d = standard_defaults()
        d["n_attack_steps"]=10
        d["beta_robustness"]=0.0
        d["gamma"]=0.0
        d["eps_pga"]=0.0
        d["eta_train"]=0.0
        d["inner_lr"] = 0.1,
        d["eta_mode"]="ind"
        d["clipping_alpha"]=0.0
        d["attack_size_mismatch"]=0.2
        d["initial_std"]=0.001
        d["pretrained"]=True
        d["burn_in"]=0
        d["workers"]=4
        d["n_epochs"] = 300
        d["dataset"] = "cifar10"
        d["architecture"] = "resnet32"
        d["start_epoch"] = 0
        return d

    @staticmethod
    def make():
        d = Resnet.default_hyperparameters()
        def mk_data_dir(mode="direct"):
            if mode=="direct":
                return "~/Datasets"
            elif mode=="bsub":
                return "/dataP/jbu"
            raise Exception("Invalid Mode")
        d["mk_data_dir"] = mk_data_dir
        d["data_dir"] = "{mk_data_dir({mode})}"
        d["code_file"] = "trainer_resnet.py"
        d["train"] = mk_runner(Resnet, ["data_dir"])
        return d

    @staticmethod
    def get_flags(dic=None):
        default_dict = {**Resnet.default_hyperparameters(), **{"data_dir":"~/Datasets"}}
        FLAGS = _get_flags(default_dict, help(), dic)
        return FLAGS

    @staticmethod
    def checker(sid, table, cache_dir):
        try:
            data = Resnet.loader(sid, table, cache_dir)
        except Exception as er:
            print("error", er)
            return False
        
        if "completed" in data and data["completed"]:
            return True
        else:
            return False

    @staticmethod
    def loader(sid, table, cache_dir):
        data = json.load(open(os.path.join("Resources/TrainingResults",f"{sid}.json"),'r'))
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, f"Resources/Models/{sid}_model.th")
        checkpoint = torch.load(model_path, map_location=device)
        data["checkpoint"] = checkpoint
        data["cnn_session_id"] = sid
        return data
