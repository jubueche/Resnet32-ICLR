DEBUG = True
if __name__ == "__main__" and DEBUG:
    import sys
    import os
    sys.path.append(os.getcwd())

from datajuicer._global import GLOBAL
from datajuicer.resource_lock import ResourceLock
import dill
import os
import subprocess
import time
import argparse
import threading
from datajuicer.task import Run
import datajuicer
from contextlib import redirect_stdout
import sys
import pathlib
import importlib


def launch(context):
    path1 = os.path.join(context["resource_lock_directory"], f"{context['unique_id']}_context.dill")
    with open(path1, "wb+") as f:
        dill.dump(context, f)
    # path2 = os.path.join(context["resource_lock_directory"], f"{context['unique_id']}_func.dill")
    # with open(path2, "wb+") as f:
    #     dill.dump(func, f, recurse=True)
    if DEBUG:
        command = f"python {__file__} -path {path1}"
    else:
        command = f"djlaunch -path {path1}"

    if context["mode"] == "thread":
        _launch(path1)
    if context["mode"] == "process":
        subprocess.run(command.split())
    if context["mode"] == "bsub":
        subprocess.run(["bsub", f'"{command}"', *context["mode_args"]])
        while(True):
            if context["cache"].is_done(context["task_name"], context["task_version"], context["run_id"]):
                break
            time.sleep(0.5)

class Logger:
 
    def __init__(self, file, mute = False):
        self.console = sys.stdout
        self.file = file
        self.mute= mute
 
    def write(self, message):
        if not self.mute:
            self.console.write(message)
        self.file.write(message)
 
    def flush(self):
        self.console.flush()
        self.file.flush()

class Namespace:
    pass

def _launch(path):
    with open(path, "rb") as f:
        context = dill.load(f)
    if context["mode"] in ["process", "bsub"]:
        
        #enable_proxy()
        curthread = threading.current_thread()
        #curthread.dj = True
        GLOBAL.resource_lock = ResourceLock(directory = context["global_resource_lock_directory"], init=False, session=context["global_resource_lock_session"]) #init=False
        GLOBAL.cache = context["global_cache"]
        pseudo_parent = Namespace()
        if not context["parent_task_name"] is None:
            pseudo_parent.task = Namespace()
            pseudo_parent.task.name = context["parent_task_name"]
            pseudo_parent.task.version = context["parent_task_version"]
            pseudo_parent.run_id = context["parent_run_id"]
            pseudo_parent.incognito = context["parent_incognito"]
        
        curthread.parent = pseudo_parent

        pseudo_task = Namespace()
        pseudo_task.name = context["task_name"]
        pseudo_task.version = context["task_version"]
        pseudo_task.cache = context["cache"]
        curthread.run_id = context["run_id"]
        curthread.task = pseudo_task
        curthread.kwargs = context["kwargs"]
        curthread.unique_id = datajuicer.utils.rand_id()
        curthread.resource_lock = ResourceLock(directory = context["resource_lock_directory"], init=False, session=context["resource_lock_session"]) #init=False
        curthread.incognito = context["incognito"]

        sys.path.append(os.path.dirname(context["file"]))
        module = importlib.import_module(pathlib.Path(context["file"]).stem)
        #print(datajuicer.GLOBAL.task_funcs)
    
    func = datajuicer.GLOBAL.task_funcs[context["task_name"]]
    # print(module.__dict__)
    # globals().update(module.__dict__)

    if not context["incognito"]:
        #redirect(context["cache"].open(context["task_name"], context["task_version"], context["run_id"], "log.txt", "w+"))
        #context["cache"].record_run(context["task_name"], context["task_version"], context["run_id"], context["kwargs"])
        logger = Logger(context["cache"].open(context["task_name"], context["task_version"], context["run_id"], "log.txt", "w+"))
        with redirect_stdout(logger):
            result = func(**context["kwargs"])#eval('func(**context["kwargs"])', module.__dict__, locals())
    else:
        result =  func(**context["kwargs"])#eval('func(**context["kwargs"])', module.__dict__, locals())
    if not context["incognito"]:
        #stop_redirect()
        context["cache"].record_result(context["task_name"], context["task_version"], context["run_id"], result)
    else:
        path = os.path.join(context["resource_lock_directory"], f"{context['unique_id']}_result.dill")
        with open(path, "wdjlaunch -path {path1}b+") as f:
            dill.dump(context, f)


def djlaunch():
    ap = argparse.ArgumentParser()
    ap.add_argument("-path", type=str)
    args = ap.parse_args()
    _launch(args.path)
    
if __name__ == "__main__" and DEBUG:
    djlaunch()