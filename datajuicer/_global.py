import multiprocessing
import threading
import datajuicer.resource_lock
import datajuicer.local_cache
import os
from datetime import datetime
import sys
import shutil

class GLOBAL:
    resource_lock = None
    cache = None
    task_versions = {}
    task_caches = {}
    task_funcs = {}


def setup(cache=None, max_workers=1, resource_directory="dj_resources/", clean=False):
    #multiprocessing.set_start_method('spawn')
    if os.path.exists(resource_directory):
        shutil.rmtree(resource_directory)
    curthread = threading.current_thread()
    assert(type(curthread) != datajuicer.task.Run)
    curthread.unique_id = datajuicer.utils.rand_id()
    #datajuicer.logging.enable_proxy()
    GLOBAL.resource_lock = datajuicer.resource_lock.ResourceLock(resource_directory, True)
    for i in range(max_workers-1):
        GLOBAL.resource_lock.release()
    # if max_workers > 1:
    #     GLOBAL.resource_lock.release(max_workers-1)
    if cache is None:
        cache = datajuicer.local_cache.LocalCache()
    GLOBAL.cache = cache
    if clean:
        GLOBAL.cache.clean()

# def subprocess_setup(cache, resource_directory):
#     curthread = threading.current_thread()
#     assert(type(curthread) != datajuicer.task.Run)
#     curthread.unique_id = datajuicer.utils.rand_id()
#     datajuicer.logging.enable_proxy()
#     GLOBAL.resource_lock = datajuicer.resource_lock.ResourceLock(resource_directory, False)
#     GLOBAL.cache = cache



def run_id():
    cur_thread = threading.currentThread()
    if hasattr(cur_thread, "run_id"):
        return threading.currentThread().run_id
    return "main"

def reserve_resources(**resources):
    cur_thread = threading.currentThread()
    if hasattr(cur_thread, "resource_lock"):
        threading.currentThread().resource_lock.reserve_resources(**resources)
    else:
        GLOBAL.resource_lock.reserve_resources(**resources)

def free_resources(**resources):
    cur_thread = threading.currentThread()
    if hasattr(cur_thread, "resource_lock"):
        threading.currentThread().resource_lock.free_resources(**resources)
    else:
        GLOBAL.resource_lock.free_resources(**resources)

def _open(path, mode):
    cur_thread = threading.currentThread()
    if hasattr(cur_thread, "_open"):
        return cur_thread._open(path, mode)
    if hasattr(cur_thread, "task"):
        return cur_thread.task.cache.open(cur_thread.task.name, cur_thread.task.version, cur_thread.run_id, path, mode)
    return open(path, mode)

def backup():
    now = datetime.now()
    if os.path.exists("dj_backups/"):
        shutil.rmtree("dj_backups/")
    datajuicer.cache.make_dir("dj_backups/")
    GLOBAL.cache.save(os.path.join("dj_backups", now.strftime("%Y-%m-%d-%H-%M-%S.backup")))

def sync_backups():
    datajuicer.cache.make_dir("dj_backups/")
    for filename in os.listdir("dj_backups"):
        GLOBAL.cache.update(os.path.join("dj_backups", filename))