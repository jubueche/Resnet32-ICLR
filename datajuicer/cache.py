import os
import shutil
import zipfile
import dill
import json

import datajuicer

def make_dir(path):
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory)

def copy(path, packet_len = 2048):
    for root, _, files in os.walk(path, topdown = False):
        for name in files:
            file_path = os.path.join(root, name)
            with open(file_path, "rb") as f:
                while True:
                    packet_path = file_path[len(path)+1:]
                    packet_data = f.read(packet_len)
                    yield (packet_path, packet_data)
                    if packet_data == b'':
                        break

def paste(producer, path):
    open_file = open(".scp.tmp", "wb+")
    for sub_path, data in producer:
        full_path = os.path.join(path, sub_path)
        if data == b'':
            open_file.close()
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            shutil.copy(".scp.tmp", full_path)
            open_file = open(".scp.tmp", "wb+")
        else:
            open_file.write(data)
    os.remove(".scp.tmp")


class BaseCache:

    def all_runs(self):
        pass

    def has_run(self, task_name, version, run_id):
        pass

    def get_newest_runs(self, task_name, version, matching):
        pass

    def is_done(self, task_name, version, run_id):
        pass

    def record_run(self, task_name, version, run_id, kwargs):
        pass

    def conditional_record_run(self, task_name, version, run_id, kwargs, matching, rids_hash):
        pass

    def record_result(self, task_name, version, run_id, result):
        pass

    def get_result(self, task_name, version, run_id):
        pass

    def open(self, task_name, version, rid, path, mode):
        pass

    def copy_files(self, task_name, version, run_id):
        pass

    def make_run(self, task_name, version, run_id, raw_args, start_time, result, files, run_deps):
        pass

    def get_raw_args(self, task_name, version, run_id):
        pass

    def get_start_time(self, task_name, version, run_id):
        pass
    
    def add_run_dependency(self, task_name, version, run_id, other_task_name, other_version, other_run_id):
        pass

    def get_run_dependencies(self, task_name, version, run_id):
        pass
    
    def delete_run(self, task_name, version, run_id):
        pass

    def transfer(self, other):
        for task_name, version, run_id in self.all_runs():
            if self.is_done(task_name, version, run_id):
                if not other.has_run(task_name, version, run_id) or not other.is_done(task_name, version, run_id):
                    raw_args = self.get_raw_args(task_name, version, run_id)
                    files = self.copy_files(task_name, version, run_id)
                    result = self.get_result(task_name, version, run_id)
                    start_time = self.get_start_time(task_name, version, run_id)
                    run_dependencies = self.get_run_dependencies(task_name, version, run_id)
                    other.make_run(task_name, version, run_id, raw_args, start_time, result, files, run_dependencies)

    def save(self, path):
        runs = []
        if os.path.isdir("dj_cache_save_tmp"):
            shutil.rmtree("dj_cache_save_tmp")
        os.makedirs("dj_cache_save_tmp/")

        class Saver(BaseCache):
            def make_run(self, task_name, version, run_id, raw_args, start_time, result, files, run_deps):
                runs.append({
                    "task_name": task_name,
                    "version": version,
                    "run_id": run_id,
                    "raw_args": raw_args,
                    "start_time":start_time,
                    "run_deps": run_deps
                    })
                os.makedirs(f"dj_cache_save_tmp/{run_id}/user_files")
                with open(f"dj_cache_save_tmp/{run_id}/result.dill", "bw+") as f:
                    dill.dump(result, f)
                paste(files, f"dj_cache_save_tmp/{run_id}/user_files")
        
        self.transfer(Saver())
        with open("dj_cache_save_tmp/runs.dill", "bw+") as f:
            dill.dump(runs, f)
        
        make_dir(path)
        shutil.make_archive(path, 'zip', "dj_cache_save_tmp")
        shutil.rmtree("dj_cache_save_tmp")


    def update(self, path):
        if os.path.isdir("dj_cache_load_tmp"):
            shutil.rmtree("dj_cache_load_tmp")
        os.makedirs("dj_cache_load_tmp")
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall("dj_cache_load_tmp")
        
        with open("dj_cache_load_tmp/runs.dill", "br") as f:
            runs = dill.load(f)

        for run in runs:
            rid = run["run_id"]
            with open(f"dj_cache_load_tmp/{rid}/result.dill", "br") as f:
                result = dill.load(f)
            self.make_run(run["task_name"], run["version"], rid, run["raw_args"], run["start_time"], result, copy(f"dj_cache_load_tmp/{rid}/user_files"), run["run_deps"])
        shutil.rmtree("dj_cache_load_tmp")
    
    def clean(self):
        for task_name, version, run_id in self.all_runs():
            if not self.is_done(task_name, version, run_id):
                self.delete_run(task_name, version, run_id)


class NoCache(BaseCache):
    pass

def sync(cache1, cache2):
    cache1.transfer(cache2)
    cache2.transfer(cache1)

def make_raw_args(kwargs):
    document = {}
    #print(1, kwargs)
    for key, val in _serialize(kwargs).items():
        document["arg_" + key[4:]] = val

    return document

def _serialize(obj):
    if type(obj) is dict:
        out = {}
        for key, val in obj.items():
            out[_serialize(key)] = _serialize(val)
        return out
    
    if type(obj) is list:
        out = []
        for item in obj:
            out.append(_serialize(item))
        return out
    
    if type(obj) is tuple:
        out = []
        for item in obj:
            out.append(_serialize(item))
        return tuple(out)
    
    if type(obj) in [int, float, bool]:
        return obj
    if type(obj) is str:
        return "str_" + obj
    if type(obj) is datajuicer.task.Run:
        return "run_" + obj.run_id
    if callable(obj):
        return f"func_{obj.__module__}_{obj.__name__}"
    if obj is None:
        return "none"
    return f"hash_{hash(dill.dumps(obj))}"