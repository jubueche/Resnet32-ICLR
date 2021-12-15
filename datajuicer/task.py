import threading
import datajuicer
from datajuicer.resource_lock import ResourceLock
import datajuicer.utils as utils
import copy
import inspect
from datajuicer._global import GLOBAL, _open
import sys
import traceback
from functools import wraps, partial
import os
import dill
import time

TIMEOUT = 1.0

# class Namespace:
#     pass

# # # @processify
# def _in_new_process(kwargs, task_info, force, incognito, parent_task_name, parent_task_version, parent_run_id):
#     #multiprocessing.set_start_method('spawn')
#     curthread = threading.current_thread()
#     #assert(type(curthread) != datajuicer.task.Run)
#     task = datajuicer.Task.make(*task_info[0:-2], **task_info[-2])(dill.loads(task_info[-1]))
#     datajuicer.logging.enable_proxy()
#     GLOBAL.resource_lock = ResourceLock(directory=task.resource_lock.directory)
#     GLOBAL.cache = task.cache
#     run = Run(task, kwargs, force, incognito, False)
#     pseudo_parent = Namespace()
#     pseudo_parent.task = Namespace()
#     if not parent_task_name is None:
#         pseudo_parent.task.name = parent_task_name
#         pseudo_parent.task.version = parent_task_version
#         pseudo_parent.run_id = parent_run_id
#         run.start(pseudo_parent)
#     else:
#         run.start()

#     return run.get(), run.run_id

# def process_func(q, *args, **kwargs):
#     try:
#         ret = _in_new_process(*args, **kwargs)
#     except Exception:
#         ex_type, ex_value, tb = sys.exc_info()
#         error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
#         ret = None
#     else:
#         error = None

#     q.put((ret, error))

# def spawn_and_ret(*args,**kwargs):
#     ctx = multiprocessing.get_context('fork')
#     target = process_func
#     q = ctx.Queue()
#     p = ctx.Process(target=target, args=[q] + list(args), kwargs=kwargs)
#     p.start()
#     ret, error = q.get()
#     p.join()

#     if error:
#         ex_type, ex_value, tb_str = error
#         message = '%s (in subprocess)\n%s' % (ex_value.args, tb_str)
#         raise ex_type(message)

#     return ret

# def _in_new_process(q, kwargs, task, force, incognito, parent_task_name, parent_task_version, parent_run_id):
#     try:
#         #multiprocessing.set_start_method('spawn')
#         curthread = threading.current_thread()
#         assert(type(curthread) != datajuicer.task.Run)

#         datajuicer.logging.enable_proxy()
#         GLOBAL.resource_lock = task.resource_lock
#         GLOBAL.cache = task.cache
#         run = Run(task, kwargs, force, incognito, False)
#         pseudo_parent = Namespace()
#         pseudo_parent.task = Namespace()
#         if not parent_task_name is None:
#             pseudo_parent.task.name = parent_task_name
#             pseudo_parent.task.version = parent_task_version
#             pseudo_parent.run_id = parent_run_id
#             run.start(pseudo_parent)
#         else:
#             run.start()

#         ret = run.get()
#     except Exception:
#         ex_type, ex_value, tb = sys.exc_info()
#         error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
#         ret = None
#     else:
#         error = None

#     q.put((ret, error))

# def launch_process(parent_uid, *other_args):
#     path = os.path.join("dj_resources", parent_uid)
#     with open(path, "wb+") as f:
#         dill.dump(other_args, f)
#     subprocess.run(["python", "datajuicer/process.py", "-path", path]).check_returncode()
#     with open(path+"out", "rb") as f:
#         ret, error = dill.load(f)
#     if error:
#         ex_type, ex_value, tb_str = error
#         message = '%s (in subprocess)\n%s' % (ex_value.args, tb_str)
#         raise ex_type(message)

#     return ret

class Namespace:
    pass

class Run(threading.Thread):
    def __init__(self, task, kwargs, force, incognito):
        super().__init__()
        self._return = None
        self.task = task
        self.kwargs = kwargs
        self.run_id = None
        self.unique_id = utils.rand_id()
        self.resource_lock = task.resource_lock
        self.force = force
        self.incognito = incognito
        #self.dj = True
    
    def start(self) -> None:
        parent_thread = threading.currentThread()
        self.parent = parent_thread
        return super().start()

    def run(self):
        self.resource_lock.acquire()
        self._return = self.task._run()
        if hasattr(self.parent, "run_id") and self.parent.run_id is not None and not self.parent.incognito and not self.incognito:
            self.parent.task.cache.add_run_dependency(
                self.parent.task.name, 
                self.parent.task.version, 
                self.parent.run_id, 
                self.task.name, 
                self.task.version, 
                self.run_id)
        self.resource_lock.free_all_resources()
        self.resource_lock.release()
        

    def join(self):
        self.resource_lock.release()
        super().join()
        self.resource_lock.acquire()
        return self


    def get(self):
        self.join()
        return self._return
    
    def get_context(self):
        if type(self.parent) is Run and self.parent.run_id is not None:
            parent_task_name = self.parent.task.name
            parent_task_version = self.parent.task.version
            parent_run_id = self.parent.run_id
            parent_incognito = self.parent.incognito
        else:
            parent_task_name = None
            parent_task_version = None
            parent_run_id = None
            parent_incognito = None
        return {"kwargs": self.kwargs, 
                "force": self.force, 
                "incognito": self.incognito,
                "parent_task_name": parent_task_name,
                "parent_task_version": parent_task_version, 
                "parent_run_id": parent_run_id,
                "parent_incognito": parent_incognito,
                "unique_id":self.unique_id,
                "global_cache": GLOBAL.cache,
                "global_resource_lock_directory": GLOBAL.resource_lock.directory,
                "global_resource_lock_session": GLOBAL.resource_lock.session,
                "task_name": self.task.name,
                "task_version": self.task.version,
                "cache": self.task.cache,
                "run_id": self.run_id
                }
            
    
    def open(self, path, mode):
        if any([x in mode for x in ["a", "w", "+"]]):
            raise Exception("Only Reading allowed")
        #self.join()
        return self._open(path, mode)

    def _open(self, path, mode):
        return self.task.cache.open(self.task.name, self.task.version, self.run_id, path, mode)
    
    def assign_run_id(self, rid):
        self.run_id = rid
    
    # def __eq__(self, other):
    #     return type(self) == type(other) and (self is other or (self.run_id == other.run_id and self.run_id is not None))
    
    def __getitem__(self, item):
        return self.kwargs[item]
    
    def __getstate__(self):
        """Return state values to be pickled."""
        return self.get_context()

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        pseudo_task = Namespace()
        pseudo_task.name = state["task_name"]
        pseudo_task.version = state["task_version"]
        pseudo_task.cache = state["cache"]
        self.task = pseudo_task
        self.run_id = state["run_id"]
        self.kwargs = state["kwargs"]
        self.unique_id = state["unique_id"]
    
    def delete(self):
        self.task.cache.delete_run(self.task.name, self.task.version, self.run_id)


class Ignore:
    pass

class Keep:
    pass

class Depend:
    def __init__(self, *keeps, **deps):
        self.deps = deps

        if "keep" in self.deps:
            for key in self.deps["keep"]:
                self.deps[key] = Keep
            for key in keeps:
                self.deps[key] = Keep
            del self.deps["keep"]
        
        if "ignore" in self.deps:
            for key in self.deps["ignore"]:
                self.deps[key] = Ignore
            del self.deps["ignore"]
        
    def modify(self, kwargs):
        default = Keep
        if Keep in self.deps.values():
            default = Ignore

        new_kwargs = copy.copy(kwargs)
        for key in kwargs:
            if key in self.deps:
                action = self.deps[key]
            else:
                action = default
            if action == Ignore:
                del new_kwargs[key]
            elif type(action) is Depend:
                new_kwargs[key] = action.modify(new_kwargs[key])

        return kwargs

class Task:
    @staticmethod
    def make(name=None, version=0.0, resource_lock = None, cache=None, mode="thread", mode_args=None, **dependencies):
        return partial(Task, name=name, version=version, resource_lock=resource_lock, cache=cache, mode=mode, mode_args=mode_args, **dependencies)

    def __init__(self, func, name=None, version=0.0, resource_lock = None, cache=None, mode="thread", mode_args=None, **dependencies):
        self.func = func
        self.lock = threading.Lock()
        self.dependencies = dependencies
        self.get_dependencies = Depend(**dependencies).modify
        #self.conds = {}
        if resource_lock is None:
            resource_lock = GLOBAL.resource_lock
        if cache is None:
            cache = GLOBAL.cache
        self.cache = cache
        self.resource_lock = resource_lock
        self.version = version
        if name is None:
            name = func.__name__
        self.name = name
        self.version = version
        self.mode = mode
        self.mode_args = mode_args
        self.file = func.__globals__['__file__']
        datajuicer.GLOBAL.task_funcs[self.name] = self.func
        datajuicer.GLOBAL.task_versions[self.name] = self.version
        datajuicer.GLOBAL.task_caches[self.name] = self.cache


    def _run(self):
        def check_run_deps(cache, name, version, rid):
            if datajuicer.GLOBAL.task_versions[name] != version:
                return False
            rdeps = cache.get_run_dependencies(name, version, rid)
            for n, v, i in rdeps:
                if not check_run_deps(datajuicer.GLOBAL.task_caches[n], n, v, i):
                    return False
            return True

        context = threading.current_thread().get_context()
        
        force = context["force"]
        kwargs = context["kwargs"]
        incognito = context["incognito"]

        dependencies = self.get_dependencies(kwargs)
        if not force:
            rids = self.cache.get_newest_runs(self.name, self.version, dependencies)
            redo = False
            while(True):
                for rid in rids:
                    
                    if check_run_deps(self.cache, self.name, self.version, rid):
                        self.resource_lock.release()
                        while not self.cache.is_done(self.name, self.version, rid) and check_run_deps(self.cache, self.name, self.version, rid):
                            time.sleep(0.5)

                        if check_run_deps(self.cache, self.name, self.version, rid):
                            threading.current_thread().assign_run_id(rid)
                            self.resource_lock.acquire()
                            return self.cache.get_result(self.name, self.version, rid)
                        redo = True
                        break
                if redo:
                    rids = self.cache.get_newest_runs(self.name, self.version, dependencies)
                    continue
                new_rid = utils.rand_id()
                if incognito:
                    break
                success, rids = self.cache.conditional_record_run(self.name, self.version, new_rid, kwargs, dependencies, hash(rids))
                if success:
                    break
        else:
            new_rid = utils.rand_id()
        threading.current_thread().assign_run_id(new_rid)
        
        context["mode"] = self.mode
        context["mode_args"] = self.mode_args
        context["resource_lock_directory"] = self.resource_lock.directory
        context["resource_lock_session"] = self.resource_lock.session
        context["cache"] = self.cache
        #context["func"] = self.func
        context["run_id"] = new_rid
        context["task_name"] = self.name
        context["task_version"] = self.version
        context["file"] = self.file
        datajuicer.launch(context)

        if incognito:
            path = os.path.join(context["resource_lock_directory"], f"{context['unique_id']}_result.dill")
            with open(path, "rb") as f:
                return dill.read(f)
        else:
            return self.cache.get_result(self.name, self.version, new_rid)
        

    def bind_args(self, *args, **kwargs):
        boundargs = inspect.signature(self.func).bind(*args,**kwargs)
        boundargs.apply_defaults()
        return boundargs.arguments

    def __call__(self, *args, **kwargs):
        force = False
        incognito = False
        if "force" in kwargs:
            force = kwargs["force"]
            del kwargs["force"]
        if "incognito" in kwargs:
            incognito = kwargs["incognito"]
            del kwargs["incognito"]

        kwargs = dict(self.bind_args(*args, **kwargs))
        
        run = Run(self, kwargs, force=force, incognito=incognito)
        run.start()
        return run