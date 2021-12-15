from sqlite3.dbapi2 import OperationalError
import threading
import datajuicer._global
import sqlite3
import json
import ctypes
import os
import time
import posix_ipc
import mmap
import datajuicer.utils as utils


# class Semaphore:
#     def __init__(self, path, init=True):
#         datajuicer.cache.make_dir(path)
#         self.db_file = path
#         while(True):
#             try:
#                 conn = sqlite3.connect(self.db_file, timeout=100, isolation_level="EXCLUSIVE")
#                 cur = conn.cursor()
#                 #cur.execute("PRAGMA JOURNAL_MODE = 'WAL'")
#                 cur.execute("CREATE TABLE IF NOT EXISTS 'semaphore' (data);")
#                 cur.execute("SELECT * FROM 'semaphore';")
#                 if len(cur.fetchall()) == 0:
#                     cur.execute(f"INSERT INTO 'semaphore' (data) VALUES ({0});")
#                 elif init:
#                     cur.execute(f"UPDATE 'semaphore' SET data = {0}")
#                 conn.commit()
#                 break
#             except OperationalError as e:
#                 conn.rollback()
#                 raise e
#             finally:
#                 conn.commit()
#                 conn.close()

#     def available(self, cur):
#         cur.execute("SELECT * FROM 'semaphore';")
#         return int(cur.fetchall()[0][0])
    
#     def acquire(self, value=1):
#         while(True):
#             try:
#                 conn = sqlite3.connect(self.db_file, timeout=1, isolation_level="EXCLUSIVE")
#                 cur = conn.cursor()
#                 #cur.execute("PRAGMA JOURNAL_MODE = 'WAL'")
#                 #print(f"I process {os.getpid()} thread {threading.current_thread().ident} run {datajuicer._global.run_id()} have locked the database")
#                 av = self.available(cur)
#                 if av >= value:
#                     cur.execute(f"UPDATE 'semaphore' SET data = {av-1};")
#                     conn.commit()
#                     #print(f"I process {os.getpid()} thread {threading.current_thread().ident} run {datajuicer._global.run_id()} have unlocked the database")
#                     return
#             except OperationalError as e:
#                 conn.rollback()
#                 raise e
#             finally:
#                 conn.commit()
#                 conn.close()
#             time.sleep(0.1)
    
#     def release(self, value=1):
#         while(True):
#             try:
#                 conn = sqlite3.connect(self.db_file, timeout=1, isolation_level="EXCLUSIVE")
#                 cur = conn.cursor()
#                 #cur.execute("PRAGMA JOURNAL_MODE = 'WAL'")
#                 #print(f"I process {os.getpid()} thread {threading.current_thread().ident} run {datajuicer._global.run_id()} have locked the database")
#                 av = self.available(cur)
#                 cur.execute(f"UPDATE 'semaphore' SET data = {av+value};")
#                 conn.commit()
#                 #print(f"I process {os.getpid()} thread {threading.current_thread().ident} run {datajuicer._global.run_id()} have unlocked the database")
#                 break
#             except OperationalError as e:
#                 conn.rollback()
#                 raise e
#             finally:
#                 conn.commit()
#                 conn.close()
#             time.sleep(0.1)
            

class ResourceLock:
    def __init__(self, directory = "dj_resources/", init=True, session=None):

        self.directory = directory
        datajuicer.cache.make_dir(directory)
        if session is None:
            session = utils.rand_id()
        #self.workers_semaphore = Semaphore(os.path.join(directory, ".worker_semaphore"),init)
        if init:
            try:
                posix_ipc.unlink_semaphore(f"djworkers_{session}")
            except Exception:
                pass
            try:
                posix_ipc.unlink_shared_memory(f"djresources_{session}")
            except Exception:
                pass
            try:
                posix_ipc.unlink_semaphore(f"djlock_{session}")
            except Exception:
                pass
        self.workers_semaphore = posix_ipc.Semaphore(f"djworkers_{session}", posix_ipc.O_CREAT)
        self.lock = posix_ipc.Semaphore(f"djlock_{session}", posix_ipc.O_CREAT)
        self.lock.release()
        memory = posix_ipc.SharedMemory(f"djresources_{session}", posix_ipc.O_CREAT, size=100)
        self.mapfile = mmap.mmap(memory.fd, memory.size)

        memory.close_fd()
        self.session = session
        with self.lock:
            self.mapfile.seek(0)
            self.mapfile.write(json.dumps({}).encode()+ b"\0")
        # self.path = os.path.join(directory, ".resources")
        # datajuicer.cache.make_dir(self.path)
        # conn = sqlite3.connect(self.path, timeout=1, isolation_level="EXCLUSIVE")
        # cur = conn.cursor()
        # cur.execute("CREATE TABLE IF NOT EXISTS 'resources' (data);")
        # cur.execute("SELECT * FROM resources;")
        # if len(cur.fetchall()) == 0:
        #     cur.execute("INSERT INTO 'resources' (data) VALUES (\"{}\");")
        # elif init:
        #     cur.execute("UPDATE resources SET data = \"{}\"")
        # conn.commit()
        # conn.close()

        #COMMENT THIS BACK OUT
        #self.release()
        #self.release()

    def available(self):
        with self.lock:
            self.mapfile.seek(0)
            s = []
            c = self.mapfile.read_byte()
            while c != 0:
                s.append(c)
                c = self.mapfile.read_byte()
            s = [chr(c) for c in s]
            s = ''.join(s)
            return json.loads(s)
        #cur.execute("SELECT * FROM 'resources';")
        #return json.loads(cur.fetchall()[0][0])
    
    def is_available(self, available, resources):
        for val in resources.values():
            if val < 0:
                raise TypeError
        for k, v in resources.items():
            if not k in available:
                return False
            if available[k] < v:
                return False
        return True
    
    def reserve_resources(self, **resources):
        uid = threading.current_thread().unique_id
        local_res_path = os.path.join(self.directory, f"{uid}_resources.json")
        if os.path.exists(local_res_path):
            with open(local_res_path, "r") as f:
                local_resources = json.load(f)
        else:
            local_resources = {}
        
        self.workers_semaphore.release()
        while(True):
            # conn = sqlite3.connect(self.path, timeout=1)
            # cur = conn.cursor()
            # cur.execute("BEGIN EXCLUSIVE")
            # av = self.available(cur)
            av = self.available()
            if self.is_available(av, resources):
                for k, v in resources.items():
                    if not k in local_resources:
                        local_resources[k] = 0
                    av[k] -= v
                    local_resources[k] += v
                with open(local_res_path, "w+") as f:
                    f.truncate(0)
                    json.dump(local_resources, f)
                with self.lock:
                    self.mapfile.seek(0)
                    self.mapfile.write(json.dumps(av).encode() + b"\0")
                # cur.execute(f"UPDATE resources SET data = '{json.dumps(av)}'")
                # conn.commit()
                # conn.close()
                break
            # conn.commit()
            # conn.close()
            time.sleep(0.1)

        self.workers_semaphore.acquire()

    def free_resources(self, **resources):
        uid = threading.current_thread().unique_id
        local_res_path = os.path.join(self.directory, f"{uid}_resources.json")
        if os.path.exists(local_res_path):
            with open(local_res_path, "r") as f:
                local_resources = json.load(f)
        else:
            local_resources = {}
        
        self.workers_semaphore.release()
        # conn = sqlite3.connect(self.path, timeout=1)
        # cur = conn.cursor()
        # cur.execute("BEGIN EXCLUSIVE")
        # av = self.available(cur)
        av = self.available()
        for k,v in resources.items():
            if not k in local_resources:
                local_resources[k] = 0
            if not k in av:
                av[k] = 0
            local_resources[k] -= v
            av[k] += v 
        with open(local_res_path, "w+") as f:
            f.truncate(0)
            json.dump(local_resources, f)
        # cur.execute(f"UPDATE resources SET data = '{json.dumps(av)}'")
        # conn.commit()
        # conn.close()
        with self.lock:
            self.mapfile.seek(0)
            self.mapfile.write(json.dumps(av).encode()+ b"\0")
        self.workers_semaphore.acquire()

    def free_all_resources(self):
        uid = threading.current_thread().unique_id
        local_res_path = os.path.join(self.directory, f"{uid}_resources.json")
        if os.path.exists(local_res_path):
            with open(local_res_path, "r") as f:
                local_resources = json.load(f)
            self.free_resources(**local_resources)

    def acquire(self):
        self.workers_semaphore.acquire()
        #print(self.workers_semaphore.value)

    def release(self):
        self.workers_semaphore.release()
        #print(self.workers_semaphore.value)

# class ResourceLock:
#     def __init__(self, n_thread_semaphore, available_cond, shared_file = "dj_resources/resources.json"):
#         self.available_cond = available_cond
#         self.n_thread_semaphore = n_thread_semaphore
#         self.shared_file = shared_file

#     def available(self, **resources):
#         for val in resources.values():
#             if val < 0:
#                 raise TypeError
#         if os.path.exists(self.shared_file):
#             with open(self.shared_file, "r") as f:
#                 av = json.load(f)
#         else:
#             av = {}
#         for k, v in resources.items():
#             if not k in av:
#                 return False
#             if av[k] < v:
#                 return False
#         return True, av
                
        


#     def reserve_resources(self, **resources):
#         uid = threading.current_thread().unique_id
#         local_res_path = os.path.join(os.path.dirname(self.shared_file), f"{uid}_resources.json")
#         if os.path.exists(local_res_path):
#             with open(local_res_path, "r") as f:
#                 local_resources = json.load(f)
#         else:
#             local_resources = {}
        
#         self.n_thread_semaphore.release()
#         with self.available_cond:
#             while(True):
#                 is_available, av = self.available(**resources)
#                 if is_available:
#                     break
#             for k, v in resources.items():
#                 if not k in local_resources:
#                     local_resources[k] = 0
#                 if not k in av:
#                     is_available = False
#                     break
#                 if av[k] < v:
#                     is_available = False
#                     break
#                 av[k] -= v
#                 local_resources[k] += v
#             with open(local_res_path, "w+") as f:
#                 f.truncate(0)
#                 json.dump(local_resources, f)
#             with open(self.shared_file, "w+") as f:
#                 f.truncate(0)
#                 json.dump(av, f)
#         self.n_thread_semaphore.acquire()

#     def free_resources(self, **resources):
#         uid = threading.current_thread().unique_id
#         local_res_path = os.path.join(os.path.dirname(self.shared_file), f"{uid}_resources.json")
#         if os.path.exists(local_res_path):
#             with open(local_res_path, "r") as f:
#                 local_resources = json.load(f)
#         else:
#             local_resources = {}
        
#         self.n_thread_semaphore.release()
#         with self.available_cond:
#             if os.path.exists(self.shared_file):
#                 with open(self.shared_file, "r") as f:
#                     av = json.load(f)
#             else:
#                 av = {}
#             for k,v in resources.items():
#                 if not k in local_resources:
#                     local_resources[k] = 0
#                 if not k in av:
#                     av[k] = 0
#                 local_resources[k] -= v
#                 av[k] += v 
#             with open(local_res_path, "w+") as f:
#                 f.truncate(0)
#                 json.dump(local_resources, f)
#             with open(self.shared_file, "w+") as f:
#                 f.truncate(0)
#                 json.dump(av, f)
#             self.available_cond.notify_all()
#         self.n_thread_semaphore.acquire()

#     def free_all_resources(self):
#         uid = threading.current_thread().unique_id
#         local_res_path = os.path.join(os.path.dirname(self.shared_file), f"{uid}_resources.json")
#         if os.path.exists(local_res_path):
#             with open(local_res_path, "r") as f:
#                 local_resources = json.load(f)
#             self.free_resources(**local_resources)

#     def acquire(self):
#         self.n_thread_semaphore.acquire()

#     def release(self):
#         self.n_thread_semaphore.release()

                
        


#     def reserve_resources(self, **resources):
#         uid = threading.current_thread().unique_id
#         local_res_path = os.path.join(os.path.dirname(self.shared_file), f"{uid}_resources.json")
#         if os.path.exists(local_res_path):
#             with open(local_res_path, "r") as f:
#                 local_resources = json.load(f)
#         else:
#             local_resources = {}
        
#         self.n_thread_semaphore.release()
#         with self.available_cond:
#             while(True):
#                 is_available, av = self.available(**resources)
#                 if is_available:
#                     break
#             for k, v in resources.items():
#                 if not k in local_resources:
#                     local_resources[k] = 0
#                 if not k in av:
#                     is_available = False
#                     break
#                 if av[k] < v:
#                     is_available = False
#                     break
#                 av[k] -= v
#                 local_resources[k] += v
#             with open(local_res_path, "w+") as f:
#                 f.truncate(0)
#                 json.dump(local_resources, f)
#             with open(self.shared_file, "w+") as f:
#                 f.truncate(0)
#                 json.dump(av, f)
#         self.n_thread_semaphore.acquire()

#     def free_resources(self, **resources):
#         uid = threading.current_thread().unique_id
#         local_res_path = os.path.join(os.path.dirname(self.shared_file), f"{uid}_resources.json")
#         if os.path.exists(local_res_path):
#             with open(local_res_path, "r") as f:
#                 local_resources = json.load(f)
#         else:
#             local_resources = {}
        
#         self.n_thread_semaphore.release()
#         with self.available_cond:
#             if os.path.exists(self.shared_file):
#                 with open(self.shared_file, "r") as f:
#                     av = json.load(f)
#             else:
#                 av = {}
#             for k,v in resources.items():
#                 if not k in local_resources:
#                     local_resources[k] = 0
#                 if not k in av:
#                     av[k] = 0
#                 local_resources[k] -= v
#                 av[k] += v 
#             with open(local_res_path, "w+") as f:
#                 f.truncate(0)
#                 json.dump(local_resources, f)
#             with open(self.shared_file, "w+") as f:
#                 f.truncate(0)
#                 json.dump(av, f)
#             self.available_cond.notify_all()
#         self.n_thread_semaphore.acquire()

#     def free_all_resources(self):
#         uid = threading.current_thread().unique_id
#         local_res_path = os.path.join(os.path.dirname(self.shared_file), f"{uid}_resources.json")
#         if os.path.exists(local_res_path):
#             with open(local_res_path, "r") as f:
#                 local_resources = json.load(f)
#             self.free_resources(**local_resources)

#     def acquire(self):
#         self.n_thread_semaphore.acquire()

#     def release(self):
#         self.n_thread_semaphore.release()


# class ResourceLock:
#     def __init__(self, db_file="dj_runs/resources.db", init=True):
#         datajuicer.cache.make_dir(db_file)
#         self.db_file = db_file
#         self.available_cond = Condition()
#         conn = sqlite3.connect(self.db_file, timeout=100)
#         cur = conn.cursor()
#         #cur.execute("BEGIN EXCLUSIVE")
#         cur.execute("CREATE TABLE IF NOT EXISTS 'resources' (data);")
#         cur.execute("SELECT * FROM resources;")
#         if len(cur.fetchall()) == 0:
#             cur.execute("INSERT INTO 'resources' (data) VALUES (\"{}\");")
#         elif init:
#             cur.execute("UPDATE resources SET data = \"{}\"")
#         conn.commit()
#         conn.close()

#     def free_resources(self, **resources):
#         uid = threading.current_thread().unique_id
#         for val in resources.values():
#             if val < 0:
#                 raise TypeError
#         conn = sqlite3.connect(self.db_file, timeout=100)
#         cur = conn.cursor()
#         #cur.execute("BEGIN EXCLUSIVE")
#         cur.execute("SELECT * FROM resources;")
#         av = json.loads(cur.fetchall()[0][0])
#         local_res_path = os.path.join(os.path.dirname(self.db_file), f"{uid}_resources.json")
#         if os.path.exists(local_res_path):
#             with open(local_res_path, "r") as f:
#                 local_resources = json.load(f)
#         else:
#             local_resources = {}
#         for k,v in resources.items():
#             if not k in local_resources:
#                 local_resources[k] = 0
#             if not k in av:
#                 av[k] = 0
#             local_resources[k] -= v
#             av[k] += v 
#         with open(local_res_path, "w+") as f:
#             f.truncate(0)
#             json.dump(local_resources, f)
#         cur.execute(f"UPDATE resources SET data = '{json.dumps(av)}';")
#         conn.commit()
#         conn.close()
#         with self.available_cond:
#             self.available_cond.notify_all()
    
#     def reserve_resources(self, **resources):
#         #self.free_resources(threads=1)
#         #resources["threads"] = resources.get("threads", 0) + 1
#         uid = threading.current_thread().unique_id
#         for val in resources.values():
#             if val < 0:
#                 raise TypeError
#         local_res_path = os.path.join(os.path.dirname(self.db_file), f"{uid}_resources.json")
#         if os.path.exists(local_res_path):
#             with open(local_res_path, "r") as f:
#                 local_resources = json.load(f)
#         else:
#             local_resources = {}
#         while(True):
#             conn = sqlite3.connect(self.db_file, timeout=100)
#             try:
#                 cur = conn.cursor()
#                 #cur.execute("BEGIN EXCLUSIVE;")
#                 cur.execute("SELECT * FROM resources;")
#                 av = json.loads(cur.fetchall()[0][0])
#                 is_available = True
#                 for k, v in resources.items():
#                     if not k in local_resources:
#                         local_resources[k] = 0
#                     if not k in av:
#                         is_available = False
#                         conn.commit()
#                         break
#                     if av[k] < v:
#                         is_available = False
#                         conn.commit()
#                         break
#                     av[k] -= v
#                     local_resources[k] += v
#                 if is_available:
#                     cur.execute(f"UPDATE resources SET data = '{json.dumps(av)}';")
#                     with open(local_res_path, "w+") as f:
#                         f.truncate(0)
#                         json.dump(local_resources, f)
#                     conn.commit()
#                     break
#             except sqlite3.OperationalError as e:
#                 conn.rollback()
#             finally:
#                 conn.close()
#             with self.available_cond:
#                 self.available_cond.wait(0.1)
#             # time.sleep(1)

#     def free_all_resources(self):
#         uid = threading.current_thread().unique_id
#         local_res_path = os.path.join(os.path.dirname(self.db_file), f"{uid}_resources.json")
#         if os.path.exists(local_res_path):
#             with open(local_res_path, "r") as f:
#                 local_resources = json.load(f)
#         self.free_resources(**local_resources)


# class ResourceLock:
#     def __init__(self, max_n_threads=1, **max_resources):
#         self.threads = {}
#         self.max_resources = max_resources
#         self.lock = threading.RLock()
#         self.used_resources = {}
#         self.available_cond = threading.Condition(self.lock)
#         self.n_thread_semaphore = threading.Semaphore(max_n_threads)

#     def available(self, **resources):
#         all_used_resources = {}
#         for key in resources:
#             all_used_resources[key] = 0.0
#         for d in self.used_resources.values():
#             for key, val in d.items():
#                 if not key in all_used_resources:
#                     all_used_resources[key] = 0.0
#                 all_used_resources[key] += val
#         return all([all_used_resources[key] + val <= self.max_resources[key] for key, val in resources.items()])

#     def reserve_resources(self, **resources):
#         for val in resources.values():
#             if val < 0:
#                 raise TypeError
#         rid = datajuicer._global.run_id()
#         self.release()
#         with self.available_cond:
#             while not self.available(**resources):
#                 self.available_cond.wait()
#             if not rid in self.used_resources:
#                 self.used_resources[rid] = {}
#             for key, val in resources.items():
#                 if not key in self.used_resources[rid]:
#                     self.used_resources[rid][key] = 0.0
#                 self.used_resources[rid][key] += val
#         self.acquire()

#     def free_resources(self, **resources):
#         for val in resources.values():
#             if val < 0:
#                 raise TypeError
#         rid = datajuicer._global.run_id()
#         self.release()
#         if rid in self.used_resources:
#             with self.available_cond:
#                 for key, val in resources.items():
#                     if key in self.used_resources[rid]:
#                         if self.used_resources[rid][key] > val:
#                             self.used_resources[rid][key] -= val
#                 self.available_cond.notify_all()
#         self.acquire()

#     def free_all_resources(self):
#         rid = datajuicer._global.run_id()
#         self.release()
#         with self.available_cond:
#             if rid in self.used_resources:
#                 del self.used_resources[rid]
#             self.available_cond.notify_all()

#         self.n_thread_semaphore.acquire()

#     def acquire(self):
#         self.n_thread_semaphore.acquire()

#     def release(self):
#         self.n_thread_semaphore.release()
