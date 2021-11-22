import argparse
import sqlite3
import os


print("merging...")

ap = argparse.ArgumentParser()
ap.add_argument("-sessions_dirs", nargs="*", default=[])
ap.add_argument("-resources_dirs", nargs="*", default=[])

flags =ap.parse_args()

sessions_dirs = [sd[0:-1] if sd[-1]=="/" else sd for sd in flags.sessions_dirs]
resources_dirs = [rd[0:-1] if rd[-1]=="/" else rd for rd in flags.resources_dirs]

merge_sessions = len(sessions_dirs)>1
merge_resources = len(resources_dirs)>1

if merge_resources:
    os.system("rsync -abviuzP " + " ".join([rd+'/' for rd in resources_dirs]))

if merge_sessions:

    os.system("cp " + sessions_dirs[0]+ "/sessions.db "+ "combined_tmp.db")

    os.system("rsync -abviuzP " + " ".join([sd+'/' for sd in sessions_dirs]))

    con3 = sqlite3.connect("combined_tmp.db")

    for sd in sessions_dirs[1:]:
        print("dba = " + str(sd+"/sessions.db"))
        if not os.path.isfile(sd+"/sessions.db"):
            continue
        con3.execute("ATTACH '"+ str(sd+"/sessions.db") + "' as dba")

        con3.execute("BEGIN")
        for row in con3.execute("SELECT * FROM dba.sqlite_master WHERE type='table'"):
            combine = "INSERT OR IGNORE INTO "+ row[1] + " SELECT * FROM dba." + row[1]
            print(combine)
            con3.execute(combine)
            delete_duplicates = f"DELETE FROM {row[1]} WHERE rowid NOT IN (SELECT MIN(rowid) FROM {row[1]} GROUP BY session_id )"
            print(delete_duplicates)
            con3.execute(delete_duplicates)
        con3.commit()
        con3.execute("detach database dba")

    con3.close()

    
    os.system(f"mv -f combined_tmp.db {sessions_dirs[-1]}/sessions.db")




