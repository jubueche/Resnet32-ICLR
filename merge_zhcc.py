import os
import os.path

print("please type the ssh location of the repository (e.g. jbu@zhcc022.zurich.ibm.com:~/Master-Thesis alternatively type 'j'")
repo = input()
if repo=="j":
    repo = "jbu@zhcc022.zurich.ibm.com:~/Master-Thesis"
s = os.path.join(repo, "Sessions")
r = os.path.join(repo, "Resources")
os.system(f"scp -r {s} from_zhcc_sessions/")
os.system(f"scp -r {r} from_zhcc_resources/")
print("Press Enter to continue")
input()
os.system("python merge.py -sessions_dirs from_zhcc_sessions Sessions -resources_dirs from_zhcc_resources Resources")