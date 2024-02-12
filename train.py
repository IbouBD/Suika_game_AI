from threading import Thread
import subprocess

def run_script(script_name):
    subprocess.run(["python", script_name])

t1 = Thread(target=run_script, args=("env1.py",))
t2 = Thread(target=run_script, args=("env1.py",))
t3 = Thread(target=run_script, args=("env1.py",))
t4 = Thread(target=run_script, args=("env1.py",))
t5 = Thread(target=run_script, args=("env1.py",))
t6 = Thread(target=run_script, args=("env1.py",))



t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()

t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()