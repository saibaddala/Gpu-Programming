import os
import shutil
import subprocess
from datetime import datetime

current_dir = os.getcwd()
INPUT = os.path.join(current_dir, "input")
OUTPUT = os.path.join(current_dir, "output")
SUBMIT = os.path.join(current_dir, "submit")

log_file_path = os.path.join(current_dir, "logFile.txt")
with open(log_file_path, "a") as log_file:
    log_file.write("======= START ========\n")
    log_file.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

os.chdir(SUBMIT)
cuda_files = [f for f in os.listdir() if f.endswith(".cu")]
ROLLNO = os.path.splitext(cuda_files[-1])[0]
print(ROLLNO)

shutil.copyfile(f"{ROLLNO}.cu", "main.cu")

subprocess.run(['python', 'compile.py'], check=True)

for i in range(0,5):
    filename="input_"+str(i)
    input_file_path = os.path.join(INPUT, "input_"+str(i))
    output_file_path = os.path.join(OUTPUT, "output_"+str(i))
    computed_ans_path = os.path.join(OUTPUT, "output_file_"+str(i))
    
    subprocess.run([os.path.join(SUBMIT, "main.exe"), input_file_path,computed_ans_path],check=True)
    
    with open(output_file_path, "r") as actual_ans, open(computed_ans_path, "r") as computed_ans:
        input_content = actual_ans.read()
        output_content = computed_ans.read()
        
        if input_content == output_content:
            print(f"{filename} success")
            with open(log_file_path, "a") as log_file:
                log_file.write(f"{filename} success\n")
        else:
            print(f"{filename} failure")
            with open(log_file_path, "a") as log_file:
                log_file.write(f"{filename} failure\n")

main_path=os.getcwd()+"\\main.cu"
os.remove(main_path)
main_path=os.getcwd()+"\\main.exp"
os.remove(main_path)
main_path=os.getcwd()+"\\main.lib"
os.remove(main_path)
main_path=os.getcwd()+"\\main.exe"
os.remove(main_path)
main_path=os.getcwd()+"\\Renderer.obj"
os.remove(main_path)
main_path=os.getcwd()+"\\SceneNode.obj"
os.remove(main_path)
main_path=os.getcwd()+"\\main.obj"
os.remove(main_path)

# Logging
with open(log_file_path, "a") as log_file:
    log_file.write("========== END ==========\n")
    log_file.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
