import subprocess

subprocess.run(['nvcc', '-std=c++17', '-c', 'SceneNode.cpp', '-o', 'SceneNode'], check=True)
subprocess.run(['nvcc', '-std=c++17', '-c', 'Renderer.cpp', '-o', 'Renderer'], check=True)
subprocess.run(['nvcc', '-c', 'main.cu', '-o', 'main'], check=True)
subprocess.run(['nvcc', 'SceneNode.obj', 'Renderer.obj', 'main.obj', '-o', 'main.exe'], check=True)
