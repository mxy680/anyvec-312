import subprocess
print("Subprocess module is available!")
result = subprocess.run(["echo", "Hello from subprocess!"], capture_output=True, text=True)
print("Output:", result.stdout)