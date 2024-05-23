import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np

src_path = "cmake-build-debug-standard/src/"
cl_path = "build\src\cl"

cuda_path = "cmake-build-debug-standard/src/cuda/Tarea2CUDA"

cpu_exe = "Tarea2CPU"
cl_exe = "Tarea2CL.exe"

def testCPU(n,m,t):
    output = "output.txt"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    current_dir = os.path.join(current_dir, src_path)
    os.chdir(current_dir)
    exe_path = os.path.join(current_dir, cpu_exe)
    args = [str(n), str(m), str(t), output]
    result = subprocess.run([exe_path] + args, capture_output=True, text=True)
    print("Salida estándar:", result.stdout)
    print("Error estándar:", result.stderr)
    if result.returncode == 0: 
        outputFile = open(output, "r")

        lines = outputFile.readlines()
        if lines:  # Verificar que el archivo no esté vacío
            last_line = lines[-1]
            tiempo_final = last_line.split(":")[1]
            return tiempo_final

def testCL(n,m,g,l,t,shared):
    output = "output2.txt"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.path.join(current_dir, cl_path)
    os.chdir(current_dir)
    exe_path = os.path.join(current_dir, cl_exe)
    args = [str(n), str(m), str(g), str(l), str(t),str(shared), output]
    result = subprocess.run([exe_path] + args, capture_output=True, text=True)
    print("Salida estándar:", result.stdout)
    print("Error estándar:", result.stderr)

    if result.returncode == 0: 
        outputFile = open(output, "r")
        lines = outputFile.readlines()
        if lines:  # Verificar que el archivo no esté vacío
            last_line = lines[-1]
            print("Última línea:", last_line)
            tiempo_final = last_line.split(",")[4].strip("\n")
            return tiempo_final
    
def testCUDA(n,m,g,t,shared):
    output = "output.txt"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exe_path = os.path.join(current_dir, cuda_path)
    outputfile_aux = os.path.join(current_dir,"",output)
    args = [str(n), str(m), str(t), str(g), str(shared), output]
    result = subprocess.run([exe_path] + args, capture_output=True, text=True)
    output = result.stdout

    valsdict = {}

    lines = output.split("\n")
    if lines:  # Verificar que el archivo no esté vacío
        for ln in lines[-6:]:
            vv = ln.split(":")
            valsdict[vv[0]] = float(vv[1])

        last_line = lines[-1]
        print(output)
        tiempo_final = valsdict["tt"]
        tiempo_ej = valsdict["ke"]

        return tiempo_final, tiempo_ej

valores = [x for x in range(32, 512+32, 32)]
steps = 50
clRes = []
clExRes = []
sharedRes = []
sharedExRes = []
cpuRes = []
for val in valores:
    res, ejres = testCUDA(val,val,18,steps,0)
    clRes.append(float(res))
    clExRes.append(float(ejres))

    res, ejres = testCUDA(val,val,18,steps,1)
    sharedRes.append(float(res))
    sharedExRes.append(float(ejres))

    res = testCPU(val,val,steps)
    cpuRes.append(float(res))

print(sharedRes)
print(clRes)
print(cpuRes)


plt.plot(valores, clRes, label='Cuda - Tiempo total', marker='o')
plt.plot(valores, clExRes, label='Cuda - Tiempo de kernel', marker='o')
plt.plot(valores, sharedRes, label='Cuda (Memoria compartida) - Tiempo total', marker='P')
plt.plot(valores, sharedExRes, label='Cuda (Memoria compartida) - Tiempo de kernel', marker='P')
plt.plot(valores, cpuRes,label='CPU', marker='x')
# Añadir título y etiquetas
plt.title('Comparacion metodos base')
plt.xlabel('Tamaño de grilla (NxN)')
plt.ylabel('Tiempo')

# Añadir una leyenda
plt.legend()

# Mostrar el gráfico
plt.savefig("CUDAGraph.png")
plt.clf()


plt.plot(valores, sharedRes, label='Cuda - Tiempo total', marker='o')
plt.plot(valores, sharedExRes, label='Cuda - Tiempo de kernel', marker='o')
plt.plot(valores, cpuRes,label='CPU', marker='x')
# Añadir título y etiquetas
plt.title('Tiempo de ejecución en memoria compartida')
plt.xlabel('Tamaño de grilla (NxN)')
plt.ylabel('Tiempo')

# Añadir una leyenda
plt.legend()

# Mostrar el gráfico
plt.savefig("CUDAGraphShared.png")
