import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np


def testCPU(n,m,t):
    output = "output.txt"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    current_dir = os.path.join(current_dir,"build\src")
    os.chdir(current_dir)
    exe_path = os.path.join(current_dir, "Tarea2CPU.exe")
    args = [str(n), str(m), str(t), output]
    result = subprocess.run([exe_path] + args, capture_output=True, text=True)
    print("Salida estándar:", result.stdout)
    print("Error estándar:", result.stderr)
    if result.returncode == 0: 
        outputFile = open(output, "r")

        lines = outputFile.readlines()
        if lines:  # Verificar que el archivo no esté vacío
            last_line = lines[-1]
            print("Última línea:", last_line)
            tiempo_final = last_line.split(",")[3].strip("\n")
            return tiempo_final

def testCL(n,m,g,l,t,shared):
    output = "output2.txt"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.path.join(current_dir,"build\src\cl")
    os.chdir(current_dir)
    exe_path = os.path.join(current_dir, "Tarea2CL.exe")
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
    
def testCUDA(n,m,g,l,t):
    output = "output.txt"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exe_path = os.path.join(current_dir, "build\src\cl\Tarea2CL.exe")
    outputfile_aux = os.path.join(current_dir,"build\src\cl",output)
    args = [str(n), str(m), str(g), str(l), str(t), output]
    result = subprocess.run([exe_path] + args, capture_output=True, text=True)
    outputFile = open(outputfile_aux, "r")


    lines = outputFile.readlines()
    if lines:  # Verificar que el archivo no esté vacío
        last_line = lines[-1]
        print("Última línea:", last_line)
        tiempo_final = last_line.strip(",")[4]
        return tiempo_final

valores = [50,100,500,1000]
clRes = []
cpuRes = []
for val in valores:
    #aviso de que opencl explota cuando el blocksize no es divisor del numero total, estoy buscando solucion
    res = testCL(val,val,val,2,3,0)
    clRes.append(float(res))

    res = testCPU(val,val,3)
    cpuRes.append(float(res))

print(clRes)
print(cpuRes)


plt.plot(valores, clRes, label='y = x^2', marker='o')
plt.plot(valores, cpuRes,label='y = x^3', marker='x')
# Añadir título y etiquetas
plt.title('Comparacion metodos base')
plt.xlabel('Tiempo')
plt.ylabel('Tamaño de grilla (NxN)')

# Añadir una leyenda
plt.legend()

# Mostrar el gráfico
plt.show()
