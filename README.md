Este proyecto corresponde a una implementacion del juego de la vida de Conway, utilizandose fronteras cíclicas (si la frontera supera el tamaño de la grilla, se utilizara la inicial).
Para el codigo se utilizo el IDE de CLion junto a su compilador, con esto deberia poder compilarse el codigo utilizando la siguiende linea de comandos en la carpeta base:

cmake -S . -B build  
cmake --build build -j 10

Luego para ejecutar el codigo se tiene que:
    Para ejecutar la version secuancial/CPU dirigirse a build/src y ejecutar Tarea2CPU con los siguientes parametros:
        N: dimension 1 sobre la grilla a ejecutar
        M: dimension 2 sobre la grilla a ejecutar
        T: numero de epocas a ejecutar el algoritmo
        Output: nombre del archivo donde se guardara el resultado con todas las epocas
    luego se ejecutaria .\Tarea2CPU N M T Output

    Para ejecutar la version CUDA dirigirse a build/src/cuda y ejecutar Tarea2CUDA con los siguientes parametros:
        N: dimension 1 sobre la grilla a ejecutar
        M: dimension 2 sobre la grilla a ejecutar
        T: numero de epocas a ejecutar el algoritmo
        g: numero de tamaño de bloque (gxg)
        shared: 1 si es que quiere usarse la version con memoria compartida, 0 sino
        Output: nombre del archivo donde se guardara el resultado con todas las epocas
    luego se ejecutaria .\Tarea2CUDA n m t g shared output

    Para ejecutar la version OpenCL dirigirse a build/src/cl y ejecutar Tarea2CL con los siguientes parametros:
        N: dimension 1 sobre la grilla a ejecutar
        M: dimension 2 sobre la grilla a ejecutar
        T: numero de epocas a ejecutar el algoritmo
        g: numero de tamaño de work-group (gxg)
        shared: 1 si es que quiere usarse la version con memoria compartida, 0 sino
        Output: nombre del archivo donde se guardara el resultado con todas las epocas
        Input (opcional): nombre de una matriz a leer inicialmente predefinida con formato 
                        por fila, separado por coma ejemplo 2x3 1 1 1 = 1,1,1,0,0,0,1,0,1
                                                                0 0 0
                                                                1 0 1 
    luego se ejecutaria .\Tarea2CUDA n m g T shared output input (notar que g y t estan intercambiados en comparacion a CUDA)

el archivo Testing.py corresponde al archivo donde estan las funciones para hacer los graficos del informe.

## Makefile
Hay un Makefile para trabajar más fácil con los siguientes comandos:
- all: Construye los ejecutables para CUDA, OpenCL y CPU.
- init: Inicializa el directorio de `build` utilizando CMake.
- cuda: Construye el ejecutable para CUDA.
- cl: Construye el ejecutable para OpenCL.
- cpu: Construye el ejecutable para CPU.
- test: Ejecuta las pruebas utilizando CTest.
- clean: Elimina los artefactos de construcción y los directorios de pruebas.
- watch: Monitorea los archivos fuente en busca de cambios y desencadena una construcción cuando se detecta un cambio.
