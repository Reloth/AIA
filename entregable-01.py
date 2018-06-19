# Ampliación de Inteligencia Artificial
# Problemas de Satisfacción de Restricciones
# Dpto. de C. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================

# -------------------------------------------------------------------- 
# Los siguientes apartados se proponen como ejercicio de programación que
# contará para la evaluación de la asignatura. Este entregable supone 0.75
# puntos de la nota total de la asignatura.  Se deberá entregar a través de la
# página de la asignatura, en el formulario a tal efecto que estará disponible
# junto a la ficha de alumno.


# IMPORTANTE: No cambiar el nombre ni a este archivo ni a las funciones que se
# piden. Si se entregan con un nombre distinto, el entregable no será
# evaluado.
# --------------------------------------------------------------------



## ###################################################################
## HONESTIDAD ACADÉMICA Y COPIAS: la realización de los ejercicios es
## un trabajo personal, por lo que deben completarse por cada
## estudiante de manera individual.  La discusión y el intercambio de
## información de carácter general con los compañeros se permite (e
## incluso se recomienda), pero NO AL NIVEL DE CÓDIGO. Igualmente el
## remitir código de terceros, obtenido a través de la red o cualquier
## otro medio, se considerará plagio.

## Cualquier plagio o compartición de código que se detecte
## significará automáticamente la calificación de CERO EN LA
## ASIGNATURA para TODOS los alumnos involucrados. Por tanto a estos
## alumnos NO se les conservará, para futuras convocatorias, ninguna
## nota que hubiesen obtenido hasta el momento. Independientemente de
## OTRAS ACCIONES DE CARÁCTER DISCIPLINARIO que se pudieran tomar.
## ###################################################################



# -----------------------------------------------------------------------

# Lo que sigue es el código visto en la práctica 01 de clase, incluyendo:
# - Clase PSR para representar problemas de satisfacción de restricciones. 
# - Representación del problema de las n-reinas
# - Implementación del algoritmo AC-3

import random,copy

class PSR:
    """Clase que describe un problema de satisfacción de
    restricciones, con los siguientes atributos:
       variables     Lista de las variables del problema
       dominios      Diccionario que asigna a cada variable su dominio
                     (una lista con los valores posibles)
       restricciones Diccionario que asocia a cada tupla de variables
                     involucrada en una restricción, una función que,
                     dados valores de los dominios de esas variables,
                     determina si cumplen o no la restricción.
                     IMPORTANTE: Supondremos que para cada combinación
                     de variables hay a lo sumo una restricción (por
                     ejemplo, si hubiera dos restricciones binarias
                     sobre el mismo par de variables, consideraríamos
                     la conjunción de ambas). 
                     También supondremos que todas las restricciones
                     son binarias
        vecinos      Diccionario que representa el grafo del PSR,
                     asociando a cada variable, una lista de las
                     variables con las que comparte restricción.

    El constructor recibe los valores de los atributos dominios y
    restricciones; los otros dos atributos serán calculados al
    construir la instancia."""

    def __init__(self, dominios, restricciones):
        """Constructor de PSRs."""

        self.dominios = dominios
        self.restricciones = restricciones
        self.variables = list(dominios.keys())

        vecinos = {v: [] for v in self.variables}
        for v1, v2 in restricciones:
            vecinos[v1].append(v2)
            vecinos[v2].append(v1)
        self.vecinos = vecinos


def n_reinas(n):
    """Devuelve el PSR correspondiente al problema de las n-reinas"""

    def n_reinas_restriccion(x,y):
        return lambda u,v: (abs(x-y) != abs(u-v) and u != v)

    doms = {x:list(range(1,n+1)) for x in range(1,n+1)}
    restrs = dict()
    for x in range(1,n):
        for y in range(x+1,n+1):
            restrs[(x,y)] = n_reinas_restriccion(x,y)
    return PSR(doms,restrs)

def dibuja_tablero_n_reinas(asig):
    def cadena_fila(i,asig):
        cadena="|"
        for j in range (1,n+1):
            if asig[i]==j:
                cadena += "X|"
            else:
                cadena += " |"
        return cadena
    n=len(asig)
    print("+"+"-"*(2*n-1)+"+")
    for i in range(1,n):
        print(cadena_fila(i,asig))
        print("|"+"-"*(2*n-1)+"|")
    print(cadena_fila(n,asig))
    print("+"+"-"*(2*n-1)+"+")



def arcos (psr):
    return {(x, y) for x in psr.variables for y in psr.vecinos[x]}

def restriccion_arco(psr, x, y):
    if (x, y) in psr.restricciones:
        return psr.restricciones[(x, y)]
    else:
        return lambda vx, vy: psr.restricciones[(y, x)](vy, vx)

def AC3(psr, doms):
    """Procedimiento para hacer arco consistente un problema de
    satisfacción de restricciones dado. Es destructivo respecto al
    atributo dominios"""

    cola = arcos(psr)
    while cola:
        (x, y) = cola.pop()
        func = restriccion_arco(psr,x, y)
        dom_previo_x = doms[x]
        mod_dom_x = False
        dom_nuevo_x = []
        for vx in dom_previo_x:
            if any(func(vx, vy) for vy in doms[y]):
                dom_nuevo_x.append(vx)
            else:
                mod_dom_x = True
        if mod_dom_x:
            doms[x] = dom_nuevo_x
            cola.update((z, x) for z in psr.vecinos[x] if z != y)
    return doms



#------------------------------------------------------------------------------
# Ejercicio 1
#------------------------------------------------------------------------------

# Definir en python una función psr_busqueda_AC3 que implemente el algoritmo
# de búsqueda AC-3 que se describe en las diapositivas 47 y 48 del tema 1.

# Aplicarlo a resolver distintas instancias del problema de las n reinas.
    
# NOTA: Elegir una estrategia particular tanto para gestionar la búsqueda como 
# para partir los dominios. Se valorará la eficiencia del procedimiento.  

# Ejemplos (no necesariamente hay devolver estas soluciones):

# >>> dibuja_tablero_n_reinas(psr_busqueda_AC3(n_reinas(4)))
# +-------+
# | | |X| |
# |-------|
# |X| | | |
# |-------|
# | | | |X|
# |-------|
# | |X| | |
# +-------+

# >>> dibuja_tablero_n_reinas(psr_busqueda_AC3(n_reinas(6)))
# +-----------+
# | | | | |X| |
# |-----------|
# | | |X| | | |
# |-----------|
# |X| | | | | |
# |-----------|
# | | | | | |X|
# |-----------|
# | | | |X| | |
# |-----------|
# | |X| | | | |
# +-----------+

# >>> dibuja_tablero_n_reinas(psr_busqueda_AC3(n_reinas(8)))
# +---------------+
# | | | | | | | |X|
# |---------------|
# | | | |X| | | | |
# |---------------|
# |X| | | | | | | |
# |---------------|
# | | |X| | | | | |
# |---------------|
# | | | | | |X| | |
# |---------------|
# | |X| | | | | | |
# |---------------|
# | | | | | | |X| |
# |---------------|
# | | | | |X| | | |
# +---------------+

# >>> dibuja_tablero_n_reinas(psr_busqueda_AC3(n_reinas(14)))
# +---------------------------+
# | | | | | | | | | | | | | |X|
# |---------------------------|
# | | | | | | | | | | | |X| | |
# |---------------------------|
# | | | | | | | | | |X| | | | |
# |---------------------------|
# | | | | | | | |X| | | | | | |
# |---------------------------|
# | | |X| | | | | | | | | | | |
# |---------------------------|
# | | | | |X| | | | | | | | | |
# |---------------------------|
# | |X| | | | | | | | | | | | |
# |---------------------------|
# | | | | | | | | | | |X| | | |
# |---------------------------|
# |X| | | | | | | | | | | | | |
# |---------------------------|
# | | | | | |X| | | | | | | | |
# |---------------------------|
# | | | | | | | | | | | | |X| |
# |---------------------------|
# | | | | | | | | |X| | | | | |
# |---------------------------|
# | | | | | | |X| | | | | | | |
# |---------------------------|
# | | | |X| | | | | | | | | | |
# +---------------------------+

def psr_busqueda_AC3(psr):
    abiertos = [psr.dominios]
    while abiertos:
        actual = abiertos[0]
        abiertos = abiertos[1:]
        AC3(psr,actual)
        if not any([actual[x]==[] for x in actual]):
            if all([len(actual[x])==1 for x in actual]):
                return {x:actual[x][0] for x in actual}
            else:
                tamMin = min([len(actual[x]) for x in actual if len(actual[x])>1])
                mins = [x for x in actual if len(actual[x])==tamMin]
                div = random.choice(mins)
                suc1 = copy.deepcopy(actual)
                suc2 = copy.deepcopy(actual)
                for x in actual:
                    if x == div:
                        randItem = random.choice(actual[x])
                        suc1[x].remove(randItem)
                        suc2[x] = [randItem]
                abiertos.append(suc2)
                abiertos.append(suc1)
    return "Fallo"

# dibuja_tablero_n_reinas(psr_busqueda_AC3(n_reinas(4)))



#------------------------------------------------------------------------------
# Ejercicio 2
#------------------------------------------------------------------------------


# Representar como problema de satisfacción de restricciones (psr) el
# siguiente problema.

# Supongamos que en la ETSII tenemos que programar los exámenes de una serie
# de asignaturas y para ello disponemos de un periodo de N días
# consecutivos. La única restricción que tenemos es que cada alumno tiene que
# tener UN MÍNIMO DE DOS DÍAS de separación entre los exámenes que tenga
# programados.  Se trata de encontrar un calendario de exámenes. Por
# simplificar, ignorareremos los fines de semana, y nombraremos los días como
# 1,2,...,N.

# En concreto, se pide definir una función calendario_examenes_psr que
# recibiendo como entrada la lista de alumnos matriculados en cada asignatura
# (en forma de diccionario) y un número N de días, construya un psr tal que la
# solución de dicho psr sea un calendario válido.

# Por ejemplo, las siguientes 13 asignaturas con sus correspondientes listas
# de alumnos matriculados:

ejemplo_asig1={"AIA":["Jorge","Juan","Beatriz"],
              "IA":["Fátima","Luis","Carlos","Jesús","Carmen"],
              "MASI":["Juan","Emilio","Jorge","Beatriz"],
              "PD":["Jorge","Beatriz","Fátima"],
              "MATI":["Luis","Carmen"],
              "SIE":["Carlos","Jesús"],
              "SI":["Esther","Laura","Elena","Oswaldo"],
              "SOS":["Miguel","Eva","Alberto","Manuel"],
              "CIMSI":["Esther","Laura","Elena","Rafael"],
              "PGPI":["Emilio","Esther","Laura","Elena"],
              "TAI":["Miguel","Margarita","Alberto","Manuel"],
              "ASD":["Miguel","Eva","Margarita","Trinidad","Alberto","Manuel"],
              "GEE":["Trinidad","Nicolás","Roberto","Julia","Rosario"]}

# Supongamos que queremos ver si se pueden programar en seis dias. Para ello,
# primero generaríamos (con la función que se pide) el correspondiente psr:

# >>> psr_ejemplo_asig1=calendario_examenes_psr(ejemplo_asig1,6)

# Y ahora llamaríamos a algún algoritmo de resolución de psr. Por ejemplo, el
# de búsqueda AC3 que hemos definido arriba:

# >>> psr_busqueda_AC3(psr_ejemplo_asig1)
# {'AIA': 1,'ASD': 3,'CIMSI': 3,'GEE': 1,'IA': 1,'MASI': 3,
#  'MATI': 3,'PD': 5,'PGPI': 1,'SI': 5,'SIE': 3,'SOS': 5,'TAI': 1}

# Como se observa, la salida consiste en un diccionario que asigna a cada
# asignatura el día en el que se realiza su examen (no necesariamente hay que 
# devolver esta solución concreta). 

def calendario_examenes_psr(matriculas, dias):

    """Devuelve el PSR correspondiente al problema del calendario de examenes"""

    def calendario_examenes_restriccion():
        return lambda u,v: (abs(u-v) >= 2)

    doms = {x:list(range(1,dias+1)) for x in [y for y in matriculas]}
    restrs = dict()
    for x in matriculas:
        for y in matriculas:
            if x!=y and len(set(matriculas[x]) & set(matriculas[y]))!=0:
                restrs[(x,y)] = calendario_examenes_restriccion()
    return PSR(doms,restrs)

# psr_ejemplo_asig1=calendario_examenes_psr(ejemplo_asig1,5)
# print(psr_busqueda_AC3(psr_ejemplo_asig1))



#------------------------------------------------------------------------------
# Ejercicio 3
#------------------------------------------------------------------------------

# Dados dos grafos no dirigidos G1=(V1,E1) y G2=(V2,G2), un homorfismo de G1
# en G2 es una función f que a cada vértice v en V1 le asigna un vértice f(v)
# en V2, de manera que si entre dos vértices v,w de V1 hay un arco en E1,
# entonces entre f(v) y f(w) hay un arco en E2. 

# Por ejemplo, dados los dos siguientes grafos G1 y G2:

#   B---C                    
#  /     \                        X   
# A       D                      / \  
#  \     /                      /   \ 
#   \   /                      Y-----Z
#     E

# Un homomorfismo entre G1 y G2 podría ser:
# A -> X, B -> Y, C -> Z D -> X, E -> Y  

# Supondremos que los grafos se representan mediante un par: su lista de
# vértices y su lista de arcos. Cada arco se representa mediante el conjunto
# del par de vértices que une el arco.


# En el caso de los grafos anteriores, los representamos así:

G1_ej1=(["A","B","C","D","E"],
        [{"A","B"}, {"B","C"}, {"C","D"}, {"D","E"}, {"A","E"}])

G2_ej1=(["X","Y","Z"],
        [{"X","Y"},{"Y","Z"},{"Z","X"}])



# Se pide representar como problema de satisfacción de restricciones (psr) el
# problema de encontrar, si existe, un homomorfismo entre dos grafos dados. En
# concreto, definir una función homomorfismo_grafo_psr(g1,g2) que recibiendo
# como entrada dos grafos no dirigidos g1 y g2, construya un psr tal que la
# solución a dicho psr se pueda interpretar como un homomorfismo entre ambos
# grafos.

# Por ejemplo:

# >>> psr_grafos_ej1=homomorfismo_grafo_psr(G1_ej1,G2_ej1)
# >>> psr_busqueda_AC3(psr_grafos_ej1)
# {'A': 'X', 'B': 'Y', 'C': 'X', 'D': 'Y', 'E': 'Z'}
# Otro ejemplo:

G1_ej2=([1,2,3,4,5,6,7,8,9,10],
        [{1,2},{1,5},{1,6},{2,3},{2,7},{3,4},{3,8},{4,5},{4,9},{5,10},
         {6,8},{6,9},{7,9},{7,10}])


G2_ej2=(["A","R","V"],[{"A","R"},{"A","V"},{"R","V"}])


# >>> psr_grafos_ej2=homomorfismo_grafo_psr(G1_ej2,G2_ej2)
# >>> psr_busqueda_AC3(psr_grafos_ej2)
# {1: 'R', 2: 'A', 3: 'V', 4: 'A', 5: 'V', 6: 'A', 7: 'V',
#  8: 'R', 9: 'R', 10: 'A'}

def homomorfismo_grafo_psr(g1,g2):

    """Devuelve el PSR correspondiente al problema de los grafos"""

    def grafos_restriccion(g):
        return lambda u,v: (({u,v} in g) and u != v)

    doms = {x:g2[0] for x in g1[0]}
    restrs = dict()
    for x in g1[0]:
        for y in g1[0]:
            if {x,y} in g1[1]:
                restrs[(x,y)] = grafos_restriccion(g2[1])
    return PSR(doms,restrs)

# print(list(G2_ej2[1][0])[0])

# psr_grafos_ej2=homomorfismo_grafo_psr(G1_ej2,G2_ej2)
# print(psr_busqueda_AC3(psr_grafos_ej2))

# psr_grafos_ej1=homomorfismo_grafo_psr(G1_ej1,G2_ej1)
# print(psr_busqueda_AC3(psr_grafos_ej1))


#------------------------------------------------------------------------------
# Ejercicio 4
#------------------------------------------------------------------------------


# Definir una función calendario_optimo que, recibiendo como
# entrada la lista de alumnos matriculados en cada asignatura (en forma de
# diccionario), devuelve el periodo de tiempo más corto (menor número de días
# consecutivos) en los que es posible planificar un calendario de exámenes
# para esas asignaturas, junto con el propio calendario para ese periodo de
# tiempo. 

# Por ejemplo:

# >>> calendario_optimo(ejemplo_asig1)
# (5, {'AIA': 5,'ASD': 1,'CIMSI': 5,'GEE': 3,'IA': 1,'MASI': 1,'MATI': 3,
#     'PD': 3,'PGPI': 3,'SI': 1,'SIE': 3,'SOS': 5,'TAI': 3})    

def calendario_optimo(matriculas):
    dias = 1
    while True:
        calendario = psr_busqueda_AC3(calendario_examenes_psr(matriculas,dias))
        if calendario != "Fallo":
            return (dias,calendario)
        dias += 1

# print(calendario_optimo(ejemplo_asig1))


#------------------------------------------------------------------------------
# Ejercicio 5
#------------------------------------------------------------------------------


# En realidad, el problema del calendario de exámenes es un caso particular
# del problema del homomorfismo de grafos. En este ejercicio se pide definir
# una función grafos_calendario que recibiendo como entrada la lista de
# alumnos matriculados en cada asignatura  (en forma de diccionario) y un
# número N de días (como en el Ejercicio 7), devuelva un par de grafos tales
# que el problema de encontrar un homomorfismo entre ambos es equivalente al
# problema de encontrar un calendario de exámenes para las asignaturas, en N
# días. 

# Una vez definida la función grafos_calendario, una manera alternativa de
# resolver el problema del calendario, sería resolver el problema del
# homomorfismo en los grafos asociados. 

# Por ejemplo:

# >>> ga1,ga2=grafos_calendario(ejemplo_asig1,6)
# >>> psr_ga=homomorfismo_grafo_psr(ga1,ga2)
# >>> psr_busqueda_AC3(psr_ga)
# {'AIA': 1, 'ASD': 3, 'CIMSI': 5, 'GEE': 1, 'IA': 1, 'MASI': 3,
# 'MATI': 3, 'PD': 5, 'PGPI': 1, 'SI': 3, 'SIE': 3, 'SOS': 5, 'TAI': 1}   


def grafos_calendario(matriculas, dias):
    asgis = matriculas.keys()
    grafo_asignaturas = (list(asgis),[])
    lista_dias = range(1,dias+1)
    grafo_dias = (list(lista_dias),[])

    matriculasY = copy.deepcopy(matriculas)
    for x in matriculas:
        for y in matriculasY:
            if x!=y and len(set(matriculas[x]) & set(matriculas[y]))!=0:
                grafo_asignaturas[1].append({x,y})
        del matriculasY[x]
    
    for a in range(1,dias+1):
        for b in range(a+1,dias+1):
            if abs(a-b)>=2:
                grafo_dias[1].append({a,b})

    return grafo_asignaturas,grafo_dias



# p1,p2 = grafos_calendario(ejemplo_asig1,6)
# print(p1,p2)

# ga1,ga2=grafos_calendario(ejemplo_asig1,5)
# psr_ga=homomorfismo_grafo_psr(ga1,ga2)
# print(psr_busqueda_AC3(psr_ga))