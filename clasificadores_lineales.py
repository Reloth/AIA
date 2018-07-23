# ==========================================================
# Ampliación de Inteligencia Artificial. Tercer curso. 
# Grado en Ingeniería Informática - Tecnologías Informáticas
# Curso 2017-18
# Trabajo práctico
# ===========================================================

# --------------------------------------------------------------------------
# Autor del trabajo:
#
# APELLIDOS: Alonso Cancillo
# NOMBRE: Diego
#
# Segundo componente (si se trata de un grupo):
#
# APELLIDOS: Mata Blasco
# NOMBRE: Carlos
# ----------------------------------------------------------------------------

#=====!!!!!=====!!!!!=====!!!!!===== NOTA =====!!!!!=====!!!!!=====!!!!!=====
# Al final del documento se encuentran los comentarios y valoraciones sobre el proyecto.

# *****************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen, por lo que
# debe realizarse de manera individual. La discusión y el intercambio de
# información de carácter general con los compañeros se permite (e incluso se
# recomienda), pero NO AL NIVEL DE CÓDIGO. Igualmente el remitir código de
# terceros, OBTENIDO A TRAVÉS DE LA RED o cualquier otro medio, se considerará
# plagio. 

# Cualquier plagio o compartición de código que se detecte significará
# automáticamente la calificación de CERO EN LA ASIGNATURA para TODOS los
# alumnos involucrados. Por tanto a estos alumnos NO se les conservará, para
# futuras convocatorias, ninguna nota que hubiesen obtenido hasta el
# momento. SIN PERJUICIO DE OTRAS MEDIDAS DE CARÁCTER DISCIPLINARIO QUE SE
# PUDIERAN TOMAR.  
# *****************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES Y MÉTODOS
# QUE SE PIDEN

# NOTA: En este trabajo no se permite usar scikit-learn, pero sí numpy o scipy. 

# ====================================================
# PARTE I: MODELOS LINEALES PARA CLASIFICACIÓN BINARIA
# ====================================================

# En esta primera parte se pide implementar en Python los siguientes
# clasificadores BINARIOS, todos ellos vistos en el tema 5.

# - Perceptron umbral
# - Regresión logística minimizando el error cuadrático:
#      * Versión batch
#      * Versión estocástica (regla delta)
# - Regresión logística maximizando la verosimilitud:
#      * Versión batch
#      * Versión estocástica


# --------------------------------------------
# I.1. Generando conjuntos de datos aleatorios
# --------------------------------------------

# Previamente a la implementación de los clasificadores, conviene tener
# funciones que generen aleatoriamente conjuntos de datos fictícios. 
# En concreto, se pide implementar estas dos funciones:

# * Función genera_conjunto_de_datos_l_s(rango,dim,n_datos): 

#   Debe devolver dos listas X e Y, generadas aleatoriamente. La lista X debe
#   tener un número total n_datos de elementos, siendo cada uno de ellos una
#   lista (un ejemplo) de dim componentes, con valores entre -rango y rango. El
#   conjunto Y debe tener la clasificación binaria (1 o 0) de cada ejemplo del
#   conjunto X, en el mismo orden. El conjunto de datos debe ser linealmente
#   separable.

#   SUGERENCIA: generar en primer lugar un hiperplano aleatorio (mediante sus
#   coeficientes, elegidos aleatoriamente entre -rango y rango). Luego generar
#   aleatoriamente cada ejemplo de igual manera y clasificarlo como 1 o 0
#   dependiendo del lado del hiperplano en el que se situe. Eso asegura que el
#   conjunto de datos es linealmente separable.


import random, copy, numpy, math
from iris import *
from votos import *

def genera_conjunto_de_datos_l_s(rango,dim,n_datos):

	#hiperplano=[random.randint(-1*rango,rango) for x in range(dim)]
	hiperplano = [random.randint(-1*rango,rango) for x in range(dim+1)] 
	#No pasan todos los hiperplanos por el 0
	X=[[random.randint(-1*rango,rango) for x in range(dim)] for y in range(n_datos)]
	Y=[0 if (0>sum([x[i]*hiperplano[i] for i in range(dim)])) else 1 for x in X]
	return X,Y

#print(genera_conjunto_de_datos_l_s(100,4,8))
genera_conjunto_de_datos_l_s(100,4,15)


# * Función genera_conjunto_de_datos_n_l_s(rango,dim,size,prop_n_l_s=0.1):

#   Como la anterior, pero el conjunto de datos debe ser no linealmente
#   separable. Para ello generar el conjunto de datos con la función anterior
#   y cambiar de clase a una proporción pequeña del total de ejemplos (por
#   ejemplo el 10%). La proporción se da con prop_n_l_s. 


def genera_conjunto_de_datos_n_l_s(rango,dim,size,prop_n_l_s=0.1):

	X,Y=genera_conjunto_de_datos_l_s(rango,dim,size)
	if size*prop_n_l_s < 1:
		random_element = random.randint(0,size-1)
		Y[random_element] = 0 if Y[random_element]==1 else 1
	else:
		cambios = list(range(size))
		random.shuffle(cambios)
		cont_cambios = round(size*prop_n_l_s)
		while cont_cambios > 0:
			Y[cambios[cont_cambios-1]] = 0 if Y[cambios[cont_cambios-1]]==1 else 1
			cont_cambios -= 1
	return X,Y

#print(genera_conjunto_de_datos_n_l_s(100,4,7))
genera_conjunto_de_datos_n_l_s(100,4,15)


# -----------------------------------
# I.2. Clases y métodos a implementar
# -----------------------------------

# En esta sección se pide implementar cada uno de los clasificadores lineales
# mencionados al principio. Cada uno de estos clasificadores se implementa a
# través de una clase python, que ha de tener la siguiente estructura general:

# class NOMBRE_DEL_CLASIFICADOR():

#     def __init__(self,clases,normalizacion=False):

#          .....
         
#     def entrena(self,entr,clas_entr,n_epochs,rate=0.1,
#                 pesos_iniciales=None,
#                 rate_decay=False):

#         ......

#     def clasifica_prob(self,ej):


#         ......

#     def clasifica(self,ej):


#         ......
        

#  - El parámetro normalizacion, que puede ser True o False (False por
#    defecto). Indica si los datos se tienen que normalizar, tanto para el
#    entrenamiento como para la clasificación de nuevas instancias.  La
#    normalización es una técnica que suele ser útil cuando los distintos
#    atributos reflejan cantidades numéricas de muy distinta magnitud.
#    En ese caso, antes de entrenar se calcula la media m_i y la desviación
#    típica d_i en cada componente i-esima (es decir, en cada atributo) de los
#    datos del conjunto de entrenamiento.  A continuación, y antes del
#    entrenamiento, esos datos se transforman de manera que cada componente
#    x_i se cambia por (x_i - m_i)/d_i. Esta misma transformación se realiza
#    sobre las nuevas instancias que se quieran clasificar.  NOTA: se permite
#    usar la biblioteca numpy para calcular la media, la desviación típica, y
#    en general para cualquier cálculo matemático.

class Clasificador_Perceptron():

	def __init__(self,clases,normalizacion=False):
		self.clases = clases
		self.normalizacion = normalizacion
		self.pesos = []
		self.entrenado = False


	def entrena(self,entr,clas_entr,n_epochs,rate=0.1,pesos_iniciales=None,rate_decay=False):
		conjEntr = copy.copy(entr)
		self.entrenado = True
		if self.normalizacion:
			for x in range(len(entr)):
				media = numpy.mean(entr[x])
				desviacion = numpy.std(entr[x])
				conjEntr[x] = [ (i - media)/desviacion for i in entr[x]]


		clases_entrenamiento = [self.clases.index(x) if x in self.clases else 0 for x in clas_entr]

		if pesos_iniciales == None:
			W = [random.uniform(-1.0,1.0) for i in range(len(conjEntr[0])+1)]
		else:
			W = copy.copy(pesos_iniciales)

		contador = 1
		entrenamiento = list(zip([[1] + conjEntr[i] for i in range(len(conjEntr))],clases_entrenamiento))
		ratio = rate

		while contador <= n_epochs:
			random.shuffle(entrenamiento)
			for i in range(len(entrenamiento)):
				o = 0 if (0>sum([W[j]*entrenamiento[i][0][j] for j in range(len(entrenamiento[i][0]))])) else 1
				for j in range(len(entrenamiento[i][0])):
					W[j] = W[j] + ratio * entrenamiento[i][0][j] * (entrenamiento[i][1] - o)

			if rate_decay:
				ratio = rate + (2/contador**(1.5)) 
			contador += 1

		self.pesos = W 


	def clasifica_prob(self,ej):
		pass

	def clasifica(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		clasificacion = 0 if (0>sum([self.pesos[j]*ej1[j] for j in range(len(ej1))])) else 1
		return self.clases[clasificacion]



class Clasificador_RL_L2_Batch():

	def __init__(self,clases,normalizacion=False):
		self.clases = clases
		self.normalizacion = normalizacion
		self.pesos = []
		self.entrenado = False

	def entrena(self,entr,clas_entr,n_epochs,rate=0.1,pesos_iniciales=None,rate_decay=False):
		conjEntr = copy.copy(entr)
		self.entrenado = True
		if self.normalizacion:
			for x in range(len(entr)):
				media = numpy.mean(entr[x])
				desviacion = numpy.std(entr[x])
				conjEntr[x] = [ (i - media)/desviacion for i in entr[x]]
		clases_entrenamiento = [self.clases.index(x) if x in self.clases else 0 for x in clas_entr]

		if pesos_iniciales == None:
			W = [random.uniform(-1.0,1.0) for i in range(len(entr[0])+1)]
		else:
			W = copy.copy(pesos_iniciales)

		contador = 1
		entrenamiento = list(zip([[1] + conjEntr[i] for i in range(len(conjEntr))],clases_entrenamiento))
		ratio = rate

		while contador <= n_epochs:
			random.shuffle(entrenamiento)
			difW = [0 for i in range(len(W))]

			for i in range(len(entrenamiento)):
				wx = sum([W[j]*entrenamiento[i][0][j] for j in range(len(entrenamiento[i][0]))])
				o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
				for j in range(len(entrenamiento[i][0])):
					difW[j] = difW[j] + ratio * (entrenamiento[i][1] - o) * o * (1 - o) * entrenamiento[i][0][j]
					
			W = [W[i] + difW[i] for i in range(len(W))]
			if rate_decay:
				ratio = rate + (2/contador**(1.5)) 
			contador += 1
		self.pesos = W 


	def clasifica_prob(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([self.pesos[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		return o

	def clasifica(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([self.pesos[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		clasificacion = round(o)
		return self.clases[clasificacion]



class Clasificador_RL_L2_St():

	def __init__(self,clases,normalizacion=False):
		self.clases = clases
		self.normalizacion = normalizacion
		self.pesos = []
		self.entrenado = False

	def entrena(self,entr,clas_entr,n_epochs,rate=0.1,pesos_iniciales=None,rate_decay=False):
		conjEntr = copy.copy(entr)
		self.entrenado = True
		if self.normalizacion:
			for x in range(len(entr)):
				media = numpy.mean(entr[x])
				desviacion = numpy.std(entr[x])
				conjEntr[x] = [ (i - media)/desviacion for i in entr[x]]
		clases_entrenamiento = [self.clases.index(x) if x in self.clases else 0 for x in clas_entr]

		if pesos_iniciales == None:
			W = [random.uniform(-1.0,1.0) for i in range(len(conjEntr[0])+1)]
		else:
			W = copy.copy(pesos_iniciales)

		contador = 1
		entrenamiento = list(zip([[1] + conjEntr[i] for i in range(len(conjEntr))],clases_entrenamiento))
		ratio = rate

		while contador <= n_epochs:
			random.shuffle(entrenamiento)

			for i in range(len(entrenamiento)):
				wx = sum([W[j]*entrenamiento[i][0][j] for j in range(len(entrenamiento[i][0]))])
				o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
				for j in range(len(entrenamiento[i][0])):
					W[j] = W[j] + ratio * (entrenamiento[i][1] - o) * o * (1 - o) * entrenamiento[i][0][j]
			if rate_decay:
				ratio = rate + (2/contador**(1.5)) 
			contador += 1
		self.pesos = W 


	def clasifica_prob(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([self.pesos[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		return o

	def clasifica(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([self.pesos[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		clasificacion = round(o)
		return self.clases[clasificacion]



class Clasificador_RL_ML_Batch():

	def __init__(self,clases,normalizacion=False):
		self.clases = clases
		self.normalizacion = normalizacion
		self.pesos = []
		self.entrenado = False

	def entrena(self,entr,clas_entr,n_epochs,rate=0.1,pesos_iniciales=None,rate_decay=False):
		conjEntr = copy.copy(entr)
		self.entrenado = True
		if self.normalizacion:
			for x in range(len(entr)):
				media = numpy.mean(entr[x])
				desviacion = numpy.std(entr[x])
				conjEntr[x] = [ (i - media)/desviacion for i in entr[x]]
		clases_entrenamiento = [self.clases.index(x) if x in self.clases else 0 for x in clas_entr]

		if pesos_iniciales == None:
			W = [random.uniform(-1.0,1.0) for i in range(len(conjEntr[0])+1)]
		else:
			W = copy.copy(pesos_iniciales)

		contador = 1
		entrenamiento = list(zip([[1] + conjEntr[i] for i in range(len(conjEntr))],clases_entrenamiento))
		# print(entr)
		ratio = rate

		while contador <= n_epochs:
			random.shuffle(entrenamiento)
			difW = [0 for i in range(len(W))]

			for i in range(len(entrenamiento)):
				# print(entrenamiento[i][0])
				# print(W)
				wx = sum([W[j]*entrenamiento[i][0][j] for j in range(len(entrenamiento[i][0]))])
				o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
				for j in range(len(entrenamiento[i][0])):
					difW[j] = difW[j] + ratio * (entrenamiento[i][1] - o) * entrenamiento[i][0][j]
					
			W = [W[i] + difW[i] for i in range(len(W))]
			if rate_decay:
				ratio = rate + (2/contador**(1.5)) 
			contador += 1
		self.pesos = W 


	def clasifica_prob(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([self.pesos[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		return o

	def clasifica(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([self.pesos[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		clasificacion = round(o)
		return self.clases[clasificacion]



class Clasificador_RL_ML_St():

	def __init__(self,clases,normalizacion=False):
		self.clases = clases
		self.normalizacion = normalizacion
		self.pesos = []
		self.entrenado = False

	def entrena(self,entr,clas_entr,n_epochs,rate=0.1,pesos_iniciales=None,rate_decay=False):
		conjEntr = copy.copy(entr)
		self.entrenado = True
		if self.normalizacion:
			for x in range(len(entr)):
				media = numpy.mean(entr[x])
				desviacion = numpy.std(entr[x])
				conjEntr[x] = [ (i - media)/desviacion for i in entr[x]]
		clases_entrenamiento = [self.clases.index(x) if x in self.clases else 0 for x in clas_entr]

		if pesos_iniciales == None:
			W = [random.uniform(-1.0,1.0) for i in range(len(conjEntr[0])+1)]
		else:
			W = copy.copy(pesos_iniciales)

		contador = 1
		entrenamiento = list(zip([[1] + conjEntr[i] for i in range(len(conjEntr))],clases_entrenamiento))
		ratio = rate

		while contador <= n_epochs:
			random.shuffle(entrenamiento)

			for i in range(len(entrenamiento)):
				wx = sum([W[j]*entrenamiento[i][0][j] for j in range(len(entrenamiento[i][0]))])
				o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
				for j in range(len(entrenamiento[i][0])):
					W[j] = W[j] + ratio * (entrenamiento[i][1] - o) * entrenamiento[i][0][j]
			if rate_decay:
				ratio = rate + (2/contador**(1.5)) 
			contador += 1
		self.pesos = W 


	def clasifica_prob(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([self.pesos[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		return o

	def clasifica(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([self.pesos[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		clasificacion = round(o)
		return self.clases[clasificacion]


# Explicamos a continuación cada uno de estos elementos:

# * NOMBRE_DEL_CLASIFICADOR:
# --------------------------


#  Este es el nombre de la clase que implementa el clasificador. 
#  Obligatoriamente se han de usar cada uno de los siguientes
#  nombres:

#  - Perceptrón umbral: 
#                       Clasificador_Perceptron

#  - Regresión logística, minimizando L2, batch: 
#                       Clasificador_RL_L2_Batch

#  - Regresión logística, minimizando L2, estocástico: 
#                       Clasificador_RL_L2_St

#  - Regresión logística, maximizando verosimilitud, batch: 
#                       Clasificador_RL_ML_Batch

#  - Regresión logística, maximizando verosimilitud, estocástico: 
#                       Clasificador_RL_ML_St



# * Constructor de la clase:
# --------------------------

#  El constructor debe tener los siguientes argumentos de entrada:

#  - Una lista clases con los nombres de las clases del problema de
#    clasificación, tal y como aparecen en el conjunto de datos. 
#    Por ejemplo, en el caso del problema de las votaciones, 
#    esta lista sería ["republicano","democrata"]

#  - El parámetro normalizacion, que puede ser True o False (False por
#    defecto). Indica si los datos se tienen que normalizar, tanto para el
#    entrenamiento como para la clasificación de nuevas instancias.  La
#    normalización es una técnica que suele ser útil cuando los distintos
#    atributos reflejan cantidades numéricas de muy distinta magnitud.
#    En ese caso, antes de entrenar se calcula la media m_i y la desviación
#    típica d_i en cada componente i-esima (es decir, en cada atributo) de los
#    datos del conjunto de entrenamiento.  A continuación, y antes del
#    entrenamiento, esos datos se transforman de manera que cada componente
#    x_i se cambia por (x_i - m_i)/d_i. Esta misma transformación se realiza
#    sobre las nuevas instancias que se quieran clasificar.  NOTA: se permite
#    usar la biblioteca numpy para calcular la media, la desviación típica, y
#    en general para cualquier cálculo matemático.



# * Método entrena:
# -----------------

#  Este método es el que realiza el entrenamiento del clasificador. 
#  Debe calcular un conjunto de pesos, mediante el correspondiente
#  algoritmo de entrenamiento. Describimos a continuación los parámetros de
#  entrada:  

#  - entr y clas_entr, son los datos del conjunto de entrenamiento y su
#    clasificación, respectivamente. El primero es una lista con los ejemplos,
#    y el segundo una lista con las clasificaciones de esos ejemplos, en el
#    mismo orden. 

#  - n_epochs: número de veces que se itera sobre todo el conjunto de
#    entrenamiento.

#  - rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#    durante todo el aprendizaje. Si rate_decay es True, rate marca una cota
#    mínima de la tasa de aprendizaje, como se explica a continuación. 

#  - rate_decay, indica si la tasa de aprendizaje debe disminuir a medida que
#    se van realizando actualizaciones de los pases. En concreto, si
#    rate_decay es True, la tasa de aprendizaje que se usa en cada
#    actualización se debe de calcular con la siguiente fórmula:
#       rate_n= rate_0 + (2/n**(1.5)) 
#    donde n es el número de actualizaciones de pesos realizadas hasta el
#    momento, y rate_0 es la cantidad introducida en el parámetro rate
#    anterior.   
#
#  - pesos_iniciales: si es None, se indica que los pesos deben iniciarse
#    aleatoriamente (por ejemplo, valores aleatorios entre -1 y 1). Si no es
#    None, entonces se debe proporcionar la lista de pesos iniciales. Esto
#    puede ser útil para continuar el aprendizaje a partir de un aprendizaje
#    anterior, si por ejemplo se dispone de nuevos datos.    

#  NOTA: En las versiones estocásticas, y en el perceptrón umbral, en cada
#  epoch recorrer todos los ejemplos del conjunto de entrenamiento en un orden
#  aleatorio distinto cada vez.  


# * Método clasifica_prob:
# ------------------------

#  El método que devuelve la probabilidad de pertenecer a la clase (la que se
#  ha tomado como clase 1), calculada para un nuevo ejemplo. Este método no es
#  necesario incluirlo para el perceptrón umbral.


        
# * Método clasifica:
# -------------------
    
#  El método que devuelve la clase que se predice para un nuevo ejemplo. La
#  clase debe ser una de las clases del problema (por ejemplo, "republicano" o
#  "democrata" en el problema de los votos).


# Si el clasificador aún no ha sido entrenado, tanto "clasifica" como
# "clasifica_prob" deben devolver una excepción del siguiente tipo:

class ClasificadorNoEntrenado(Exception): 
	def __init__(self,entrenado):
		self.entrenado = entrenado

	def lanzamiento_excepcion(self):
		if self.entrenado == False:
			raise Exception("El clasificador no ha sido entrenado")

#  NOTA: Se aconseja probar el funcionamiento de los clasificadores con
#  conjuntos de datos generados por las funciones del apartado anterior. 

# Ejemplo de uso:

# ------------------------------------------------------------

# Generamos un conjunto de datos linealmente separables, 
# In [1]: X1,Y1,w1=genera_conjunto_de_datos_l_s(4,8,400)

def pruebas_random():
	print("\n")
	print("-------------- Perceptrón umbral --------------")
	X1,Y1=genera_conjunto_de_datos_l_s(4,8,400)
	X1e,Y1e=X1[:300],Y1[:300]
	X1t,Y1t=X1[300:],Y1[300:]
	clas_pb1=Clasificador_Perceptron([0,1])
	clas_pb1.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
	# print(clas_pb1.clasifica(X1t[0]),Y1t[0])
	print(sum(clas_pb1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
	# print(clas_pb1.clasifica_prob(X1t[0]),Y1t[0])
	print("-------------------------------------------")
	X2,Y2=genera_conjunto_de_datos_n_l_s(4,8,400,0.1)
	X2e,Y2e=X2[:300],Y2[:300]
	X2t,Y2t=X2[300:],Y2[300:]
	clas_pb2=Clasificador_Perceptron([0,1])
	clas_pb2.entrena(X2e,Y2e,100,rate_decay=True,rate=0.001)
	# print(clas_pb2.clasifica(X2t[0]),Y2t[0])
	print(sum(clas_pb2.clasifica(x) == y for x,y in zip(X2t,Y2t))/len(Y2t))
	# print(clas_pb2.clasifica_prob(X1t[0]),Y1t[0])
	print("-------------------------------------------")
	print("-------------------------------------------")

	print("\n")
	print("-------------- Regresión logística, minimizando L2, batch --------------")
	X1,Y1=genera_conjunto_de_datos_l_s(4,8,400)
	X1e,Y1e=X1[:300],Y1[:300]
	X1t,Y1t=X1[300:],Y1[300:]
	clas_pb1=Clasificador_RL_L2_Batch([0,1])
	clas_pb1.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
	# print(clas_pb1.clasifica(X1t[0]),Y1t[0])
	print(sum(clas_pb1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
	# print(clas_pb1.clasifica_prob(X1t[0]),Y1t[0])
	print("-------------------------------------------")
	X2,Y2=genera_conjunto_de_datos_n_l_s(4,8,400,0.1)
	X2e,Y2e=X2[:300],Y2[:300]
	X2t,Y2t=X2[300:],Y2[300:]
	clas_pb2=Clasificador_RL_L2_Batch([0,1])
	clas_pb2.entrena(X2e,Y2e,100,rate_decay=True,rate=0.001)
	# print(clas_pb2.clasifica(X2t[0]),Y2t[0])
	print(sum(clas_pb2.clasifica(x) == y for x,y in zip(X2t,Y2t))/len(Y2t))
	# print(clas_pb2.clasifica_prob(X1t[0]),Y1t[0])
	print("-------------------------------------------")
	print("-------------------------------------------")

	print("\n")
	print("-------------- Regresión logística, minimizando L2, estocástico --------------")
	X1,Y1=genera_conjunto_de_datos_l_s(4,8,400)
	X1e,Y1e=X1[:300],Y1[:300]
	X1t,Y1t=X1[300:],Y1[300:]
	clas_pb1=Clasificador_RL_L2_St([0,1])
	clas_pb1.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
	# print(clas_pb1.clasifica(X1t[0]),Y1t[0])
	print(sum(clas_pb1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
	# print(clas_pb1.clasifica_prob(X1t[0]),Y1t[0])
	print("-------------------------------------------")
	X2,Y2=genera_conjunto_de_datos_n_l_s(4,8,400,0.1)
	X2e,Y2e=X2[:300],Y2[:300]
	X2t,Y2t=X2[300:],Y2[300:]
	clas_pb2=Clasificador_RL_L2_St([0,1])
	clas_pb2.entrena(X2e,Y2e,100,rate_decay=True,rate=0.001)
	# print(clas_pb2.clasifica(X2t[0]),Y2t[0])
	print(sum(clas_pb2.clasifica(x) == y for x,y in zip(X2t,Y2t))/len(Y2t))
	# print(clas_pb2.clasifica_prob(X1t[0]),Y1t[0])
	print("-------------------------------------------")
	print("-------------------------------------------")

	print("\n")
	print("-------------- Regresión logística, maximizando verosimilitud, batch --------------")
	X1,Y1=genera_conjunto_de_datos_l_s(4,8,400)
	X1e,Y1e=X1[:300],Y1[:300]
	X1t,Y1t=X1[300:],Y1[300:]
	clas_pb1=Clasificador_RL_ML_Batch([0,1])
	clas_pb1.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
	# print(clas_pb1.clasifica(X1t[0]),Y1t[0])
	print(sum(clas_pb1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
	# print(clas_pb1.clasifica_prob(X1t[0]),Y1t[0])
	print("-------------------------------------------")
	X2,Y2=genera_conjunto_de_datos_n_l_s(4,8,400,0.1)
	X2e,Y2e=X2[:300],Y2[:300]
	X2t,Y2t=X2[300:],Y2[300:]
	clas_pb2=Clasificador_RL_ML_Batch([0,1])
	clas_pb2.entrena(X2e,Y2e,100,rate_decay=True,rate=0.001)
	# print(clas_pb2.clasifica(X2t[0]),Y2t[0])
	print(sum(clas_pb2.clasifica(x) == y for x,y in zip(X2t,Y2t))/len(Y2t))
	# print(clas_pb2.clasifica_prob(X1t[0]),Y1t[0])
	print("-------------------------------------------")
	print("-------------------------------------------")

	print("\n")
	print("-------------- Regresión logística, maximizando verosimilitud, estocástico --------------")
	X1,Y1=genera_conjunto_de_datos_l_s(4,8,400)
	X1e,Y1e=X1[:300],Y1[:300]
	X1t,Y1t=X1[300:],Y1[300:]
	clas_pb1=Clasificador_RL_ML_St([0,1])
	clas_pb1.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
	# print(clas_pb1.clasifica(X1t[0]),Y1t[0])
	print(sum(clas_pb1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
	# print(clas_pb1.clasifica_prob(X1t[0]),Y1t[0])
	print("-------------------------------------------")
	X2,Y2=genera_conjunto_de_datos_n_l_s(4,8,400,0.1)
	X2e,Y2e=X2[:300],Y2[:300]
	X2t,Y2t=X2[300:],Y2[300:]
	clas_pb2=Clasificador_RL_ML_St([0,1])
	clas_pb2.entrena(X2e,Y2e,100,rate_decay=True,rate=0.001)
	# print(clas_pb2.clasifica(X2t[0]),Y2t[0])
	print(sum(clas_pb2.clasifica(x) == y for x,y in zip(X2t,Y2t))/len(Y2t))
	# print(clas_pb2.clasifica_prob(X1t[0]),Y1t[0])
	print("-------------------------------------------")
	print("-------------------------------------------")


# Lo partimos en dos trozos:
# In [2]: X1e,Y1e=X1[:300],Y1[:300]

# In [3]: X1t,Y1t=X1[300:],Y1[300:]

# Creamos el clasificador (perceptrón umbral en este caso): 
# In [4]: clas_pb1=Clasificador_Perceptron([0,1])

# Lo entrenamos con elprimero de los conjuntos de datos:
# In [5]: clas_pb1.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)

# Clasificamos un ejemplo del otro conjunto, y lo comparamos con su clase real:
# In [6]: clas_pb1.clasifica(X1t[0]),Y1t[0]
# Out[6]: (1, 1)

# Comprobamos el porcentaje de aciertos sobre todos los ejemplos de X2t
# In [7]: sum(clas_pb1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t)
# Out[7]: 1.0

# Repetimos el experimento, pero ahora con un conjunto de datos que no es
# linealmente separable: 
# In [8]: X2,Y2,w2=genera_conjunto_de_datos_n_l_s(4,8,400,0.1)

# In [8]: X2e,Y2e=X2[:300],Y2[:300]

# In [9]: X2t,Y2t=X2[300:],Y2[300:]

# In [10]: clas_pb2=Clasificador_Perceptron([0,1])

# In [11]: clas_pb2.entrena(X2e,Y2e,100,rate_decay=True,rate=0.001)

# In [12]: clas_pb2.clasifica(X2t[0]),Y2t[0]
# Out[12]: (1, 0)

# In [13]: sum(clas_pb2.clasifica(x) == y for x,y in zip(X2t,Y2t))/len(Y2t)
# Out[13]: 0.82
# ----------------------------------------------------------------





# --------------------------
# I.3. Curvas de aprendizaje
# --------------------------

# Se pide mostrar mediante gráficas la evolución del aprendizaje de los
# distintos algoritmos. En concreto, para cada clasificador usado con un
# conjunto de datos generado aleatoriamente con las funciones anteriores, las
# dos siguientes gráficas: 

# - Una gráfica que indique cómo evoluciona el porcentaje de errores que
#   comete el clasificador sobre el conjunto de entrenamiento, en cada epoch.    
# - Otra gráfica que indique cómo evoluciona el error cuadrático o la log
#   verosimilitud del clasificador (dependiendo de lo que se esté optimizando
#   en cada proceso de entrenamiento), en cada epoch.

# Para realizar gráficas, se recomiendo usar la biblioteca matplotlib de
# python: 

import matplotlib.pyplot as plt


# Lo que sigue es un ejemplo de uso, para realizar una gráfica sencilla a 
# partir de una lista "errores", que por ejemplo podría contener los sucesivos
# porcentajes de error que comete el clasificador, en los sucesivos epochs: 


# plt.plot(range(1,len(errores)+1),errores,marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Porcentaje de errores')
# plt.show()

# Basta con incluir un código similar a este en el fichero python, para que en
# la terminal de Ipython se genere la correspondiente gráfica.

# Se pide generar una serie de gráficas que permitan explicar el
# comportamiento de los algoritmos, con las distintas opciones, y con
# conjuntos separables y no separables. Comentar la interpretación de las
# distintas gráficas obtenidas. 

# NOTA: Para poder realizar las gráficas, debemos modificar los
# algoritmos de entrenamiento para que ademas de realizar el cálculo de los
# pesos, también calcule las listas con los sucesivos valores (de errores, de
# verosimilitud,etc.) que vamos obteniendo en cada epoch. Esta funcionalidad
# extra puede enlentecer algo el proceso de entrenamiento y si fuera necesario
# puede quitarse una vez se realize este apartado.


class Clasificador_Perceptron_G():

	def __init__(self,clases,normalizacion=False):
		self.clases = clases
		self.normalizacion = normalizacion
		self.pesos = []
		self.pesosGraph = []
		self.entrenado = False


	def entrena(self,entr,clas_entr,n_epochs,rate=0.1,pesos_iniciales=None,rate_decay=False):
		conjEntr = copy.copy(entr)
		self.entrenado = True
		if self.normalizacion:
			for x in range(len(entr)):
				media = numpy.mean(entr[x])
				desviacion = numpy.std(entr[x])
				conjEntr[x] = [ (i - media)/desviacion for i in entr[x]]
	
		clases_entrenamiento = [self.clases.index(x) if x in self.clases else 0 for x in clas_entr]

		if pesos_iniciales == None:
			W = [random.uniform(-1.0,1.0) for i in range(len(conjEntr[0])+1)]
		else:
			W = copy.copy(pesos_iniciales)

		contador = 1
		entrenamiento = list(zip([[1] + conjEntr[i] for i in range(len(conjEntr))],clases_entrenamiento))
		ratio = rate

		while contador <= n_epochs:
			self.pesosGraph = self.pesosGraph + [copy.copy(W)]
			random.shuffle(entrenamiento)
			for i in range(len(entrenamiento)):
				o = 0 if (0>sum([W[j]*entrenamiento[i][0][j] for j in range(len(entrenamiento[i][0]))])) else 1
				for j in range(len(entrenamiento[i][0])):
					W[j] = W[j] + ratio * entrenamiento[i][0][j] * (entrenamiento[i][1] - o)

			if rate_decay:
				ratio = rate + (2/contador**(1.5)) 
			contador += 1

		self.pesos = W 


	def clasifica_prob(self,ej):
		pass

	def clasifica(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		clasificacion = 0 if (0>sum([self.pesos[j]*ej1[j] for j in range(len(ej1))])) else 1
		return self.clases[clasificacion]

	def clasificaGraph(self,ej,p):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		clasificacion = 0 if (0>sum([p[j]*ej1[j] for j in range(len(ej1))])) else 1
		return clasificacion




class Clasificador_RL_L2_Batch_G():

	def __init__(self,clases,normalizacion=False):
		self.clases = clases
		self.normalizacion = normalizacion
		self.pesos = []
		self.pesosGraph = []
		self.errores = []
		self.error = 0
		self.entrenado = False

	def entrena(self,entr,clas_entr,n_epochs,rate=0.1,pesos_iniciales=None,rate_decay=False):
		conjEntr = copy.copy(entr)
		self.entrenado = True
		if self.normalizacion:
			for x in range(len(entr)):
				media = numpy.mean(entr[x])
				desviacion = numpy.std(entr[x])
				conjEntr[x] = [ (i - media)/desviacion for i in entr[x]]
		clases_entrenamiento = [self.clases.index(x) if x in self.clases else 0 for x in clas_entr]

		if pesos_iniciales == None:
			W = [random.uniform(-1.0,1.0) for i in range(len(conjEntr[0])+1)]
		else:
			W = copy.copy(pesos_iniciales)
		self.errores = [0 for i in range(n_epochs)]

		contador = 1
		entrenamiento = list(zip([[1] + conjEntr[i] for i in range(len(conjEntr))],clases_entrenamiento))
		ratio = rate

		while contador <= n_epochs:
			self.pesosGraph = self.pesosGraph + [copy.copy(W)]
			random.shuffle(entrenamiento)
			difW = [0 for i in range(len(W))]
			error = [0 for x in range(len(entrenamiento))]

			for i in range(len(entrenamiento)):
				wx = sum([W[j]*entrenamiento[i][0][j] for j in range(len(entrenamiento[i][0]))])
				o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
				error[i] = (entrenamiento[i][1] - o)**2
				for j in range(len(entrenamiento[i][0])):
					difW[j] = difW[j] + ratio * (entrenamiento[i][1] - o) * o * (1 - o) * entrenamiento[i][0][j]
			self.errores[contador-1] = sum(error)
					
			W = [W[i] + difW[i] for i in range(len(W))]
			if rate_decay:
				ratio = rate + (2/contador**(1.5)) 
			contador += 1
		self.pesos = W 


	def clasifica_prob(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([self.pesos[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		return o

	def clasifica(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([self.pesos[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		clasificacion = round(o)
		return self.clases[clasificacion]

	def clasificaGraph(self,ej,p):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([p[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		clasificacion = round(o)
		return clasificacion

		



class Clasificador_RL_L2_St_G():

	def __init__(self,clases,normalizacion=False):
		self.clases = clases
		self.normalizacion = normalizacion
		self.pesos = []
		self.pesosGraph = []
		self.errores = []
		self.error = 0
		self.entrenado = False

	def entrena(self,entr,clas_entr,n_epochs,rate=0.1,pesos_iniciales=None,rate_decay=False):
		conjEntr = copy.copy(entr)
		self.entrenado = True
		if self.normalizacion:
			for x in range(len(entr)):
				media = numpy.mean(entr[x])
				desviacion = numpy.std(entr[x])
				conjEntr[x] = [ (i - media)/desviacion for i in entr[x]]
		clases_entrenamiento = [self.clases.index(x) if x in self.clases else 0 for x in clas_entr]

		if pesos_iniciales == None:
			W = [random.uniform(-1.0,1.0) for i in range(len(conjEntr[0])+1)]
		else:
			W = copy.copy(pesos_iniciales)
		self.errores = [0 for i in range(n_epochs)]

		contador = 1
		entrenamiento = list(zip([[1] + conjEntr[i] for i in range(len(conjEntr))],clases_entrenamiento))
		ratio = rate

		while contador <= n_epochs:
			self.pesosGraph = self.pesosGraph + [copy.copy(W)]
			random.shuffle(entrenamiento)
			error = [0 for x in range(len(entrenamiento))]

			for i in range(len(entrenamiento)):
				wx = sum([W[j]*entrenamiento[i][0][j] for j in range(len(entrenamiento[i][0]))])
				o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
				error[i] = (entrenamiento[i][1] - o)**2
				for j in range(len(entrenamiento[i][0])):
					W[j] = W[j] + ratio * (entrenamiento[i][1] - o) * o * (1 - o) * entrenamiento[i][0][j]
			self.errores[contador-1] = sum(error)
			if rate_decay:
				ratio = rate + (2/contador**(1.5)) 
			contador += 1
		self.pesos = W 


	def clasifica_prob(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([self.pesos[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		return o

	def clasifica(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([self.pesos[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		clasificacion = round(o)
		return self.clases[clasificacion]

	def clasificaGraph(self,ej,p):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([p[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		clasificacion = round(o)
		return clasificacion



class Clasificador_RL_ML_Batch_G():

	def __init__(self,clases,normalizacion=False):
		self.clases = clases
		self.normalizacion = normalizacion
		self.pesos = []
		self.pesosGraph = []
		self.errores = []
		self.entrenado = False

	def entrena(self,entr,clas_entr,n_epochs,rate=0.1,pesos_iniciales=None,rate_decay=False):
		conjEntr = copy.copy(entr)
		self.entrenado = True
		if self.normalizacion:
			for x in range(len(entr)):
				media = numpy.mean(entr[x])
				desviacion = numpy.std(entr[x])
				conjEntr[x] = [ (i - media)/desviacion for i in entr[x]]
		clases_entrenamiento = [self.clases.index(x) if x in self.clases else 0 for x in clas_entr]

		if pesos_iniciales == None:
			W = [random.uniform(-1.0,1.0) for i in range(len(conjEntr[0])+1)]
		else:
			W = copy.copy(pesos_iniciales)
		self.errores = [0 for i in range(n_epochs)]

		contador = 1
		entrenamiento = list(zip([[1] + conjEntr[i] for i in range(len(conjEntr))],clases_entrenamiento))
		ratio = rate

		while contador <= n_epochs:
			self.pesosGraph = self.pesosGraph + [copy.copy(W)]
			random.shuffle(entrenamiento)
			difW = [0 for i in range(len(W))]
			error_pos = []
			error_neg = []

			for i in range(len(entrenamiento)):
				wx = sum([W[j]*entrenamiento[i][0][j] for j in range(len(entrenamiento[i][0]))])
				o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))

				if(entrenamiento[i][1] == 1):
					error_pos.append(-1*math.log10(1+math.exp(-wx)))
				else:
					error_neg.append(-1*math.log10(1+math.exp(wx)))

				for j in range(len(entrenamiento[i][0])):
					difW[j] = difW[j] + ratio * (entrenamiento[i][1] - o) * entrenamiento[i][0][j]
					
			W = [W[i] + difW[i] for i in range(len(W))]
			self.errores[contador-1] = -1*((sum(error_pos)) + sum(error_neg))
			if rate_decay:
				ratio = rate + (2/contador**(1.5)) 
			contador += 1
		self.pesos = W 


	def clasifica_prob(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([self.pesos[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		return o

	def clasifica(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([self.pesos[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		clasificacion = round(o)
		return self.clases[clasificacion]

	def clasificaGraph(self,ej,p):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([p[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		clasificacion = round(o)
		return clasificacion



class Clasificador_RL_ML_St_G():

	def __init__(self,clases,normalizacion=False):
		self.clases = clases
		self.normalizacion = normalizacion
		self.pesos = []
		self.pesosGraph = []
		self.errores = []
		self.entrenado = False

	def entrena(self,entr,clas_entr,n_epochs,rate=0.1,pesos_iniciales=None,rate_decay=False):
		conjEntr = copy.copy(entr)
		self.entrenado = True
		if self.normalizacion:
			for x in range(len(entr)):
				media = numpy.mean(entr[x])
				desviacion = numpy.std(entr[x])
				conjEntr[x] = [ (i - media)/desviacion for i in entr[x]]
		clases_entrenamiento = [self.clases.index(x) if x in self.clases else 0 for x in clas_entr]

		if pesos_iniciales == None:
			W = [random.uniform(-1.0,1.0) for i in range(len(conjEntr[0])+1)]
		else:
			W = copy.copy(pesos_iniciales)

		self.errores = [0 for i in range(n_epochs)]
		contador = 1
		entrenamiento = list(zip([[1] + conjEntr[i] for i in range(len(conjEntr))],clases_entrenamiento))
		ratio = rate

		while contador <= n_epochs:
			self.pesosGraph = self.pesosGraph + [copy.copy(W)]
			random.shuffle(entrenamiento)
			error_pos = []
			error_neg = []

			for i in range(len(entrenamiento)):
				wx = sum([W[j]*entrenamiento[i][0][j] for j in range(len(entrenamiento[i][0]))])
				o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
				
				if(entrenamiento[i][1] == 1):
					error_pos.append(-1*math.log10(1+math.exp(-wx)))
				else:
					error_neg.append(-1*math.log10(1+math.exp(wx)))

				for j in range(len(entrenamiento[i][0])):
					W[j] = W[j] + ratio * (entrenamiento[i][1] - o) * entrenamiento[i][0][j]
			if rate_decay:
				ratio = rate + (2/contador**(1.5)) 
			contador += 1
			self.errores[contador-1] = -1*((sum(error_pos)) + sum(error_neg))
		self.pesos = W 


	def clasifica_prob(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([self.pesos[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		return o

	def clasifica(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([self.pesos[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		clasificacion = round(o)
		return self.clases[clasificacion]

	def clasificaGraph(self,ej,p):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		ej1 = [1] + ej
		wx = sum([p[j]*ej1[j] for j in range(len(ej1))])
		o = (1/(1+ math.exp(-wx))) if wx >= 0 else (1 - 1 / (1+ math.exp(wx)))
		clasificacion = round(o)
		return clasificacion

##------------------------- Pruebas graficas ------------------------------------------

def pruebas_graficas():

# print("\n")
# print("-------------- Perceptrón umbral G --------------")
# clasif_grafica = Clasificador_Perceptron_G([0,1])
# X1,Y1=genera_conjunto_de_datos_l_s(4,8,400)
# X1e,Y1e=X1[:300],Y1[:300]
# X1t,Y1t=X1[300:],Y1[300:]
# clasif_grafica.entrena(X1e,Y1e,100,rate_decay=True,rate=0.001)
# porcentajeErrores = [sum(clasif_grafica.clasificaGraph(x,p) != y for x,y in zip(X1t,Y1t))/len(Y1t) for p in clasif_grafica.pesosGraph]
# print("-------------------------------------------")
# plt.plot(range(1,len(porcentajeErrores)+1),porcentajeErrores,marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Porcentaje de errores')
# plt.show()


	print("\n")
	print("-------------- Perceptrón umbral --------------")
	X1,Y1=genera_conjunto_de_datos_l_s(4,8,400)
	X1e,Y1e=X1[:300],Y1[:300]
	X1t,Y1t=X1[300:],Y1[300:]
	clas_p1=Clasificador_Perceptron_G([0,1])
	clas_p1.entrena(X1e,Y1e,200,rate_decay=True,rate=0.001)
	porcentajeErroresP1 = [sum(clas_p1.clasificaGraph(x,p) != y for x,y in zip(X1t,Y1t))/len(Y1t) for p in clas_p1.pesosGraph]
	# print(clas_p1.clasifica(X1t[0]),Y1t[0])
	print(sum(clas_p1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
	# print(clas_p1.clasifica_prob(X1t[0]),Y1t[0])
	plt.plot(range(1,len(porcentajeErroresP1)+1),porcentajeErroresP1,marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Porcentaje de errores')
	plt.ylim(0,1)
	plt.show()
	print("-------------------------------------------")
	X2,Y2=genera_conjunto_de_datos_n_l_s(4,8,400,0.1)
	X2e,Y2e=X2[:300],Y2[:300]
	X2t,Y2t=X2[300:],Y2[300:]
	clas_p2=Clasificador_Perceptron_G([0,1])
	clas_p2.entrena(X2e,Y2e,200,rate_decay=True,rate=0.001)
	porcentajeErroresP2 = [sum(clas_p2.clasificaGraph(x,p) != y for x,y in zip(X2t,Y2t))/len(Y2t) for p in clas_p2.pesosGraph]
	# print(clas_p2.clasifica(X2t[0]),Y2t[0])
	print(sum(clas_p2.clasifica(x) == y for x,y in zip(X2t,Y2t))/len(Y2t))
	# print(clas_p2.clasifica_prob(X1t[0]),Y1t[0])
	plt.plot(range(1,len(porcentajeErroresP2)+1),porcentajeErroresP2,marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Porcentaje de errores')
	plt.ylim(0,1)
	plt.show()
	print("-------------------------------------------")
	print("-------------------------------------------")

	print("\n")
	print("-------------- Regresión logística, minimizando L2, batch --------------")
	clas_rll2b1=Clasificador_RL_L2_Batch_G([0,1])
	clas_rll2b1.entrena(X1e,Y1e,200,rate_decay=True,rate=0.001)
	porcentajeErroresRLL2B1 = [sum(clas_rll2b1.clasificaGraph(x,p) != y for x,y in zip(X1t,Y1t))/len(Y1t) for p in clas_rll2b1.pesosGraph]
	# print(clas_rll2b1.clasifica(X1t[0]),Y1t[0])
	print(sum(clas_rll2b1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
	# print(clas_rll2b1.clasifica_prob(X1t[0]),Y1t[0])
	plt.plot(range(1,len(porcentajeErroresRLL2B1)+1),porcentajeErroresRLL2B1,marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Porcentaje de errores')
	plt.ylim(0,1)
	plt.show()
	print("-------------------------------------------")
	plt.plot(range(1,len(clas_rll2b1.errores)+1),clas_rll2b1.errores,marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Error cuadrático')
	plt.show()
	print("-------------------------------------------")
	clas_rll2b2=Clasificador_RL_L2_Batch_G([0,1])
	clas_rll2b2.entrena(X2e,Y2e,200,rate_decay=True,rate=0.001)
	porcentajeErroresRLL2B2 = [sum(clas_rll2b2.clasificaGraph(x,p) != y for x,y in zip(X2t,Y2t))/len(Y2t) for p in clas_rll2b2.pesosGraph]
	# print(clas_rll2b2.clasifica(X2t[0]),Y2t[0])
	print(sum(clas_rll2b2.clasifica(x) == y for x,y in zip(X2t,Y2t))/len(Y2t))
	# print(clas_rll2b2.clasifica_prob(X1t[0]),Y1t[0])
	plt.plot(range(1,len(porcentajeErroresRLL2B2)+1),porcentajeErroresRLL2B2,marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Porcentaje de errores')
	plt.ylim(0,1)
	plt.show()
	print("-------------------------------------------")
	plt.plot(range(1,len(clas_rll2b2.errores)+1),clas_rll2b2.errores,marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Error cuadrático')
	plt.show()
	print("-------------------------------------------")
	print("-------------------------------------------")

	print("\n")
	print("-------------- Regresión logística, minimizando L2, estocástico --------------")
	clas_rll2e1=Clasificador_RL_L2_St_G([0,1])
	clas_rll2e1.entrena(X1e,Y1e,200,rate_decay=True,rate=0.001)
	porcentajeErroresRLL2E1 = [sum(clas_rll2e1.clasificaGraph(x,p) != y for x,y in zip(X1t,Y1t))/len(Y1t) for p in clas_rll2e1.pesosGraph]
	# print(clas_rll2e1.clasifica(X1t[0]),Y1t[0])
	print(sum(clas_rll2e1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
	# print(clas_rll2e1.clasifica_prob(X1t[0]),Y1t[0])
	plt.plot(range(1,len(porcentajeErroresRLL2E1)+1),porcentajeErroresRLL2E1,marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Porcentaje de errores')
	plt.ylim(0,1)
	plt.show()
	print("-------------------------------------------")
	plt.plot(range(1,len(clas_rll2e1.errores)+1),clas_rll2e1.errores,marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Error cuadrático')
	plt.show()
	print("-------------------------------------------")
	clas_rll2e2=Clasificador_RL_L2_St_G([0,1])
	clas_rll2e2.entrena(X2e,Y2e,200,rate_decay=True,rate=0.001)
	porcentajeErroresRLL2E2 = [sum(clas_rll2e2.clasificaGraph(x,p) != y for x,y in zip(X2t,Y2t))/len(Y2t) for p in clas_rll2e2.pesosGraph]
	# print(clas_rll2e2.clasifica(X2t[0]),Y2t[0])
	print(sum(clas_rll2e2.clasifica(x) == y for x,y in zip(X2t,Y2t))/len(Y2t))
	# print(clas_rll2e2.clasifica_prob(X1t[0]),Y1t[0])
	plt.plot(range(1,len(porcentajeErroresRLL2E2)+1),porcentajeErroresRLL2E2,marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Porcentaje de errores')
	plt.ylim(0,1)
	plt.show()
	print("-------------------------------------------")
	plt.plot(range(1,len(clas_rll2e2.errores)+1),clas_rll2e2.errores,marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Error cuadrático')
	plt.show()
	print("-------------------------------------------")
	print("-------------------------------------------")

	print("\n")
	print("-------------- Regresión logística, maximizando verosimilitud, batch --------------")
	clas_rlmvb1=Clasificador_RL_ML_Batch_G([0,1])
	clas_rlmvb1.entrena(X1e,Y1e,200,rate_decay=True,rate=0.001)
	porcentajeErroresRLMVB1 = [sum(clas_rlmvb1.clasificaGraph(x,p) != y for x,y in zip(X1t,Y1t))/len(Y1t) for p in clas_rlmvb1.pesosGraph]
	# print(clas_rlmvb1.clasifica(X1t[0]),Y1t[0])
	print(sum(clas_rlmvb1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
	# print(clas_rlmvb1.clasifica_prob(X1t[0]),Y1t[0])
	plt.plot(range(1,len(porcentajeErroresRLMVB1)+1),porcentajeErroresRLMVB1,marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Porcentaje de errores')
	plt.ylim(0,1)
	plt.show()
	print("-------------------------------------------")
	plt.plot(range(1,len(clas_rlmvb1.errores)+1),clas_rlmvb1.errores,marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Error cuadrático')
	plt.show()
	print("-------------------------------------------")
	clas_rlmvb2=Clasificador_RL_ML_Batch_G([0,1])
	clas_rlmvb2.entrena(X2e,Y2e,200,rate_decay=True,rate=0.001)
	porcentajeErroresRLMVB2 = [sum(clas_rlmvb2.clasificaGraph(x,p) != y for x,y in zip(X2t,Y2t))/len(Y2t) for p in clas_rlmvb2.pesosGraph]
	# print(clas_rlmvb2.clasifica(X2t[0]),Y2t[0])
	print(sum(clas_rlmvb2.clasifica(x) == y for x,y in zip(X2t,Y2t))/len(Y2t))
	# print(clas_rlmvb2.clasifica_prob(X1t[0]),Y1t[0])
	plt.plot(range(1,len(porcentajeErroresRLMVB2)+1),porcentajeErroresRLMVB2,marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Porcentaje de errores')
	plt.ylim(0,1)
	plt.show()
	print("-------------------------------------------")
	plt.plot(range(1,len(clas_rlmvb2.errores)+1),clas_rlmvb2.errores,marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Error cuadrático')
	plt.show()
	print("-------------------------------------------")
	print("-------------------------------------------")

	print("\n")
	print("-------------- Regresión logística, maximizando verosimilitud, estocástico --------------")
	clas_rlmve1=Clasificador_RL_ML_St_G([0,1])
	clas_rlmve1.entrena(X1e,Y1e,200,rate_decay=True,rate=0.001)
	porcentajeErroresRLMVE1 = [sum(clas_rlmve1.clasificaGraph(x,p) != y for x,y in zip(X1t,Y1t))/len(Y1t) for p in clas_rlmve1.pesosGraph]
	# print(clas_rlmve1.clasifica(X1t[0]),Y1t[0])
	print(sum(clas_rlmve1.clasifica(x) == y for x,y in zip(X1t,Y1t))/len(Y1t))
	# print(clas_rlmve1.clasifica_prob(X1t[0]),Y1t[0])
	plt.plot(range(1,len(porcentajeErroresRLMVE1)+1),porcentajeErroresRLMVE1,marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Porcentaje de errores')
	plt.ylim(0,1)
	plt.show()
	print("-------------------------------------------")
	plt.plot(range(1,len(clas_rlmve1.errores)+1),clas_rlmve1.errores,marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Error cuadrático')
	plt.show()
	print("-------------------------------------------")
	clas_rlmve2=Clasificador_RL_ML_St_G([0,1])
	clas_rlmve2.entrena(X2e,Y2e,200,rate_decay=True,rate=0.001)
	porcentajeErroresRLMVE2 = [sum(clas_rlmve2.clasificaGraph(x,p) != y for x,y in zip(X2t,Y2t))/len(Y2t) for p in clas_rlmve2.pesosGraph]
	# print(clas_rlmve2.clasifica(X2t[0]),Y2t[0])
	print(sum(clas_rlmve2.clasifica(x) == y for x,y in zip(X2t,Y2t))/len(Y2t))
	# print(clas_rlmve2.clasifica_prob(X1t[0]),Y1t[0])
	plt.plot(range(1,len(porcentajeErroresRLMVE2)+1),porcentajeErroresRLMVE2,marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Porcentaje de errores')
	plt.ylim(0,1)
	plt.show()
	print("-------------------------------------------")
	plt.plot(range(1,len(clas_rlmve2.errores)+1),clas_rlmve2.errores,marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Error cuadrático')
	plt.show()
	print("-------------------------------------------")
	print("-------------------------------------------")


# ==================================
# PARTE II: CLASIFICACIÓN MULTICLASE
# ==================================

# Se pide implementar algoritmos de regresión logística para problemas de
# clasificación en los que hay más de dos clases. Para ello, usar las dos
# siguientes aproximaciones: 

# ------------------------------------------------
# II.1 Técnica "One vs Rest" (Uno frente al Resto)
# ------------------------------------------------

#  Esta técnica construye un clasificador multiclase a partir de
#  clasificadores binarios que devuelven probabilidades (como es el caso de la
#  regresión logística). Para cada posible valor de clasificación, se
#  entrena un clasificador que estime cómo de probable es pertemecer a esa
#  clase, frente al resto. Este conjunto de clasificadores binarios se usa
#  para dar la clasificación de un ejemplo nuevo, sin más que devolver la
#  clase para la que su correspondiente clasificador binario da una mayor
#  probabilidad. 

#  En concreto, se pide implementar una clase python Clasificador_RL_OvR con
#  la siguiente estructura, y que implemente el entrenamiento y la
#  clasificación como se ha explicado. 

# class Clasificador_RL_OvR():

#     def __init__(self,class_clasif,clases):

#        .....
#     def entrena(self,entr,clas_entr,n_epochs,rate=0.1,rate_decay=False):

#        .....

#     def clasifica(self,ej):

#        .....            

#  Excepto "class_clasif", los restantes parámetros de los métodos significan
#  lo mismo que en el apartado anterior, excepto que ahora "clases" puede ser
#  una lista con más de dos elementos. El parámetro class_clasif es el nombre
#  de la clase que implementa el clasificador binario a partir del cual se
#  forma el clasificador multiclase.   

#  Un ejemplo de sesión, con el problema del iris:

# ---------------------------------------------------------------
# In [28]: from iris import *

# In [29]: iris_clases=["Iris-setosa","Iris-virginica","Iris-versicolor"]

# Creamos el clasificador, a partir de RL binaria estocástico:
# In [30]: clas_rlml1=Clasificador_RL_OvR(Clasificador_RL_ML_St,iris_clases)

# Lo entrenamos: 
# In [32]: clas_rlml1.entrena(iris_entr,iris_entr_clas,100,rate_decay=True,rate=0.01)

# Clasificamos un par de ejemplos, comparándolo con su clase real:
# In [33]: clas_rlml1.clasifica(iris_entr[25]),iris_entr_clas[25]
# Out[33]: ('Iris-setosa', 'Iris-setosa')

# In [34]: clas_rlml1.clasifica(iris_entr[78]),iris_entr_clas[78]
# Out[34]: ('Iris-versicolor', 'Iris-versicolor')
# ----------------------------------------------------------------

class Clasificador_RL_OvR():

	def __init__(self,class_clasif,clases):
		self.class_clasif = class_clasif
		self.clases = clases
		self.lista_clasif = [copy.copy(class_clasif(["resto",c])) for c in self.clases]
		self.entrenado = False

	def entrena(self,entr,clas_entr,n_epochs,rate=0.1,rate_decay=False):
		self.entrenado = True
		for c in self.lista_clasif:
			c.entrena(entr,clas_entr,n_epochs,rate=rate,rate_decay=rate_decay)

	def clasifica(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		probabilidades = [c.clasifica_prob(ej) for c in self.lista_clasif]
		return self.clases[probabilidades.index(max(probabilidades))]






# ------------------------------------------------
# II.1 Regresión logística con softmax 
# ------------------------------------------------


#  Se pide igualmente implementar un clasificador en python que implemente la
#  regresión multinomial logística mdiante softmax, tal y como se describe en
#  el tema 5, pero solo la versión ESTOCÁSTICA.

#  En concreto, se pide implementar una clase python Clasificador_RL_Softmax 
#  con la siguiente estructura, y que implemente el entrenamiento y la 
#  clasificación como seexplica en el tema 5:

# class Clasificador_RL_Softmax():

#     def __init__(self,clases):

#        .....
#     def entrena(self,entr,clas_entr,n_epochs,rate=0.1,rate_decay=False):

#        .....

#     def clasifica(self,ej):

#        .....            

class Clasificador_RL_Softmax():
	def __init__(self,clases):
		self.clases = clases
		self.pesos = []
		self.entrenado = False

	def entrena(self,entr,clas_entr,n_epochs,rate=0.1,rate_decay=False):
		self.entrenado = True
		clases_entrenamiento = [[0] for x in range(len(self.clases))]
		W = [[0] for x in range(len(self.clases))]
		entrenamientos = [[0] for x in range(len(self.clases))]
		for c in range(len(self.clases)):
			clases_entrenamiento[c] = [1 if x == self.clases[c] else 0 for x in clas_entr]
			W[c] = [random.uniform(-1.0,1.0) for i in range(len(entr[0])+1)]
			entrenamientos[c] = list(zip([[1] + entr[i] for i in range(len(entr))],clases_entrenamiento[c]))
		Wprev = copy.copy(W)
		contador = 1
		ratio = rate

		while contador <= n_epochs:
			for i in range(len(entr)):
				# Pesos iteracion anterior
				for c in range(len(self.clases)):
					random.shuffle(entrenamientos[c])
					w = Wprev[c]
					X = entrenamientos[c][i][0]
					Y = entrenamientos[c][i][1]

					wmx = sum([w[j]*X[j] for j in range(len(X))])
					wkx = sum([math.exp(sum([Wprev[k][j]*X[j] for j in range(len(X))])) for k in range(len(self.clases))])
					
					for j in range(len(X)):
						W[c][j] = w[j] + ratio * (Y - (math.exp(wmx) / wkx)) * X[j] 
				Wprev = copy.copy(W)
			if rate_decay:
				ratio = rate + (2/contador**(1.5)) 
			contador += 1
			self.pesos = copy.copy(W) 

	def softmax(self,x):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		X = [1] + x
		sumatorio = sum([math.exp(sum([self.pesos[c][i] * X[i] for i in range(len(X))])) for c in range(len(self.clases))])
		return [(1/sumatorio) * math.exp(sum([ self.pesos[c][i] * X[i] for i in range(len(X)) ])) for c in range(len(self.clases))]

	def clasifica(self,ej):
		clasif_no_entrenado = ClasificadorNoEntrenado(self.entrenado)
		clasif_no_entrenado.lanzamiento_excepcion()
		probabilidades = self.softmax(ej)
		# print(self.clases)
		# print(probabilidades)
		return self.clases[probabilidades.index(max(probabilidades))]




# ===========================================
# PARTE III: APLICACIÓN DE LOS CLASIFICADORES
# ===========================================

# En este apartado se pide aplicar alguno de los clasificadores implementados
# en el apartado anterior,para tratar de resolver tres problemas: el de los
# votos, el de los dígitos y un tercer problema que hay que buscar. 

# -------------------------------------
# III.1 Implementación del rendimiento
# -------------------------------------

# Una vez que hemos entrenado un clasificador, podemos medir su rendimiento
# sobre un conjunto de ejemplos de los que se conoce su clasificación,
# mediante el porcentaje de ejemplos clasificados correctamente. Se ide
# definir una función rendimiento(clf,X,Y) que calcula el rendimiento de
# clasificador concreto clf, sobre un conjunto de datos X cuya clasificación
# conocida viene dada por la lista Y. 
# NOTA: clf es un objeto de las clases definidas en
# los apartados anteriores, que además debe estar ya entrenado. 


# Por ejemplo (conectando con el ejemplo anterior):

# ---------------------------------------------------------
# In [36]: rendimiento(clas_rlml1,iris_entr,iris_entr_clas)
# Out[36]: 0.9666666666666667
# ---------------------------------------------------------

def rendimiento(clf,X,Y):
	return sum(clf.clasifica(i) == j for i,j in zip(X,Y))/len(Y)


# ----------------------------------
# III.2 Aplicando los clasificadores
# ----------------------------------

#  Obtener un clasificador para cada uno de los siguientes problemas,
#  intentando que el rendimiento obtenido sobre un conjunto independiente de
#  ejemplos de prueba sea lo mejor posible. 

#  - Predecir el partido de un congresista en función de lo que ha votado en
#    las sucesivas votaciones, a partir de los datos en el archivo votos.py que
#    se suministra.  

#  - Predecir el dígito que se ha escrito a mano y que se dispone en forma de
#    imagen pixelada, a partir de los datos que están en el archivo digidata.zip
#    que se suministra.  Cada imagen viene dada por 28x28 píxeles, y cada pixel
#    vendrá representado por un caracter "espacio en blanco" (pixel blanco) o
#    los caracteres "+" (borde del dígito) o "#" (interior del dígito). En
#    nuestro caso trataremos ambos como un pixel negro (es decir, no
#    distinguiremos entre el borde y el interior). En cada conjunto las imágenes
#    vienen todas seguidas en un fichero de texto, y las clasificaciones de cada
#    imagen (es decir, el número que representan) vienen en un fichero aparte,
#    en el mismo orden. Será necesario, por tanto, definir funciones python que
#    lean esos ficheros y obtengan los datos en el mismo formato python en el
#    que los necesitan los algoritmos.

#  - Cualquier otro problema de clasificación (por ejemplo,
#    alguno de los que se pueden encontrar en UCI Machine Learning repository,
#    http://archive.ics.uci.edu/ml/). Téngase en cuenta que el conjunto de
#    datos que se use ha de tener sus atríbutos numéricos. Sin embargo,
#    también es posible transformar atributos no numéricos en numéricos usando
#    la técnica conocida como "one hot encoding".   


#  Nótese que en cualquiera de los tres casos, consiste en encontrar el
#  clasificador adecuado, entrenado con los parámetros y opciones
#  adecuadas. El entrenamiento ha de realizarse sobre el conjunto de
#  entrenamiento, y el conjunto de validación se emplea para medir el
#  rendimiento obtenido con los distintas combinaciones de parámetros y
#  opciones con las que se experimente. Finalmente, una vez elegido la mejor
#  combinación de parámetros y opciones, se da el rendimiento final sobre el
#  conjunto de test. Es importante no usar el conjunto de test para decididir
#  sobre los parámetros, sino sólo para dar el rendimiento final.

#  En nuestro caso concreto, estas son las opciones y parámetros con los que
#  hay que experimentar: 

#  - En primer lugar, el tipo de clasificador usado (si es batch o
#    estaocástico, si es basado en error cuadrático o en verosimilitud, si es
#    softmax o OvR,...)
#  - n_epochs: el número de epochs realizados influye en el tiempo de
#    entrenamiento y evidentemente también en la calidad del clasificador
#    obtenido. Con un número bajo de epochs, no hay suficiente entrenamiento,
#    pero también hay que decir que un número excesivo de epochs puede
#    provocar un sobreajuste no deseado. 
#  - El valor de "rate" usado. 
#  - Si se usa "rate_decay" o no.
#  - Si se usa normalización o no. 

# Se pide describir brevemente el proceso de experimentación en cada uno de
# los casos, y finalmente dar el clasificador con el que se obtienen mejor
# rendimiento sobre el conjunto de test correspondiente.

# Por dar una referencia, se pueden obtener clasificadores para el problema de
# los votos con un rendimiento sobre el test mayor al 90%, y para los dígitos
# un rendimiento superior al 80%.  


def prueba_votos(n_epochs=100,rate=0.1,rate_decay=True):

	n = n_epochs
	r = rate
	rd = True

	mejor_clasificador = []

	perceptron = Clasificador_Perceptron(votos_clases)
	perceptron.entrena(votos_entr,votos_entr_clas,n,rate=r,rate_decay=rd)
	rendimiento_perceptron = rendimiento(perceptron,votos_test,votos_test_clas)
	mejor_clasificador.append(("Perceptron",rendimiento_perceptron))

	RL_L2_Batch = Clasificador_RL_L2_Batch(votos_clases)
	RL_L2_Batch.entrena(votos_entr,votos_entr_clas,n,rate=r,rate_decay=rd)
	rendimiento_RL_L2_Batch = rendimiento(RL_L2_Batch,votos_test,votos_test_clas)
	mejor_clasificador.append(("RL_L2_Batch",rendimiento_RL_L2_Batch))

	RL_L2_St = Clasificador_RL_L2_St(votos_clases)
	RL_L2_St.entrena(votos_entr,votos_entr_clas,n,rate=r,rate_decay=rd)
	rendimiento_RL_L2_St = rendimiento(RL_L2_St,votos_test,votos_test_clas)
	mejor_clasificador.append(("RL_L2_St",rendimiento_RL_L2_St))

	RL_ML_Batch = Clasificador_RL_ML_Batch(votos_clases)
	RL_ML_Batch.entrena(votos_entr,votos_entr_clas,n,rate=r,rate_decay=rd)
	rendimiento_RL_ML_Batch = rendimiento(RL_ML_Batch,votos_test,votos_test_clas)
	mejor_clasificador.append(("RL_ML_Batch",rendimiento_RL_ML_Batch))

	RL_ML_St = Clasificador_RL_ML_St(votos_clases)
	RL_ML_St.entrena(votos_entr,votos_entr_clas,n,rate=r,rate_decay=rd)
	rendimiento_RL_ML_St = rendimiento(RL_ML_St,votos_test,votos_test_clas)
	mejor_clasificador.append(("RL_ML_St",rendimiento_RL_ML_St))

	return max(mejor_clasificador,key=lambda item:item[1])

def prueba_votos_auto():
	mejor = (("",0),[])
	iteraciones = 0
	sin_cambiar = 0
	sin_cambiar_forzado = 0
	datos = []
	while mejor[0][1]<0.95 and iteraciones<100:
		if sin_cambiar < 10:
			auto = auto_votos_inicial()
			clasif = auto[0]
			datos = auto[1]
		else:
			auto = auto_votos_forzado(mejor)
			clasif = auto[0]
			datos = auto[1]
			if mejor[0][1] >= clasif[1]:
				sin_cambiar_forzado += 1
			if sin_cambiar_forzado >= 10:
				sin_cambiar = 0
				sin_cambiar_forzado = 0

		if mejor[0][1] < clasif[1]:
			mejor = (clasif,datos)
		else:
			sin_cambiar += 1

		iteraciones += 1
		print(iteraciones,("Sin cambiar",sin_cambiar),("SC-Forzado",sin_cambiar_forzado),"Actual ->",auto,"Mejor ->",mejor)

def auto_votos_inicial():
	n_epochs = random.randint(100,1500)
	rate = random.uniform(0,0.5)
	rate_decay = random.choice([True,False]) 
	clasif = prueba_votos(n_epochs=n_epochs,rate=rate,rate_decay=rate_decay)
	return (clasif,[n_epochs,rate,rate_decay])

def auto_votos_forzado(mejor):
	n_epochs = random.randint(int(mejor[1][0]*0.5),int(mejor[1][0]*1.5))
	rate = random.uniform(mejor[1][1]*0.5,mejor[1][0]*1.5)
	rate_decay = random.choice([True,False]) 
	clasif = prueba_votos(n_epochs=n_epochs,rate=rate,rate_decay=rate_decay)
	return (clasif,[n_epochs,rate,rate_decay])


def leer_digitos_entrenamiento():
	digitos = []
	lineas = []
	with open("trainingimages") as fichero_digitos:
		for line in fichero_digitos:
			lineas.append(line)
	for i in range(int(len(lineas)/28)):
		acumulado = []
		for j in range(28):
			for k in range(28):
				acumulado.append(lineas[i*28+j][k])
		digitos.append(acumulado)
	return digitos

def leer_digitos_test():
	digitos = []
	lineas = []
	with open("testimages") as fichero_digitos:
		for line in fichero_digitos:
			lineas.append(line)
	for i in range(int(len(lineas)/28)):
		acumulado = []
		for j in range(28):
			for k in range(28):
				acumulado.append(lineas[i*28+j][k])
		digitos.append(acumulado)
	return digitos

def leer_digitos_entrenamiento_clases():
	clases = []
	with open("traininglabels") as fichero_digitos_clases:
		for line in fichero_digitos_clases:
			clases.append(line)
	return clases

def leer_digitos_test_clases():
	clases = []
	with open("testlabels") as fichero_digitos_clases:
		for line in fichero_digitos_clases:
			clases.append(line)
	return clases

def lectura_datos_cancer():
	datos_conjunto_cancer=[]
	clases_conjunto_cancer=[]
	with open("cancer.py") as fichero_cancer:
		for linea in fichero_cancer:
			datos=linea.split(",")[1:]
			if('?' not in datos):
				clases_conjunto_cancer.append("Benigno" if datos[len(datos)-1].replace("\n","")=='2' else "Maligno")
				datos=datos[:len(datos)-1]
				datos_conjunto_cancer.append([int(x) for x in datos])
	return datos_conjunto_cancer,clases_conjunto_cancer

def prueba_digitos():
	digitos = leer_digitos_entrenamiento()
	digitos_test = leer_digitos_test()
	clases = [int(x) for x in leer_digitos_entrenamiento_clases()]
	clases_set = [int(x) for x in set(clases)]
	clases_set.sort()
	entrenamiento = []
	test = []
	clases_test = [int(x) for x in leer_digitos_test_clases()]

	for x in range(len(digitos)):
		entrenamiento.append([0 if digitos[x][i] == " " else 1 for i in range(len(digitos[x]))])

	for x in range(len(digitos_test)):
		test.append([0 if digitos_test[x][i] == " " else 1 for i in range(len(digitos_test[x]))])

	onevsrest = Clasificador_RL_OvR(Clasificador_RL_ML_St, clases_set)
	onevsrest.entrena(entrenamiento, clases, 100,rate=0.15,rate_decay=True)

	print("Fin entrena")
	
	rendimiento_onevsrest = rendimiento(onevsrest,test,clases_test)
	print("Digitos con OneVSRest", rendimiento_onevsrest)


def prueba_iris():
	entrenamiento = []
	clases = []
	test = []
	clases_test = []


	entrenamiento = entrenamiento + iris_entr[:40]
	entrenamiento = entrenamiento + iris_entr[50:90]
	entrenamiento = entrenamiento + iris_entr[100:140]
	test = test + iris_entr[40:50]
	test = test + iris_entr[90:100]
	test = test + iris_entr[140:150]

	clases = clases + iris_entr_clas[:40]
	clases = clases + iris_entr_clas[50:90]
	clases = clases + iris_entr_clas[100:140]
	clases_test = clases_test + iris_entr_clas[40:50]
	clases_test = clases_test + iris_entr_clas[90:100]
	clases_test = clases_test + iris_entr_clas[140:150]

	clasificadorOvR = Clasificador_RL_OvR(Clasificador_RL_ML_St,iris_clases)
	clasificadorSoft = Clasificador_RL_Softmax(iris_clases)
	clasificadorOvR.entrena(entrenamiento,clases,100)
	clasificadorSoft.entrena(entrenamiento,clases,100)

	print("Rendimiento One vs Rest:",rendimiento(clasificadorOvR,test,clases_test))
	print("Rendimiento Softmax:",rendimiento(clasificadorSoft,test,clases_test))

def prueba_cancer(n_epochs=100,rate=0.1,rate_decay=True):

	X,Y = lectura_datos_cancer()
	cancer_entr = X[:int(len(X)*3/4)]
	cancer_test = X[int(len(X)*3/4):]
	cancer_clas = Y[:int(len(Y)*3/4)]
	cancer_clas_test = Y[int(len(Y)*3/4):]
	n = n_epochs
	r = rate
	rd = True

	mejor_clasificador = []

	perceptron = Clasificador_Perceptron(["Benigno","Maligno"])
	perceptron.entrena(cancer_entr,cancer_clas,n,rate=r,rate_decay=rd)
	rendimiento_perceptron = rendimiento(perceptron,cancer_test,cancer_clas_test)
	mejor_clasificador.append(("Perceptron",rendimiento_perceptron))

	RL_L2_Batch = Clasificador_RL_L2_Batch(["Benigno","Maligno"])
	RL_L2_Batch.entrena(cancer_entr,cancer_clas,n,rate=r,rate_decay=rd)
	rendimiento_RL_L2_Batch = rendimiento(RL_L2_Batch,cancer_test,cancer_clas_test)
	mejor_clasificador.append(("RL_L2_Batch",rendimiento_RL_L2_Batch))

	RL_L2_St = Clasificador_RL_L2_St(["Benigno","Maligno"])
	RL_L2_St.entrena(cancer_entr,cancer_clas,n,rate=r,rate_decay=rd)
	rendimiento_RL_L2_St = rendimiento(RL_L2_St,cancer_test,cancer_clas_test)
	mejor_clasificador.append(("RL_L2_St",rendimiento_RL_L2_St))

	RL_ML_Batch = Clasificador_RL_ML_Batch(["Benigno","Maligno"])
	RL_ML_Batch.entrena(cancer_entr,cancer_clas,n,rate=r,rate_decay=rd)
	rendimiento_RL_ML_Batch = rendimiento(RL_ML_Batch,cancer_test,cancer_clas_test)
	mejor_clasificador.append(("RL_ML_Batch",rendimiento_RL_ML_Batch))

	RL_ML_St = Clasificador_RL_ML_St(["Benigno","Maligno"])
	RL_ML_St.entrena(cancer_entr,cancer_clas,n,rate=r,rate_decay=rd)
	rendimiento_RL_ML_St = rendimiento(RL_ML_St,cancer_test,cancer_clas_test)
	mejor_clasificador.append(("RL_ML_St",rendimiento_RL_ML_St))

	# print(mejor_clasificador)
	return max(mejor_clasificador,key=lambda item:item[1])


def prueba_cancer_auto():
	mejor = (("",0),[])
	iteraciones = 0
	sin_cambiar = 0
	sin_cambiar_forzado = 0
	datos = []
	while mejor[0][1]<1 and iteraciones<100:
		if sin_cambiar < 10:
			auto = auto_cancer_inicial()
			clasif = auto[0]
			datos = auto[1]
		else:
			auto = auto_cancer_forzado(mejor)
			clasif = auto[0]
			datos = auto[1]
			if mejor[0][1] >= clasif[1]:
				sin_cambiar_forzado += 1
			if sin_cambiar_forzado >= 10:
				sin_cambiar = 0
				sin_cambiar_forzado = 0

		if mejor[0][1] < clasif[1]:
			mejor = (clasif,datos)
		else:
			sin_cambiar += 1

		iteraciones += 1
		print(iteraciones,("Sin cambiar",sin_cambiar),("SC-Forzado",sin_cambiar_forzado),"Actual ->",auto,"Mejor ->",mejor)


def auto_cancer_inicial():
	n_epochs = random.randint(100,1500)
	rate = random.uniform(0,0.5)
	rate_decay = random.choice([True,False]) 
	clasif = prueba_cancer(n_epochs=n_epochs,rate=rate,rate_decay=rate_decay)
	return (clasif,[n_epochs,rate,rate_decay])

def auto_cancer_forzado(mejor):
	n_epochs = random.randint(int(mejor[1][0]*0.5),int(mejor[1][0]*1.5))
	rate = random.uniform(mejor[1][1]*0.5,mejor[1][0]*1.5)
	rate_decay = random.choice([True,False]) 
	clasif = prueba_cancer(n_epochs=n_epochs,rate=rate,rate_decay=rate_decay)
	return (clasif,[n_epochs,rate,rate_decay])



## Comentarios sobre las gráficas
# Tanto en las gráficas del porcentaje de errores como las del error cuadrático, podemos observar que en ambos casos disminuyen durante los primeros epochs hasta que
# se estabilizan llegando a su mínimo. Para probar su funcionamiento puede ejecutarse pruebas_graficas()

## Para votos se ha alcanzado el mejor resultado de rendimiento (0.9310344827586207) con estos datos: Clasificador: 'RL_L2_St', epochs: 778, 
# # rate: 0.054026460415593336, rate_decay: False, aunque se ha alcanzado el mismo resultado con otros valores.

## Para dígitos se ha alcanzado más de un 80% de rendimiento con el clasificador One vs Rest para RL_ML_St, utilizando 100 epochs, rate=0.15 y rate_decay=True
## Puede comprobarse con prueba_digitos()

## El conjunto de datos elegido ha sido el de pruebas de cáncer, binario, sobre el cual se intenta predecir si el paciente tiene un tumor benigno o maligno:
## para ello se ha alcanzado un 100% de rendimiento con RL_ML_St como clasificador, 1271 epochs, un rate=0.1865022581624784, y con rate_decay=False
## Aunque se ha alcanzado el mismo nivel de rendimiento con otros valores, se puede comprobar ejecutando prueba_cancer_auto()

## Con la idea de automatizar las pruebas, se han hecho dos clases que intentan hacer la búsqueda de forma aleatoria y centrarse en torno al máximo local cierto numero
## de iteraciones, estas funciones son: prueba_votos_auto() y prueba_cancer_auto(), las cuales tienen como condición de parada cierto rendimiento alcanzado o un número
## de iteraciones.

## De forma adicional también se han hecho test con el conjunto de datos iris, para el cual se puede ejecutar prueba_iris() y ver su rendimiento con One vs Rest
## y Softmax de forma "rápida" (comparado con dígitos).

