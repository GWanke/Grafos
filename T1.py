#Trabalho destinado a segunda implementacao do trabalho pratico da disciplina de Grafos
#da Universidade Estadual de Maringa, ano de 2021.

#Autores:Gustavo Rodrigues Wanke - RA:91671
#		 Fernando Silva Silverio - RA:98936

import numpy as np
from collections import deque as dq
import pytest
import itertools
import random
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functools import reduce
from functools import wraps
import time
import operator
import math


#funcao utilizada como decorator para saber o tempo de execucao de um metodo.(Basta colocar @timeit em cima do mesmo).
def timeit(my_func):
    @wraps(my_func)
    def timed(*args, **kw):
    
        tstart = time.time()
        output = my_func(*args, **kw)
        tend = time.time()
        
        print('"{}" demorou {:.3f} ms para executar\n'.format(my_func.__name__, (tend - tstart) * 1000))
        return output
    return timed

class Vertice():
	def __init__(self, no, cor = None, pai = None, dist = None, visitado = None, chave = None):
		self.idx = no
		self.cor = cor
		self.pai = pai
		self.dist = dist
		self.chave = chave
		self.visitado = visitado
		self.listaAdj = {}

	def show_lista_adj(self):
		return self.listaAdj.keys()

	def add_vizinho(self,other,peso):
		self.listaAdj[other] = peso

	def get_idx(self):
		return self.idx

	def get_conexoes(self):
		return self.listaAdj.keys()

	def get_peso(self, other):
		return self.listaAdj[other]

	def __str__(self):
		return str(self.idx) + ' adjacente' + str([x.idx for x in self.listaAdj])
		#return str(self.idx)



#Grafo implementado com a estrutura de dados dicionario do python.
class Grafo():
	def __init__(self):
		self.vert_dict = {}
		#self.aresta_dict = {}
		self.aresta_list = []
		self.num_vert = 0
		self.num_arestas = 0

	#Metodo necessario para iterar sobre o grafo. E chamado quando fizermos um for em cima do grafo.
	def __iter__(self):
		return iter(self.vert_dict.values())

	#adicionar um vertice ao grafo e retorna o mesmo.
	#Entrada: Grafo e um vertice.
	#Saida: Metodo sem retorno(adiciona o efeito colateral de acionar um vertice no dicionario do Grafo).
	def add_vert(self, no):
		self.num_vert += 1
		novo_vert = Vertice(no)
		self.vert_dict[no] = novo_vert
		return novo_vert

	def get_vert(self, n):
		if n in self.vert_dict:
			return self.vert_dict[n]

	#Checa se ambos os vertices estao na estrutura. Caso contrario, adiciona. 
	#Adiciona uma ligacao entre os vertices, com o custo padrao de 1.
	
	#Entrada: Grafo, dois vertices e um opcional de custo da aresta.
	#Saida: Metodo sem retorno(adiciona o efeito colateral de acionar uma ligacao entre dois vertices do Grafo).
	def add_aresta(self,de,para,custo = 1):
		self.num_arestas += 1
		if de not in self.vert_dict:
			self.add_vert(de)
		if para not in self.vert_dict:
			self.add_vert(para)
		self.vert_dict[de].add_vizinho(self.vert_dict[para], custo)
		self.vert_dict[para].add_vizinho(self.vert_dict[de], custo)
		itemAresta = de,para,custo
		self.aresta_list.append(itemAresta)


	def showListaAdjGlobal(self):
		for vertex in self:
			print(self.get_vert(vertex.idx))

	@property
	def vertices(self):
		return self.vert_dict.keys()
   
	#Realiza o BFS na classe grafo.
	# Entrada: raiz -> Vértice
	# Saída: resultado -> Lista de vértices (ordem de enfileiramento)
	def bfs(self,raiz):
		for vertice in self:
			vertice.dist = -1
			vertice.cor = 'Branco'
		fila = dq([raiz])
		raiz.dist = 0
		raiz.cor = 'Cinza'
		resultado = []
		while fila:
			noAtual = fila.popleft()
			resultado.append(noAtual.idx)
			for vizinho in noAtual.listaAdj:
				if vizinho.cor == 'Branco':
					vizinho.cor = 'Cinza'
					vizinho.pai = noAtual
					vizinho.dist = noAtual.dist +1
					fila.append(vizinho)
				#checagem usada pra detectar ciclos.
				elif noAtual.pai != vizinho:
					return False		
			noAtual.cor = 'Preto'
		return resultado

	# Objetivo: Retornar o diâmetro de um grafo de entrada
	# Entrada: self (grafo G)
	# Saída: valor absoluto entre dois vértices mais distantes no grafo
	def diametro(self):
		primeiroVisitado = self.bfs(self.vert_dict.values().__iter__().__next__())
		#Aplica BFS no primeiro vertice(Indice baseado no dicionario interno do grafo).
		if(primeiroVisitado):
			#ultimo da lista do BFS(primeiro vertice)
			mais_distante1 = self.vert_dict[primeiroVisitado[-1]]
			#aplica o BFS novamente no ultimo da lista do BFS anterior.
			listaR2 = self.bfs(mais_distante1)
			#ultimo da lista do segundo BFS(segundo vertice)
			mais_distante2 = self.vert_dict[listaR2[-1]]
			#Calculo da diferenca entre a distancia dos dois vertices.
			return abs(mais_distante1.dist - mais_distante2.dist)
		else:
			return None

	# Objetivo: Pegar um vertice randomicamente do grafo.
    # Entrada: O próprio grafo (G)
    # Saída: Um vertice do grafo.
	def getRandomVertex(self,numMaxVert):
		rnmdStr = str(random.randint(0, numMaxVert - 1))
		vertRandom = self.vert_dict[rnmdStr]
		return vertRandom

	# Objetivo: Verificar se todos os vértices possuem acesso a todos os outros
	# Entrada: O próprio grafo (G)
	# Saída: True ou False
	def conexo(self):
		#Reduz as vertices para um set.
		result = set(self.vert_dict.keys())
		#Transforma o resultado do BFS em um Set
		r = set(self.bfs(self.vert_dict.values().__iter__().__next__()))
		#Comparacao dos dois Sets. Se todas as vertices estiverem no resultado do BFS, isso implica que 
		#todos os vertices foram visitados. Logo, o grafo e conexo.
		if r == result:
			return True
		else:
			return False

    # Objetivo: Verifica se o grafo G é uma árvore
    # Entrada: O próprio grafo (G)
    # Saída: True ou False
	def isTree(self):
		if self.num_arestas != (self.num_vert - 1):
			return False
		if self.conexo():
			return True
		return False


####Funcoes auxiliares para o algoritmo de KRUSKAL####

### Estruturas de dados para conjuntos disjuntos retiradas do livro do Cormen

def find_set(x):
	if x != x.pai:
		x.pai = find_set(x.pai)
	return x.pai

def link(x,y):
	if x.rank > y.rank:
		y.pai = x
	else:
		x.pai = y
		if x.rank == y.rank:
			y.rank += + 1

def union(x,y):
	link(find_set(x),find_set(y))


def make_set(x):
	x.pai = x
	x.rank = 0


######### Auxiliares para PRIM ############

# Objetivo: Encontra o vértice de menor chave
# Entrada: Q -> Arranjo simples que contém todos os vértices do grafo
# Saída: Vértice de menor chave

def extract_min(Q):
	#for v in Q:
		#print (v)
	vertice_menor = min(Q, key=lambda x: x.chave)
	return vertice_menor
    
    


# Objetivo: Gerar um grafo ponderado
# Entrada: n -> int (o número de vértices)
# Saída: G -> objeto da classe Grafo (Grafo com pesos de valores entre 0 e 1 nas arestas)

def grafo_completo_com_peso(n):
	G=Grafo()
	for i in range(0,n):
		for j in range(i+1,n):
			G.add_aresta(str(i),str(j),random.uniform(0,1))	
	return G

####Passeios aleatorios####

# Objetivo: Realiza um passeio aleatório na lista de vértices
# Entrada: n -> int (a quantidade de vértices, onde n > 0)
# Saída: Árvore aleatória (G) e seu diametro.
def random_tree_random_walk(n):
	# Criando grafo com n vértices
	G = Grafo()
	for i in range(n):
		G.add_vert(str(i))
	# Vertice qualquer de G
	u = G.getRandomVertex(n)
	u.visitado = True
	#Enquanto não for adicionado o número correto de arestas para a árvore geradora, seguindo a propriedade.
	while G.num_arestas < (n - 1):
	# Vertice aleatorio de G
		v = G.getRandomVertex(n)		
		if not v.visitado:
			G.add_aresta(u.idx,v.idx)
			v.visitado = True
		u = v
	return G,G.diametro()



# Objetivo: Realiza um passeio utilizando o algoritmo de kruskal
# Entrada: int (a quantidade de vértices, onde n > 0)
# Saída: A árvore (G) e seu diametro.

def random_tree_kruskal(n):
	GFinal = Grafo()
	G = grafo_completo_com_peso(n)
	arestasFinais = MST_Kruskal(G)
	for aresta in arestasFinais:
		GFinal.add_aresta(aresta[0],aresta[1])
	return GFinal,GFinal.diametro()


# Objetivo: Realiza a execução do algoritmo de Kruskal, retornando uma árvore geradora mínima
# Entrada: grafo -> objeto da classe Grafo
# Saída: A -> Lista que contém todas as arestas pertencentes à arvore geradora mínima
		
def MST_Kruskal(grafo):
	A = []
	for vertice in grafo:
		make_set(vertice)
	#sort por peso das arestas
	grafo.aresta_list.sort(key = operator.itemgetter(2))
	for aresta in grafo.aresta_list:
		v1 = aresta[0]
		v2 = aresta[1]
		peso = aresta[2]
		if find_set(grafo.vert_dict[v1]) != find_set(grafo.vert_dict[v2]):
			arestaAdicionada = v1,v2,peso
			A.append(arestaAdicionada)
			union(grafo.vert_dict[v1],grafo.vert_dict[v2])
	return A


# Objetivo: Realiza a execução do algoritmo de Prim, retornando uma árvore geradora mínima
# Entrada: grafo, r -> onde r é um vértice arbitrário e grafo um objeto da classe Grafo
# Saída:  tree -> Lista que contém todas as arestas pertencentes à arvore geradora mínima

def MST_Prim(grafo, r):
	tree = []
	for vertice in grafo:
		vertice.chave = math.inf
		vertice.pai = None
	r.chave = 0
	Q = list(grafo)
	while Q:
		u = extract_min(Q)
		Q.remove(u)
		tree.append(u)
		for v in u.listaAdj:
			if v in Q and v.get_peso(u) < v.chave: 
				v.pai = u
				v.chave = v.get_peso(u)
	# necessario uma arvore para ser retornada, fernando n sabe fazer isso aqui kkk 
	return tree

# Objetivo: Realiza um passeio utilizando o algoritmo de prim
# Entrada: int (a quantidade de vértices, onde n > 0)
# Saída: A árvore (G) e seu diametro.

def random_tree_prim(n):
	GFinal = Grafo()
	G = grafo_completo_com_peso(n)
	r = G.getRandomVertex(len(G.vertices))
	arvore = MST_Prim(G,r)
	for no in arvore:
		if no.pai != None:
			GFinal.add_aresta(no.idx,no.pai.idx,no.chave)
	return GFinal,GFinal.diametro()




########################################################  Testes  ##############################################################
#Para as arvores nao foi testado explicitamente o diametro, pois sao arvores geradas randomicamentes.

#Na teoria basta freezar a seed do random e daria pra atrelar valor pros testes.

#Entretanto, sabemos que o valor do diametro esta certo, pois esta sendo utilizado na
#funcao auxiliar execute(abordada mais pra frente). 
#-----ARVORES-----
@pytest.fixture
def arvore_um_random():
	#Fixture de uma arvore aleatória com 6 vertices.(Caminho gerado randomicamente.)
	return random_tree_random_walk(6)

@pytest.fixture
def arvore_dois_random():
	#Fixture de uma arvore aleatória com 18 vertices.(Caminho gerado randomicamente.)
	return random_tree_random_walk(18)

@pytest.fixture
def arvore_tres_random():
	#Fixture de uma arvore aleatória com 36 vertices.(Caminho gerado randomicamente.)
	return random_tree_random_walk(36)

@pytest.fixture
def arvore_um_kruskal():
	'''Fixture de uma arvore aleatoria com 6 vertices, gerada pelo MST_Kruskal'''
	random.seed(20)
	return random_tree_kruskal(6)

@pytest.fixture
def arvore_dois_kruskal():
	'''Fixture de uma arvore aleatoria com 18 vertices, gerada pelo MST_Kruskal'''
	random.seed(20)
	return random_tree_kruskal(18)

@pytest.fixture
def arvore_tres_kruskal():
	'''Fixture de uma arvore aleatoria com 54 vertices, gerada pelo MST_Kruskal'''
	random.seed(20)
	return random_tree_kruskal(54)

	 

#-----GRAFOS-----	
@pytest.fixture
def grafo_um():
	'''Retorna um grafo inicializados para testes.'''
	g = Grafo()
	g.add_vert('a')
	g.add_vert('b')
	g.add_vert('c')
	g.add_vert('d')
	g.add_vert('e')
	g.add_vert('f')

	g.add_aresta('a', 'b')  
	g.add_aresta('a', 'c')
	g.add_aresta('a', 'f')
	g.add_aresta('b', 'd')
	g.add_aresta('b', 'a')
	g.add_aresta('b', 'e')
	g.add_aresta('c', 'a')
	g.add_aresta('d', 'b')
	g.add_aresta('e', 'b')
	g.add_aresta('f', 'a')

	return g

@pytest.fixture
def grafo_dois():
	'''Retorna um grafo inicializados para testes.'''
	g = Grafo()
	g.add_vert('a')
	g.add_vert('b')
	g.add_vert('c')
	g.add_vert('d')

	g.add_aresta('a', 'b')  
	g.add_aresta('a', 'c')
	g.add_aresta('c', 'd')

	return g

@pytest.fixture
def grafo_tres():
	'''Retorna um grafo inicializados para testes. - Grafo contem ciclos. Portanto, o BFS Ira retornar False.'''
	g = Grafo()
	g.add_vert('a')
	g.add_vert('b')
	g.add_vert('c')
	g.add_vert('d')
	g.add_vert('e')
	g.add_vert('f')

	g.add_aresta('a', 'b')  
	g.add_aresta('a', 'c')
	g.add_aresta('a', 'f')
	g.add_aresta('b', 'c')
	g.add_aresta('b', 'd')
	g.add_aresta('c', 'd')
	g.add_aresta('c', 'f')
	g.add_aresta('d', 'e')
	g.add_aresta('e', 'f')

	return g

@pytest.fixture
def grafo_completo_um():
	random.seed(20)
	'''Fixture de um grafo completo com 6 vertices.'''
	return grafo_completo_com_peso(6)

@pytest.fixture
def grafo_completo_dois():
	random.seed(20)
	'''Fixture de um grafo completo com 20 vertices.'''
	return grafo_completo_com_peso(10)

@pytest.fixture
def grafo_completo_tres():
	random.seed(20)
	'''Fixture de um grafo completo com 40 vertices.'''
	return grafo_completo_com_peso(15)

def test_BFS(grafo_um,grafo_dois,grafo_tres):
	'''tipagem de retorno do BFS ->lista'''
	assert isinstance(grafo_um.bfs(grafo_um.vert_dict['a']), list)
	'''valores de retorno. Para as variaveis de resultados, foram testados os BFS com a raiz no vertice 'a', sempre.'''
	resultado_um_bfs = ['a', 'b', 'c', 'f', 'd', 'e']
	resultado_dois_bfs = ['a', 'b', 'c', 'd']
	resultado_tres_bfs = False

	assert all([x==y for x,y in zip(grafo_um.bfs(grafo_um.vert_dict['a']),resultado_um_bfs)])
	assert all([x==y for x,y in zip(grafo_dois.bfs(grafo_dois.vert_dict['a']),resultado_dois_bfs)])
	assert grafo_tres.bfs(grafo_tres.vert_dict['a']) == resultado_tres_bfs

'''Os testes a seguir equivalem ao teste anterior(BFS), porem executados 
	em todos os valores possiveis de raiz para os grafos inicializados.'''


'''parametrize do primeiro grafo'''
@pytest.mark.parametrize("raizBFS1,resultadoBFS1", [
    ('a',['a', 'b', 'c', 'f', 'd', 'e']),
    ('b',['b', 'a', 'd', 'e', 'c', 'f']),
    ('c',['c', 'a', 'b', 'f', 'd', 'e']),
    ('d',['d', 'b', 'a', 'e', 'c', 'f']),
    ('e',['e', 'b', 'a', 'd', 'c', 'f']),
    ('f',['f', 'a', 'b', 'c', 'd', 'e']),
])

def test_grafoUmBFS(grafo_um,raizBFS1,resultadoBFS1):
	assert grafo_um.bfs(grafo_um.vert_dict[raizBFS1]) == resultadoBFS1

'''parametrize do segundo grafo'''
@pytest.mark.parametrize("raizBFS2,resultadoBFS2", [
    ('a',['a', 'b', 'c', 'd']),
    ('b',['b', 'a', 'c', 'd']),
    ('c',['c', 'a', 'd', 'b']),
    ('d',['d', 'c', 'a', 'b']),
])
def test_grafoDoisBFS(grafo_dois,raizBFS2,resultadoBFS2):
	assert grafo_dois.bfs(grafo_dois.vert_dict[raizBFS2]) == resultadoBFS2

'''parametrize do terceiro grafo'''
@pytest.mark.parametrize("raizBFS3,resultadoBFS3", [
    ('a',False),
    ('b',False),
    ('c',False),
    ('d',False),
    ('e',False),
    ('f',False),
])
def test_grafoTresBFS(grafo_tres,raizBFS3,resultadoBFS3):
	assert grafo_tres.bfs(grafo_tres.vert_dict[raizBFS3]) == resultadoBFS3

def test_diametro(grafo_um,grafo_dois,grafo_tres):
	
	assert grafo_um.diametro() == 3
	assert grafo_dois.diametro() == 3
	assert grafo_tres.diametro() == None

@pytest.mark.parametrize("Conexo,Arvore", [
    (True,True),
])
def test_geral_arvore(arvore_um_random,arvore_dois_random,arvore_tres_random,Conexo,Arvore):
	assert arvore_um_random[0].conexo() == Conexo
	assert arvore_dois_random[0].conexo() == Conexo
	assert arvore_tres_random[0].conexo() == Conexo

	assert arvore_um_random[0].isTree() == Arvore
	assert arvore_dois_random[0].isTree() == Arvore
	assert arvore_tres_random[0].isTree() == Arvore

@pytest.mark.parametrize("vertice1", [
    ('0'),
    ('1'),
    ('2'),
    ('3'),
    ('4'),
    ('5'),
])
def test_arvoreUmKruskal(arvore_um_kruskal,vertice1):

	for vert in arvore_um_kruskal[0]:
		make_set(vert)

	assert find_set(arvore_um_kruskal[0].vert_dict[vertice1]) == arvore_um_kruskal[0].vert_dict[vertice1]

	'''Nao sabemos fazer asser em diferenca com o pytest. Portanto, para isso foi usado um assert fixo.'''
	
	assert find_set(arvore_um_kruskal[0].vert_dict['5']) != arvore_um_kruskal[0].vert_dict['3']
	
	union(arvore_um_kruskal[0].vert_dict['3'],arvore_um_kruskal[0].vert_dict['4'])
	union(arvore_um_kruskal[0].vert_dict['2'],arvore_um_kruskal[0].vert_dict['3'])
	union(arvore_um_kruskal[0].vert_dict['0'],arvore_um_kruskal[0].vert_dict['5'])
	union(arvore_um_kruskal[0].vert_dict['3'],arvore_um_kruskal[0].vert_dict['5'])
	union(arvore_um_kruskal[0].vert_dict['1'],arvore_um_kruskal[0].vert_dict['5'])
	
	'''Os valores foram previamente calculados. Esta eh a razao das fixtures estarem com random.seeds setadas.'''
	assert arvore_um_kruskal[0].vert_dict['3'].rank == 0
	assert arvore_um_kruskal[0].vert_dict['4'].rank == 1
	assert arvore_um_kruskal[0].vert_dict['2'].rank == 0
	assert arvore_um_kruskal[0].vert_dict['5'].rank == 2
	assert arvore_um_kruskal[0].vert_dict['1'].rank == 0

@pytest.mark.parametrize("vertice2", [
    ('0'),
    ('2'),
    ('5'),
    ('8'),
    ('11'),
    ('14'),
    ('17'),
])
def test_arvoreUmKruskal(arvore_dois_kruskal,vertice2):

	for vert in arvore_dois_kruskal[0]:
		make_set(vert)

	assert find_set(arvore_dois_kruskal[0].vert_dict[vertice2]) == arvore_dois_kruskal[0].vert_dict[vertice2]

	'''Nao sabemos fazer asser em diferenca com o pytest. Portanto, para isso foi usado um assert fixo.'''
	
	assert find_set(arvore_dois_kruskal[0].vert_dict['11']) != arvore_dois_kruskal[0].vert_dict['16']
	
	union(arvore_dois_kruskal[0].vert_dict['4'],arvore_dois_kruskal[0].vert_dict['8'])
	union(arvore_dois_kruskal[0].vert_dict['7'],arvore_dois_kruskal[0].vert_dict['15'])
	union(arvore_dois_kruskal[0].vert_dict['6'],arvore_dois_kruskal[0].vert_dict['8'])
	union(arvore_dois_kruskal[0].vert_dict['2'],arvore_dois_kruskal[0].vert_dict['9'])
	union(arvore_dois_kruskal[0].vert_dict['2'],arvore_dois_kruskal[0].vert_dict['4'])
	
	'''Os valores foram previamente calculados. Esta eh a razao das fixtures estarem com random.seeds setadas.'''
	assert arvore_dois_kruskal[0].vert_dict['0'].rank == 0
	assert arvore_dois_kruskal[0].vert_dict['11'].rank == 0
	assert arvore_dois_kruskal[0].vert_dict['8'].rank == 2
	assert arvore_dois_kruskal[0].vert_dict['17'].rank == 0
	assert arvore_dois_kruskal[0].vert_dict['1'].rank == 0
	assert arvore_dois_kruskal[0].vert_dict['7'].rank == 0

'''O TERCEIRO FIXTURE SERIA SEMELHANTE AOS DOIS PRIMEIROS.'''

def test_MSTkruskal(grafo_completo_um,grafo_completo_dois,grafo_completo_tres):
	n1 = grafo_completo_um.num_vert
	n2 = grafo_completo_dois.num_vert
	n3 = grafo_completo_tres.num_vert

	r1 = [('3', '4', 0.10324779991117994), ('2', '3', 0.1693780871255699), ('0', '5', 0.2598274474889769), ('3', '5', 0.31913914884928973), ('1', '5', 0.5729406692492218)]


	'''teste rapido do numero de arestas do grafo completo'''

	assert grafo_completo_um.num_arestas == n1*((n1-1)/2)
	assert grafo_completo_dois.num_arestas == n2*((n2-1)/2)
	assert grafo_completo_tres.num_arestas == n3*((n3-1)/2)

	'''Teste na lista de retorno do MST_Kruskal'''

	assert len(MST_Kruskal(grafo_completo_um)) == n1-1
	assert len(MST_Kruskal(grafo_completo_dois)) == n2-1
	assert len(MST_Kruskal(grafo_completo_tres)) == n3-1 

	'''Teste de igualdade na lista de retorno do MST_Kruskal.'''
	assert MST_Kruskal(grafo_completo_um) == r1 



	

############################ MAIN E FUNCOES AUXILIARES PARA A MAIN.##########################

'''#Funcao destinada a execucao de 500 vezes para um metodo de gerar arvores aleatorias. O metodo nos da a certeza de
ser uma arvore o resultado do passeio aleatorio, e utiliza diametro deste passeio para calcular a media dos passeios,
apos 500 execucoes.

Entrada:Numero Vertices para o passeio aleatorio
Saida: Int referente a media de diametro de 500 execucoes no passeio.'''

#@timeit
# def execute(nVertic):
# 	somador = 0
# 	arvores = 0
# 	for _ in itertools.repeat(None,20):
# 		grafo,diametro = random_tree_kruskal(nVertic)
# 		if grafo.isTree():
# 			somador += diametro
# 			arvores += 1
# 	if arvores == 20:
# 		return somador/20



# def fileRandomWalk(opt):
# 	if opt == 1:
# 		with open("randomwalk.txt", "w") as f:
# 			r1 = execute(250)
# 			f.write('250 ' + str(r1) + '\n')
# 			r2 = execute(500)
# 			f.write('500 ' + str(r2) + '\n')
# 			r3 = execute(750)
# 			f.write('750 ' + str(r3) + '\n')
# 			r4 = execute(1000)
# 			f.write('1000 ' + str(r4) + '\n')
# 			r5 = execute(1250)
# 			f.write('1250 ' + str(r5) + '\n')
# 			r6 = execute(1500)
# 			f.write('1500 ' + str(r6) + '\n')
# 			r7 = execute(1750)
# 			f.write('1750 ' + str(r7) + '\n')
# 			r8 = execute(2000)
# 			f.write('2000 ' + str(r8) + '\n')
# 	elif opt == 2:
# 		with open("kruskal.txt", "w") as f:
# 			r1 = execute(250)
# 			f.write('250 ' + str(r1) + '\n')
# 			r2 = execute(500)
# 			f.write('500 ' + str(r2) + '\n')
# 			r3 = execute(750)
# 			f.write('750 ' + str(r3) + '\n')
# 			r4 = execute(1000)
# 			f.write('1000 ' + str(r4) + '\n')
# 			r5 = execute(1250)
# 			f.write('1250 ' + str(r5) + '\n')
# 			r6 = execute(1500)
# 			f.write('1500 ' + str(r6) + '\n')
# 			r7 = execute(1750)
# 			f.write('1750 ' + str(r7) + '\n')
# 			r8 = execute(2000)
# 			f.write('2000 ' + str(r8) + '\n')
	# elif opt == 3:
	# 	with open("prim.txt", "w") as f:
	# 		r1 = execute(250)
	# 		f.write('250 ' + str(r1) + '\n')
	# 		r2 = execute(500)
	# 		f.write('500 ' + str(r2) + '\n')
	# 		r3 = execute(750)
	# 		f.write('750 ' + str(r3) + '\n')
	# 		r4 = execute(1000)
	# 		f.write('1000 ' + str(r4) + '\n')
	# 		r5 = execute(1250)
	# 		f.write('1250 ' + str(r5) + '\n')
	# 		r6 = execute(1500)
	# 		f.write('1500 ' + str(r6) + '\n')
	# 		r7 = execute(1750)
	# 		f.write('1750 ' + str(r7) + '\n')
	# 		r8 = execute(2000)
	# 		f.write('2000 ' + str(r8) + '\n')



# def fit(fun, x, y):
# 	a, b = curve_fit(fun, x, y)
# 	return round(a[0], 2), fun(x, a)

@timeit
def main():
	g1,d1 = random_tree_kruskal(1500)
	#g2,d1 = random_tree_random_walk(1500)
	g,d = random_tree_prim(1500)
	print(d1,d)
	# alg = sys.argv[1]
	# if alg == 'randomwalk':
	# 	fileRandomWalk(1)
	# 	fun = lambda x, a: a * np.power(x, 1/2)
	# 	p = r'$\times \sqrt{n}$'
	# elif alg == 'kruskal' or alg == 'prim':
	# 	fileRandomWalk(2)
	# 	fun = lambda x, a: a * np.power(x, 1/3)
	# 	p = r'$\times \sqrt[3]{n}$'
	# else:
	# 	print("Algoritmo inválido:", alg)
	# lines = sys.stdin.readlines()
	# data = np.array([list(map(float, line.split())) for line in lines])
	# n = data[:, 0]
	# data = data[:, 1]
	# a, fitted = fit(fun, n, data)
	# plt.plot(n, data, 'o', label=alg.capitalize())
	# plt.plot(n, fitted, label= str(a) + p, color='grey')
	# plt.xlabel('Número de vértices')
	# plt.ylabel('Diâmetro')
	# plt.legend()
	# plt.savefig(alg + '.pdf')




if __name__ == '__main__':
	main()

#pytest T1.py -rf
#python T1.py randomwalk < randomwalk.txt || python T1.py kruskal < kruskal.txt
#python -m cProfile -s tottime T1.py


