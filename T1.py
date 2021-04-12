import numpy as np
from collections import deque as dq
import pytest
import itertools
import random
from functools import reduce

class Vertice():
	def __init__(self, no, cor = None, pai = None, dist = None, visitado = None):
		self.idx = no
		self.cor = cor
		self.pai = pai
		self.dist = dist
		self.visitado = visitado
		self.listaAdj = {}

	def show_lista_adj(self):
		return self.listaAdj.keys()

	def add_vizinho(self,other,peso):
		self.listaAdj[other] = peso

	# def __iter__(self):
	# 	return iter(self.listaAdj.values())
	def get_idx(self):
		return self.idx

	def __str__(self):
		return str(self.idx) + ' adjacente' + str([x.idx for x in self.listaAdj])
		#return str(self.idx)

#Grafo implementado com a estrutura de dados dicionario do python.
class Grafo():
	def __init__(self):
		self.vert_dict = {}
		self.num_vert = 0
		self.num_arestas = 0

	def copy(self):
		return Grafo()

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

	def showListaAdjGlobal(self):
		for vertex in self:
			print(self.get_vert(vertex.idx))

   
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
			noAtual.cor = 'Preto'
		return resultado

	# Objetivo: Retornar o diâmetro de um grafo de entrada
	# Entrada: self (grafo G)
	# Saída: valor absoluto entre dois vértices mais distantes no grafo
	def diametro(self):
		#Aplica BFS no primeiro vertice(Indice baseado no dicionario interno do grafo).
		listaR = self.bfs(list(self.vert_dict.values())[0])
		#ultimo da lista do BFS(primeiro vertice)
		mais_distante1 = self.vert_dict[listaR[-1]]
		#aplica o BFS novamente no ultimo da lista do BFS anterior.
		listaR2 = self.bfs(mais_distante1)
		#ultimo da lista do segundo BFS(segundo vertice)
		mais_distante2 = self.vert_dict[listaR2[-1]]
		#Calculo da diferenca entre a distancia dos dois vertices.
		return abs(mais_distante1.dist - mais_distante2.dist)

	def getRandomVertex(self,numMaxVert):
		rnmdStr = str(random.randint(0, numMaxVert - 1))
		vertRandom = self.vert_dict[rnmdStr]
		return vertRandom

	# Objetivo: Realiza um passeio aleatório na lista de vértices
	# Entrada: n -> int (a quantidade de vértices, onde n > 0)
	# Saída: Árvore aleatória (G)
	def random_tree_random_walk(self, n):
		# Criando grafo com n vértices
		G = self.copy()
		for i in range(n):
			G.add_vert(str(i))
		for vertex in G:
			vertex.visitado = False	
		# Vertice qualquer de G
		u = G.getRandomVertex(n)
		u.visitado = True
		# Enquanto não for adicionado o número correto de arestas para a árvore geradora, seguindo a propriedade.
		while G.num_arestas < (n - 1):
		# Vertice aleatorio de G
			v = G.getRandomVertex(n)		
			if not v.visitado:
				G.add_aresta(u.idx,v.idx)
				v.visitado = True
			u = v
		return G.diametro()

	# Objetivo: Verificar se todos os vértices possuem acesso a todos os outros
	# Entrada: O próprio grafo (G)
	# Saída: True ou False
	def conexo(self):
		branco = False
		for v in self:
			if v.cor == 'Branco':
				branco = True
		if branco == True:
			return False
		else:
			return True
    
    # Objetivo: Verifica se o grafo G é uma árvore
    # Entrada: O próprio grafo (G)
    # Saída: True ou False
	def isArvore(self):
		if self.num_arestas != (self.num_vert - 1):
			return False
		s = list(self.vert_dict.values())[0]
		self.bfs(s)
		return conexo(self)
 
# def repeatArvore(N, f, g,*args): 
#     for i in range(N): 
#          yield f(*args)
 
########################################################  Testes  ##############################################################
	
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
	g.add_aresta('b', 'c')
	g.add_aresta('b', 'd')
	g.add_aresta('c', 'd')
	g.add_aresta('c', 'f')
	g.add_aresta('d', 'e')
	g.add_aresta('e', 'f')

	return g

def test_BFS(grafo_um,grafo_dois,grafo_tres):
	#tipagem de retorno do BFS ->lista
	assert isinstance(grafo_um.bfs(grafo_um.vert_dict['a']), list)
	#valores de retorno. Para as variaveis de resultados, foram testados os BFS com a raiz no vertice 'a', sempre.
	resultado_um_bfs = ['a', 'b', 'c', 'f', 'd', 'e']
	resultado_dois_bfs = ['a', 'b', 'c', 'd']
	resultado_tres_bfs = ['a', 'b', 'c', 'f', 'd', 'e']

	assert all([x==y for x,y in zip(grafo_um.bfs(grafo_um.vert_dict['a']),resultado_um_bfs)])
	assert all([x==y for x,y in zip(grafo_dois.bfs(grafo_dois.vert_dict['a']),resultado_dois_bfs)])
	assert all([x==y for x,y in zip(grafo_tres.bfs(grafo_tres.vert_dict['a']),resultado_tres_bfs)])

#Os testes a seguir equivalem ao teste anterior(BFS), porem executados 
# em todos os valores possiveis de raiz para os grafos inicializados.

#parametrize do primeiro grafo
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

#parametrize do segundo grafo
@pytest.mark.parametrize("raizBFS2,resultadoBFS2", [
    ('a',['a', 'b', 'c', 'd']),
    ('b',['b', 'a', 'c', 'd']),
    ('c',['c', 'a', 'd', 'b']),
    ('d',['d', 'c', 'a', 'b']),
])
def test_grafoDoisBFS(grafo_dois,raizBFS2,resultadoBFS2):
	assert grafo_dois.bfs(grafo_dois.vert_dict[raizBFS2]) == resultadoBFS2

#parametrize do terceiro grafo
@pytest.mark.parametrize("raizBFS3,resultadoBFS3", [
    ('a',['a', 'b', 'c', 'f', 'd', 'e']),
    ('b',['b', 'a', 'c', 'd', 'f', 'e']),
    ('c',['c', 'a', 'b', 'd', 'f', 'e']),
    ('d',['d', 'b', 'c', 'e', 'a', 'f']),
    ('e',['e', 'd', 'f', 'b', 'c', 'a']),
    ('f',['f', 'a', 'c', 'e', 'b', 'd']),
])
def test_grafoTresBFS(grafo_tres,raizBFS3,resultadoBFS3):
	assert grafo_tres.bfs(grafo_tres.vert_dict[raizBFS3]) == resultadoBFS3

def test_diametro(grafo_um,grafo_dois,grafo_tres):
	
	assert grafo_um.diametro() == 3
	assert grafo_dois.diametro() == 3
	assert grafo_tres.diametro() == 2

def diametro_average(values):
   sum, n = 0, 0
   for x in values:
      sum += x
      n += 1
   return float(sum)/n


def main():
	g = Grafo()
	resp = []
	# g.add_vert('a')
	# g.add_vert('b')
	# g.add_vert('c')
	# g.add_vert('d')
	# g.add_vert('e')
	# g.add_vert('f')

	# g.add_aresta('a', 'b')  
	# g.add_aresta('a', 'c')
	# g.add_aresta('a', 'f')
	# g.add_aresta('b', 'c')
	# g.add_aresta('b', 'd')
	# g.add_aresta('c', 'd')
	# g.add_aresta('c', 'f')
	# g.add_aresta('d', 'e')
	# g.add_aresta('e', 'f')
	#resp = []
	for _ in itertools.repeat(None,500):
		resp.append(g.random_tree_random_walk(1500))
	media = reduce(np.add, resp)/500
	print(media)
 
if __name__ == '__main__':
	main()