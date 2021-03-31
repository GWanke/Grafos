import numpy as np
from collections import deque as dq
import pytest
import unittest

class Vertice():
	def __init__(self, no, cor = None, pai = None, dist = None):
		self.idx = no
		self.cor = cor
		self.pai = pai
		self.dist = dist
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
		#return str(self.idx) + ' adjacente' + str([x.idx for x in self.listaAdj])
		return str(self.idx)

#Grafo implementado com a estrutura de dados dicionario do python.
class Grafo():
	def __init__(self):
		self.vert_dict = {}
		self.num_vert = 0
		#self.g = Grafo()
	#Necessario para iterar sobre o grafo
	def __iter__(self):
		return iter(self.vert_dict.values())

	#adicionar um vertice ao grafo e retorna o mesmo.
	def add_vert(self, no):
		self.num_vert += 1
		novo_vert = Vertice(no)
		self.vert_dict[no] = novo_vert
		return novo_vert

	#Checa se ambos os nos estao na estrutura. Caso contrario, adiciona. 
	#Adiciona uma ligacao entre os nos, com o custo padrao de 1.
	def add_aresta(self,de,para,custo = 1):
		if de not in self.vert_dict:
			self.add_vert(de)
		if para not in self.vert_dict:
			self.add_vert(para)
		self.vert_dict[de].add_vizinho(self.vert_dict[para],custo)
		self.vert_dict[para].add_vizinho(self.vert_dict[de],custo)

   
	#Realiza o BFS na classe grafo.
	def bfs(self,raiz):
		for vertice in self:
			vertice.dist = -1
			vertice.cor = 'Branco'
		fila = dq([raiz])
		raiz.dist = 0
		raiz.cor = 'Cinza'
		#somente para visualizacao
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
	def diametro(self):
		#Pega o primeiro vertice do grafo.
		listaR = self.bfs(list(self.vert_dict.values())[0])
		#for item in self.vert_dict.values():
		#ultimo da lista do BFS
		mais_distante1 = self.vert_dict[listaR[-1]]
		#aplica o BFS novamente no ultimo da lista do BFS anterior
		listaR2 = self.bfs(mais_distante1)
		mais_distante2 = self.vert_dict[listaR2[-1]]
		#diferenca entre as distancias e o diametro(provavelmente pode tirar o abs e deixar a dif de mais_distante2 e mais_distante1)
		return abs(mais_distante1.dist - mais_distante2.dist)

#Testes
	
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

def test_diametro(grafo_um,grafo_dois,grafo_tres):
	

	assert grafo_um.diametro() == 3
	assert grafo_dois.diametro() == 3
	assert grafo_tres.diametro() == 2

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

@pytest.mark.parametrize("raizBFS,resultadoBFS,diametro", [
    ('a',['a', 'b', 'c', 'f', 'd', 'e'], 3),
    ('b',['b', 'a', 'd', 'e', 'c', 'f'], 3),
    ('c',['c', 'a', 'b', 'f', 'd', 'e'], 3),
    ('d',['d', 'b', 'a', 'e', 'c', 'f'], 3),
    ('e',['e', 'b', 'a', 'd', 'c', 'f'], 3),
    ('f',['f', 'a', 'b', 'c', 'd', 'e'], 3),
])

def test_integracao1(grafo_um,raizBFS,resultadoBFS,diametro):

	assert grafo_um.bfs(grafo_um.vert_dict[raizBFS]) == resultadoBFS
	assert grafo_um.diametro() == diametro
 


def main():
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

	print(g.bfs(g.vert_dict['f']))

if __name__ == '__main__':
	#unittest.main()
	main()