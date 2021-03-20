import numpy as np
from collections import deque as dq


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
		#random vertice
		listaR = self.bfs(self.vert_dict['c'])
		#ultimo da lista do BFS
		mais_distante1 = self.vert_dict[listaR[-1]]
		#aplica o BFS novamente no ultimo da lista do BFS anterior
		listaR2 = self.bfs(mais_distante1)
		mais_distante2 = self.vert_dict[listaR2[-1]]
		#diferenca entre as distancias e o diametro(provavelmente pode tirar o abs e deixar a dif de mais_distante2 e mais_distante1)
		return abs(mais_distante1.dist - mais_distante2.dist)




def main():
    g = Grafo()
    g.add_vert('a')
    g.add_vert('b')
    g.add_vert('c')
    g.add_vert('d')
    g.add_vert('e')
    g.add_vert('f')

    g.add_aresta('a', 'b', 7)  
    g.add_aresta('a', 'c', 9)
    g.add_aresta('a', 'f', 14)
    g.add_aresta('b', 'c', 10)
    g.add_aresta('b', 'd', 15)
    g.add_aresta('c', 'd', 11)
    g.add_aresta('c', 'f', 2)
    g.add_aresta('d', 'e', 6)
    g.add_aresta('e', 'f', 9)

    #print(g.bfs(g.vert_dict['a']))
    print(g.diametro())

if __name__ == '__main__':
    main()