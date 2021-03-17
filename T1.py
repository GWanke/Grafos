import numpy as np
import collections
#Grafo implementado com a estrutura de dados dicionario do python.
class Grafo():
	def __init__(self, dictGraf=None):
		self.dictGraf = dictGraf


   

	def bfs(self,raiz='a'):

		#Substitui a necessidade de utilizar atributo de cores na classe de nos.
		visitados = set([raiz])
		fila = collections.deque([raiz])
		resultado = []
	 
		while fila:
			noAtual = fila.popleft()
			resultado.append(noAtual)
			for vizinho in self.dictGraf[noAtual]:
				if vizinho not in visitados:
					visitados.add(vizinho)
					fila.append(vizinho)         
		return resultado

def main():
    listaAdj = { "a" : set(["b","c"]),
                "b" : set(["a", "d"]),
                "c" : set(["a", "d"]),
                "d" : set(["e"]),
                "e" : set(["a"])
                }
    grafo = Grafo(listaAdj)
    print(grafo.bfs())


if __name__ == '__main__':
    main()