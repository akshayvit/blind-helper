import dwave_networkx as dnx
import matplotlib.pyplot as plt
graph = dnx.chimera_graph(1, 1, 4)
dnx.draw_chimera(graph)
plt.show()
