# -*- coding: utf-8 -*-
from .graph import Edge

# Ford-Fulkerson algorithm
def augment(g, u, matchTo, visited):
  if u < 0: return True
  for e in g[u]:
    if not visited[e.dst]:
      visited[e.dst] = True
      if augment(g, matchTo[e.dst], matchTo, visited):
        matchTo[e.src] = e.dst
        matchTo[e.dst] = e.src
        return True
  return False

# g: bipartite graph
# L: size of the left side
def bipartiteMatching(g, L, matching):
  n = len(g)
  matchTo = [-1 for n in range(n)]
  match = 0
  for u in range(L):
    visited = [False]*n
    if augment(g, u, matchTo, visited):
      match+=1
  for u in range(L):
    if matchTo[u] >= 0:
      matching.append(Edge(u, matchTo[u]))
  return match

