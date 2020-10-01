# -*- coding: utf-8 -*-
from .graph import Edge, reverse

# return reachable nodes from src
def getDst(g, src):
  def visit(g, u, visited):
    visited[u] = True
    for e in g[u]:
      if not visited[e.dst]:
        visit(g, e.dst, visited)

  visited = [False]*len(g)
  for u in src:
    if not visited[u]:
      visit(g, u, visited)
  dst = [v for v in range(len(g)) if visited[v]]
  return dst

def getSrc(g, dst):
  return getDst(reverse(g), dst)
