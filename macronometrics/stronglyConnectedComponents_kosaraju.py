# -*- coding: utf-8 -*-
from .graph import Edge, reverse


def visit1(g, v, visited, order, k):
    visited[v] = True
    for e in g[v]:
        if not visited[e.dst]:
            visit1(g, e.dst, visited, order, k)
    order[k[0]] = v
    k[0] += 1


def visit2(g, v, visited, scc, k):
    visited[v] = True
    for e in g[v]:
        if not visited[e.dst]:
            visit2(g, e.dst, visited, scc, k)
    scc[k].append(v)

# Kosaraju's algorithm


def stronglyConnectedComponents(g, scc):
    scc[:] = []
    n = len(g)

    visited = [False]*n
    order = [-1]*n
    k = [0]
    for v in range(n):
        if not visited[v]:
            visit1(g, v, visited, order, k)

    g = reverse(g)
    k = -1
    visited = [False]*n
    for u in range(n):
        if not visited[order[n-1-u]]:
            k += 1
            scc.append([])
            visit2(g, order[n-1-u], visited, scc, k)

    return scc
