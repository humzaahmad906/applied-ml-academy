# 06 — Trees and Graphs

Trees and graphs are where interview problems get their reputation for difficulty, but they run on a small number of reusable templates. A tree is just a graph with no cycles and a single root; a graph is the general case of nodes connected by edges. Once you internalize depth-first search (DFS) and breadth-first search (BFS) — and the fact that they are the *same* traversal with a stack swapped for a queue — most of these problems become fill-in-the-blank. This lesson gives you those templates and connects them to a fact worth remembering: the computation graphs and dependency DAGs at the heart of ML systems are exactly these structures.

## Binary trees and traversals

A binary tree node holds a value and up to two children, `left` and `right`. Traversal means visiting every node in some order, and the three depth-first orders differ only in *when* you process the current node relative to its children.

```python
class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

#        1
#       / \
#      2   3
#     / \
#    4   5
root = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))
```

**In-order** (left, node, right) visits a binary *search* tree in sorted order. **Pre-order** (node, left, right) is handy for copying a tree. **Post-order** (left, right, node) processes children before the parent, which is what you need for anything that aggregates upward, like computing heights.

```python
def inorder(node, out):
    if node is None:          # base case: empty subtree
        return
    inorder(node.left, out)   # left
    out.append(node.val)      # node
    inorder(node.right, out)  # right

result = []
inorder(root, result)
print(result)   # [4, 2, 5, 1, 3]   O(n) time, O(h) stack space (h = height)
```

All three visit every node once, so they are O(n) time. The space cost is the recursion depth, O(h), where `h` is the tree height — that is O(log n) for a balanced tree but O(n) for a degenerate, list-like one.

## DFS and BFS are one idea

Here is the unlock. Depth-first search dives as deep as possible before backing up; breadth-first search explores level by level. The *only* structural difference is the container holding nodes waiting to be visited: DFS uses a **stack** (LIFO, so it dives), BFS uses a **queue** (FIFO, so it fans out).

DFS is most naturally written with recursion (the call stack *is* the stack), but you can write it iteratively with an explicit stack:

```python
def dfs_iterative(root):
    if root is None:
        return []
    out, stack = [], [root]
    while stack:
        node = stack.pop()             # LIFO -> go deep
        out.append(node.val)
        # push right first so left is processed first
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return out

print(dfs_iterative(root))   # [1, 2, 4, 5, 3]
```

BFS is the same loop with a `deque` instead of a stack, and it naturally groups nodes by level — which is why "level-order traversal" and "shortest path in an unweighted graph" are both BFS.

```python
from collections import deque

def bfs_levels(root):
    if root is None:
        return []
    out, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):        # process exactly this level
            node = q.popleft()         # FIFO -> go wide
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        out.append(level)
    return out

print(bfs_levels(root))   # [[1], [2, 3], [4, 5]]   O(n)
```

The `for _ in range(len(q))` trick snapshots the current level size so you process one full level per outer iteration — the standard way to get level-grouped output from BFS.

## Representing a graph

Trees come with `left`/`right` pointers, but general graphs need an explicit representation. The **adjacency list** — a dict mapping each node to its neighbors — is the interview default, because it is compact for the sparse graphs most problems use and gives O(1) access to a node's neighbors.

```python
from collections import defaultdict

def build_graph(edges):
    g = defaultdict(list)
    for u, v in edges:
        g[u].append(v)
        g[v].append(u)      # omit this line for a *directed* graph
    return g

graph = build_graph([(0, 1), (0, 2), (1, 3), (2, 3)])
print(dict(graph))   # {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}
```

The other representation, an adjacency matrix (an n×n grid of 0/1), is worth mentioning when the graph is dense or you need O(1) edge-existence checks, but it costs O(n²) space regardless of edge count.

## Graph traversal, with a visited set

Traversing a graph is DFS/BFS again, with one addition: because graphs can have cycles, you must track visited nodes or you will loop forever. A `set` of visited nodes is the standard guard.

```python
def graph_bfs(graph, start):
    visited = {start}
    order, q = [], deque([start])
    while q:
        node = q.popleft()
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:   # O(1) guard against cycles
                visited.add(neighbor)
                q.append(neighbor)
    return order          # O(V + E): every vertex and edge seen once

print(graph_bfs(graph, 0))   # [0, 1, 2, 3]
```

Graph traversal is O(V + E): you visit each vertex once and traverse each edge once. This is the yardstick to quote for connectivity, reachability, counting connected components, and shortest path in an unweighted graph (BFS gives it for free because it expands in order of distance).

## Topological sort

When a directed graph has no cycles (a DAG), a topological sort orders the nodes so every edge points "forward" — every prerequisite comes before what depends on it. This is exactly how a build system orders compilation, how a scheduler orders tasks, and — the ML connection — how a framework orders operations in a computation graph so each tensor is computed before the ops that consume it.

Kahn's algorithm builds the order using in-degrees (how many edges point *into* each node). Repeatedly take a node with in-degree zero (nothing left blocking it), emit it, and decrement its neighbors.

```python
def topo_sort(graph, nodes):
    indeg = {n: 0 for n in nodes}
    for u in graph:
        for v in graph[u]:
            indeg[v] += 1
    q = deque([n for n in nodes if indeg[n] == 0])   # no prerequisites
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph[u]:
            indeg[v] -= 1               # one prerequisite satisfied
            if indeg[v] == 0:
                q.append(v)
    if len(order) != len(nodes):
        raise ValueError("graph has a cycle")   # DAG requirement violated
    return order                       # O(V + E)

# 0 -> 1 -> 3, 0 -> 2 -> 3   (must do 0 first, 3 last)
dag = {0: [1, 2], 1: [3], 2: [3], 3: []}
print(topo_sort(dag, [0, 1, 2, 3]))   # [0, 1, 2, 3]
```

The check `len(order) != len(nodes)` doubles as cycle detection: if some nodes never reach in-degree zero, they sit in a cycle and can never be ordered.

## ML systems are graphs

This is not a coincidence you should let pass in an interview. A neural network's forward pass is a directed acyclic graph of operations; autograd walks that graph in reverse (a topological order) to backpropagate gradients. A data pipeline in Airflow or a training DAG is literally a graph of tasks with dependency edges. Feature dependencies, model lineage, and microservice call graphs are all graphs. When you show an ML-engineer interviewer that you see BFS/DFS/topo-sort as the machinery under these systems — not just abstract puzzles — you connect the coding round to the job.

## Key takeaways

- A tree is an acyclic, single-root graph; traversals (in/pre/post-order) differ only in when the current node is processed, and all are O(n) time, O(h) space.
- DFS and BFS are the same traversal: DFS uses a stack (recursion or explicit) and dives deep; BFS uses a `deque` queue and explores level by level.
- BFS's level-by-level order makes it the tool for level-order traversal and shortest paths in unweighted graphs.
- Represent general graphs with an adjacency list (a dict of neighbor lists) — compact and O(1) neighbor access for the sparse graphs interviews use.
- Graph traversal needs a `visited` set to avoid infinite loops on cycles; it runs in O(V + E).
- Topological sort (Kahn's algorithm, via in-degrees) orders a DAG so dependencies come first, and the length check doubles as cycle detection.
- ML systems — autograd graphs, training DAGs, feature and service dependencies — are graphs, so these templates are directly job-relevant.

## Try it

1. **Maximum depth of a binary tree.** Write a recursive function returning the height (number of nodes on the longest root-to-leaf path). *Hint: the depth of a node is 1 plus the max depth of its children — a post-order aggregation.* What is the time and space complexity?

2. **Number of islands.** Given a grid of `"1"` (land) and `"0"` (water), count the connected groups of land (4-directionally adjacent). Treat the grid as a graph and run DFS or BFS from each unvisited land cell, marking cells visited as you go. State the complexity in terms of rows and columns.

3. **Course schedule.** Given `numCourses` and a list of `[course, prerequisite]` pairs, return whether you can finish all courses (i.e., the dependency graph has no cycle). Build an adjacency list and run a topological sort; if it cannot order every course, there is a cycle. Test on `(2, [[1,0]])` (True) and `(2, [[1,0],[0,1]])` (False).
