import math
from platform import node
import random
from collections import deque, defaultdict
import heapq
import numpy as np

random.seed(42)

###############################################################################
#                                Node Class                                   #
###############################################################################

class Node:
    """
    Represents a graph node with an undirected adjacency list.
    'value' can store (row, col), or any unique identifier.
    'neighbors' is a list of connected Node objects (undirected).
    """
    def __init__(self, value):
        self.value = value
        self.neighbors = []

    def add_neighbor(self, node):
        """
        Adds an undirected edge between self and node:
         - self includes node in self.neighbors
         - node includes self in node.neighbors (undirected)
        """
        # TODO: Implement adding a neighbor in an undirected manner
        if node not in self.neighbors: 
            self.neighbors.append(node) # Add node to self's neighbor list
        if self not in node.neighbors:
            node.neighbors.append(self) # Add self to node's neighbor list
        
    def __repr__(self):
        return f"Node({self.value})"
    
    def __lt__(self, other):
        return self.value < other.value


###############################################################################
#                   Maze -> Graph Conversion (Undirected)                     #
###############################################################################

def parse_maze_to_graph(maze):
    """
    Converts a 2D maze (numpy array) into an undirected graph of Node objects.
    maze[r][c] == 0 means open cell; 1 means wall/blocked.

    Returns:
        nodes_dict: dict[(r, c): Node] mapping each open cell to its Node
        start_node : Node corresponding to (0, 0), or None if blocked
        goal_node  : Node corresponding to (rows-1, cols-1), or None if blocked
    """
    rows, cols = maze.shape
    nodes_dict = {}

    # 1) Create a Node for each open cell
    # 2) Link each node with valid neighbors in four directions (undirected)
    # 3) Identify start_node (if (0,0) is open) and goal_node (if (rows-1, cols-1) is open)

    # TODO: Implement the logic to build nodes and link neighbors

    start_node = None
    goal_node = None

    # 1)
    for r in range(rows):
            for c in range(cols):
                if maze[r][c] == 0:  # Open cell
                    nodes_dict[(r, c)] = Node((r, c))

    # 2)
    for (r,c), node in nodes_dict.items():
            for dir_r, dir_c in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor_r, neighbor_c = r + dir_r, c + dir_c
                if (neighbor_r, neighbor_c) in nodes_dict:
                    neighbor_node = nodes_dict[(neighbor_r, neighbor_c)]
                    node.add_neighbor(neighbor_node)

    # 3)
    # TODO: Assign start_node and goal_node if they exist in nodes_dict
    if (0,0) in nodes_dict:
        start_node = nodes_dict[(0,0)]
    if (rows-1, cols-1) in nodes_dict:
        goal_node = nodes_dict[(rows-1, cols-1)]

    return nodes_dict, start_node, goal_node


###############################################################################
#                         BFS (Graph-based)                                    #
###############################################################################

def bfs(start_node, goal_node):
    """
    Breadth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a queue (collections.deque) to hold nodes to explore.
      2. Track visited nodes so you don't revisit.
      3. Also track parent_map to reconstruct the path once goal_node is reached.
    """
    # TODO: Implement BFS
    queue = deque([start_node]) # FIFO
    visited = set([start_node])
    parent_map = {start_node: None}

    while queue:
        current_node = queue.popleft() # Get first node in queue

        if current_node == goal_node:
            return reconstruct_path(current_node, parent_map)

        for neighbor in current_node.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                parent_map[neighbor] = current_node
                queue.append(neighbor)

    return None  # No path found


###############################################################################
#                          DFS (Graph-based)                                   #
###############################################################################

def dfs(start_node, goal_node):
    """
    Depth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a stack (Python list) to hold nodes to explore.
      2. Keep track of visited nodes to avoid cycles.
      3. Reconstruct path via parent_map if goal_node is found.
    """
    # TODO: Implement DFS
    queue = [start_node] # LIFO
    visited = set([start_node])
    parent_map = {start_node: None}

    while queue:
        current_node = queue.pop() # Get last node in stack

        if current_node == goal_node:
            return reconstruct_path(current_node, parent_map)

        for neighbor in current_node.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                parent_map[neighbor] = current_node
                queue.append(neighbor)

    return None  # No path found


###############################################################################
#                    A* (Graph-based with Manhattan)                           #
###############################################################################

def astar(start_node, goal_node):
    """
    A* search on an undirected graph of Node objects.
    Uses manhattan_distance as the heuristic, assuming node.value = (row, col).
    Returns a path (list of (row, col)) or None if not found.

    Steps (suggested):
      1. Maintain a min-heap/priority queue (heapq) where each entry is (f_score, node).
      2. f_score[node] = g_score[node] + heuristic(node, goal_node).
      3. g_score[node] is the cost from start_node to node.
      4. Expand the node with the smallest f_score, update neighbors if a better path is found.
    """
    # TODO: Implement A*
    openList = [start_node] # Nodes to be evaluated
    closedList = [] # Nodes already evaluated

    g_score = {start_node: 0}
    h_score = {start_node: manhattan_distance(start_node, goal_node)}
    f_score = {start_node: g_score[start_node] + h_score[start_node]}
    start_node.parent = None

    while openList:
        # Want node with lowest f(n) = g(n) + h(n) -- Priority Queue
        current = min(openList, key=lambda node: f_score.get(node, float('inf')))

        if current == goal_node:
            return reconstruct_path(current, {node: node.parent for node in closedList + [current]})

        # Remove current node from open to closed list
        openList.remove(current)
        closedList.append(current)

        # Check neighboring nodes
        for neighbor in current.neighbors:
            if neighbor in closedList:
                continue

            tentative_g_score = g_score[current] + 1 # Assuming cost/distance between nodes is 1

            if neighbor not in openList:
                openList.append(neighbor)
            elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue # Means this path is not better

            # Record best path so far
            neighbor.parent = current
            g_score[neighbor] = tentative_g_score
            h_score[neighbor] = manhattan_distance(neighbor, goal_node)
            f_score[neighbor] = g_score[neighbor] + h_score[neighbor]

    return None  # No path found

def manhattan_distance(node_a, node_b):
    """
    Helper: Manhattan distance between node_a.value and node_b.value 
    if they are (row, col) pairs.
    """
    # TODO: Return |r1 - r2| + |c1 - c2|
    node_a = node_a.value
    node_b = node_b.value

    r1, c1 = node_a
    r2, c2 = node_b

    return abs(r1 - r2) + abs(c1 - c2)

###############################################################################
#                 Bidirectional Search (Graph-based)                          #
###############################################################################

def bidirectional_search(start_node, goal_node):
    """
    Bidirectional search on an undirected graph of Node objects.
    Returns list of (row, col) from start to goal, or None if not found.

    Steps (suggested):
      1. Maintain two frontiers (queues), one from start_node, one from goal_node.
      2. Alternate expansions between these two queues.
      3. If the frontiers intersect, reconstruct the path by combining partial paths.
    """
    # TODO: Implement bidirectional search
    # Two simultaneous BFS
    start_queue = deque([start_node]) # From start node
    end_queue = deque([goal_node])  # From goal node

    start_visited = {start_node: None}
    end_visited = {goal_node: None}

    while start_queue and end_queue:
        # BFS from start
        if start_queue:
            current_start = start_queue.popleft() # FIFO
            if current_start in end_visited:
                # Path found, success
                return reconstruct_path(current_start, start_visited) + reconstruct_path(current_start, end_visited)[::-1][1:]

            # Explore neighbors for start path
            for neighbor in current_start.neighbors:
                if neighbor not in start_visited:
                    start_visited[neighbor] = current_start
                    start_queue.append(neighbor)

        # BFS from goal
        if end_queue:
            current_end = end_queue.popleft() # FIFO
            if current_end in start_visited:
                # Path found, success
                return reconstruct_path(current_end, start_visited) + reconstruct_path(current_end, end_visited)[::-1][1:]

            # Explore neighbors for end path
            for neighbor in current_end.neighbors:
                if neighbor not in end_visited:
                    end_visited[neighbor] = current_end
                    end_queue.append(neighbor)

    return None  # No path found


###############################################################################
#             Simulated Annealing (Graph-based)                               #
###############################################################################

def simulated_annealing(start_node, goal_node, temperature=1.0, cooling_rate=0.99, min_temperature=0.01):
    """
    A basic simulated annealing approach on an undirected graph of Node objects.
    - The 'cost' is the manhattan_distance to the goal.
    - We randomly choose a neighbor and possibly move there.
    Returns a list of (row, col) from start to goal (the path traveled), or None if not reached.

    Steps (suggested):
      1. Start with 'current' = start_node, compute cost = manhattan_distance(current, goal_node).
      2. Pick a random neighbor. Compute next_cost.
      3. If next_cost < current_cost, move. Otherwise, move with probability e^(-cost_diff / temperature).
      4. Decrease temperature each step by cooling_rate until below min_temperature or we reach goal_node.
    """
    # TODO: Implement simulated annealing
    current_node = start_node
    compute_cost = manhattan_distance(current_node, goal_node)
    
    path = [current_node.value]
    
    while temperature > min_temperature:
        # Path found, success
        if current_node == goal_node:
            return path

        neighbors = current_node.neighbors
        if not neighbors:
            break  # No more moves possible

        next_node = random.choice(neighbors)
        next_cost = manhattan_distance(next_node, goal_node)
        cost_diff = next_cost - compute_cost

        # If next cost is lower, move. Otherwise move with probability e^(-cost_diff / temperature)
        if next_cost < compute_cost or random.random() < math.exp(-cost_diff / temperature):
            current_node = next_node
            compute_cost = next_cost
            path.append(current_node.value)

        temperature *= cooling_rate 

    return None  #No path found
    
###############################################################################
#                           Helper: Reconstruct Path                           #
###############################################################################

def reconstruct_path(end_node, parent_map):
    """
    Reconstructs a path by tracing parent_map up to None.
    Returns a list of node.value from the start to 'end_node'.

    'parent_map' is typically dict[Node, Node], where parent_map[node] = parent.

    Steps (suggested):
      1. Start with end_node, follow parent_map[node] until None.
      2. Collect node.value, reverse the list, return it.
    """
    # TODO: Implement path reconstruction
    path = []
    current_node = end_node

    while current_node is not None:
        path.append(current_node.value)
        current_node = parent_map.get(current_node)

    path.reverse()  # Reverse to get path from start to end
    return path


###############################################################################
#                              Demo / Testing                                 #
###############################################################################
if __name__ == "__main__":
    # A small demonstration that the code runs (with placeholders).
    # This won't do much yet, as everything is unimplemented.
    random.seed(42)
    np.random.seed(42)

    # Example small maze: 0 => open, 1 => wall
    maze_data = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ])

    # Parse into an undirected graph
    nodes_dict, start_node, goal_node = parse_maze_to_graph(maze_data)
    print("Created graph with", len(nodes_dict), "nodes.")
    print("Start Node:", start_node)
    print("Goal Node :", goal_node)

    # Test BFS (will return None until implemented)
    path_bfs = bfs(start_node, goal_node)
    print("BFS Path:", path_bfs)

    # DFS
    path_dfs = dfs(start_node, goal_node)
    print("DFS Path:", path_dfs)

    # A*
    path_astar = astar(start_node, goal_node)
    print("A* Path:", path_astar)
    
    # Bidirectional
    path_bidirectional = bidirectional_search(start_node, goal_node)
    print("Bidirectional Path:", path_bidirectional)

    # Simulated Annealing
    path_annealing = simulated_annealing(start_node, goal_node)
    print("Simulated Annealing Path:", path_annealing)