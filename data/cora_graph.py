class Graph:
    def __init__(self):
        self.vertices = {}
        self.edges = {}
        self.ids = set() 

    def add_vertex(self, vertex_id, data=None):
        if vertex_id in self.ids:
            raise ValueError(f"ID {vertex_id} already used for a vertex or edge.")
        self.vertices[vertex_id] = {"connections": [], "data": data}
        self.ids.add(vertex_id)

    def add_edge(self, edge_id, vertex1, vertex2, data=None):
        if edge_id in self.ids:
            raise ValueError(f"ID {edge_id} already used for a vertex or edge.")
        if vertex1 not in self.vertices or vertex2 not in self.vertices:
            raise ValueError("Both vertices must exist before adding an edge.")
        self.edges[edge_id] = {"connections": (vertex1, vertex2), "data": data}
        self.vertices[vertex1]["connections"].append(edge_id)
        self.vertices[vertex2]["connections"].append(edge_id)
        self.ids.add(edge_id)

    def get_vertex(self, vertex_id):
        return self.vertices.get(vertex_id, None)

    def get_edge(self, edge_id):
        return self.edges.get(edge_id, None)

    def display_graph(self):
        print("Vertices:")
        for vid, details in self.vertices.items():
            print(f"  ID {vid}: Connections -> {details['connections']}, Data -> {details['data']}")
        print("Edges:")
        for eid, details in self.edges.items():
            print(f"  ID {eid}: Connections -> {details['connections']}, Data -> {details['data']}")


if __name__ == "__main__":
    g = Graph()
    
    g.add_vertex(1, data="A")
    g.add_vertex(3, data="A")
    g.add_vertex(5, data="A")
    g.add_vertex(7, data="A")
    g.add_vertex(11, data="A")
    g.add_vertex(13, data="A")
    g.add_vertex(9, data="A")
    
    g.add_edge(2, 1, 3, data="B")
    g.add_edge(4, 3, 5, data="B")
    g.add_edge(6, 5, 7, data="B")
    g.add_edge(8, 7, 9, data="B")
    g.add_edge(10, 7, 11, data="B")
    g.add_edge(12, 11, 13, data="B")
    g.add_edge(14, 13, 3, data="B")
    
    g.display_graph()
