import graphviz

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = graphviz.Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(name=str(id(n)), label = "{ w %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

def draw_mlp(mlp, format='svg'):
    dot = graphviz.Digraph(format=format, graph_attr={'rankdir': 'LR'})

    layer_nodes = []

    for layer_idx, layer in enumerate(mlp.layers):
        layer_nodes.append([])

        for neuron_idx, neuron in enumerate(layer.neurons):
            node_id = f"layer{layer_idx}_neuron{neuron_idx}"
            label = f"Neuron {neuron_idx}\n{neuron}"
            dot.node(node_id, label, shape='circle', style="filled", fillcolor="lightblue")
            layer_nodes[-1].append(node_id)

    for l in range(len(layer_nodes) - 1):
        for src in layer_nodes[l]:
            for dst in layer_nodes[l + 1]:
                dot.edge(src, dst)

    return dot