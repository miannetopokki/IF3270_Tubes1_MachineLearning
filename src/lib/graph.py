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
    for l in mlp.layers:
        print(l)
    
    layer_nodes.append([])
    for i in range(mlp.inputlayer):
        node_id = f"input{i}"
        label = f"X{i}"
        dot.node(node_id, label, shape='circle', style="filled", fillcolor="red",width="1.0", height="1.0")
        layer_nodes[-1].append(node_id)

    h = 0
    for layer_idx, layer in enumerate(mlp.layers):
        layer_nodes.append([])
        
        
        if(layer_idx == len(mlp.layers) - 1):
            for neuron_idx, neuron in enumerate(layer.neurons):
                node_id = f"layer{layer_idx}_neuron{neuron_idx}"
                label = f"O{neuron_idx +1}"
                dot.node(node_id, label, shape='circle', style="filled", fillcolor="green",width="1.0", height="1.0")
                layer_nodes[-1].append(node_id)
        
        else:
            for neuron_idx, neuron in enumerate(layer.neurons):
                node_id = f"layer{layer_idx}_neuron{neuron_idx}"
                label = f"H{layer_idx}_{neuron_idx}"
                dot.node(node_id, label, shape='circle', style="filled", fillcolor="yellow",width="1.0", height="1.0")
                layer_nodes[-1].append(node_id)


    for l in range(len(layer_nodes) - 1):
        for src in layer_nodes[l]:
            for dst in layer_nodes[l + 1]:
                dot.edge(src, dst)

    return dot