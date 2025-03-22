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


import graphviz

def draw_mlp(mlp, format='svg'):
    dot = graphviz.Digraph(format=format, graph_attr={'rankdir': 'LR'})  

    layer_nodes = []
    
    layer_nodes.append([])
    for i in range(mlp.inputlayer):
        node_id = f"input{i}"
        label = f"X{i+1}"
        dot.node(node_id, label, shape='circle', style="filled", fillcolor="red", width="1.0", height="1.0")
        layer_nodes[-1].append(node_id)

    for layer_idx, layer in enumerate(mlp.layers):
        layer_nodes.append([])
        
        for neuron_idx, neuron in enumerate(layer.neurons):
            node_id = f"layer{layer_idx}_neuron{neuron_idx}"
            if layer_idx == len(mlp.layers) - 1: 
                label = f"O{neuron_idx+1}"
                color = "green"
            else:  
                label = f"H{layer_idx+1}_{neuron_idx+1}"
                color = "yellow"

            dot.node(node_id, label, shape='circle', style="filled", fillcolor=color, width="1.0", height="1.0")
            layer_nodes[-1].append(node_id)
        
        if layer_idx < len(mlp.layers):  
            bias_id = f"bias{layer_idx+1}"
            dot.node(bias_id, f"B{layer_idx+1}", shape='circle', style="filled", fillcolor="gray", width="0.8", height="0.8")
            
            dot.attr(rank='same')
            
            for neuron_id in layer_nodes[-1]:
                dot.edge(bias_id, neuron_id, constraint="false")

    for l in range(len(layer_nodes) - 1):
        for src in layer_nodes[l]:  
            for dst in layer_nodes[l + 1]: 
                dot.edge(src, dst)

    return dot



