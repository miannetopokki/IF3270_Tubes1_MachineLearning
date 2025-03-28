import graphviz

def draw_mlp(mlp, format='svg'):
    dot = graphviz.Digraph(format=format, graph_attr={'rankdir': 'LR'})  

    layer_nodes = []
    
    layer_nodes.append([])
    for i in range(mlp.layers[0].input_size):
        node_id = f"input{i}"
        label = f"X{i+1}"
        dot.node(node_id, label, shape='circle', style="filled", fillcolor="red", width="0.8", height="0.8")
        layer_nodes[-1].append(node_id)

    for layer_idx, layer in enumerate(mlp.layers):
        layer_nodes.append([])
        
        for neuron_idx in range(layer.n_neurons):
            node_id = f"layer{layer_idx}_neuron{neuron_idx}"
            if layer_idx == len(mlp.layers) - 1:  
                label = f"O{neuron_idx+1}"
                color = "green"
            else:  
                label = f"H{layer_idx+1}_{neuron_idx+1}"
                color = "yellow"

            dot.node(node_id, label, shape='circle', style="filled", fillcolor=color, width="0.8", height="0.8")
            layer_nodes[-1].append(node_id)
        
        bias_id = f"bias{layer_idx+1}"
        dot.node(bias_id, f"B{layer_idx+1}", shape='circle', style="filled", fillcolor="gray", width="0.6", height="0.6")
        
        for neuron_id in layer_nodes[-1]:
            dot.edge(bias_id, neuron_id, constraint="false")

    for l in range(len(layer_nodes) - 1):
        for src in layer_nodes[l]:  
            for dst in layer_nodes[l + 1]: 
                dot.edge(src, dst)

    return dot
