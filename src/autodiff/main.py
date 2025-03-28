#import from /lib
from lib.MLP import *
from lib.graph import *
from lib.value import *
import os
import sys

if __name__ == "__main__":
    #fitur, 3 input, 4 iterasi
    xs = [
        [2.0,3.0,-1],
        [3.0,-1.0,0.5],
        [0.5,1.0,1.0],
        [1.0,1.0,-1.0]
    ]


    #label  
    ys = [
        [1.0, -1.0],
        [-1.0, 1.0,],
        [-1.0, 1.0],
        [1.0, -1.0]
    ]  #s



    input_layer = 3
    layer_f_activations = [
    [2,'softmax'],#hidden layer 1
    [2,'softmax'] #output layer
    ]

    #  Zero initialization
    # Random dengan distribusi uniform.
    # Menerima parameter lower bound (batas minimal) dan upper bound (batas maksimal)
    # Menerima parameter seed untuk reproducibility
    # Random dengan distribusi normal.
    # Menerima parameter mean dan variance
    # Menerima parameter seed untuk reproducibility

    

    weight = Weight("uniform", 30, input_layer, lower=-0.1, upper=0.1)
    biasW = Weight("uniform", 42, input_layer, lower=-0.1, upper=0.1)

    n = MLP(input_layer,[n[0] for n in layer_f_activations],activations=[n[1] for n in layer_f_activations],weight=weight, biasW=biasW)
    print(n.layers[0].neurons[0].w)
    for i in range(50): #50 epoch

        #Forward
        ypred = [n(x) for x in xs]
        #Sum rumus MSE
        #Todo, Loss function yg lain
        loss = sum([sum((yout_i - ygt_i)**2 for ygt_i, yout_i in zip(ygt, yout)) for ygt, yout in zip(ys, ypred)])

        #flush bobot w
        n.zero_grad()

        ##Backward
        loss.backward()
        learning_rate =0.01

         #gradient descent
        for p in n.parameters():
            #W + -lr*deltaW
            p.data += -1 *learning_rate * p.grad

        # print(i,"Lost Func (MSE) " ,loss.data)
        


    # print(n.parameters())
   


    # for x in ypred:
    #     print(x)

    # Save model
    n.save("model.pkl")

    # Load model 
    loaded_model = MLP.load("model.pkl")

    # Test output model
    original_preds = [n(x) for x in xs]
    loaded_preds = [loaded_model(x) for x in xs]

    print("\nComparing predictions:")
    for i, (orig, loaded) in enumerate(zip(original_preds, loaded_preds)):
        for j, (o, l) in enumerate(zip(orig, loaded)):
            print(f"Sample {i}, Output {j}: Original {o.data:.6f}, Loaded {l.data:.6f}")
        
  
    # draw_dot(loss).render("graph_output.dot",view = True)
    

    # draw_mlp(n).render("mlp.dot",view= True)
    # draw_mlp(loaded_model).render("mlp_loaded.dot",view= True)
