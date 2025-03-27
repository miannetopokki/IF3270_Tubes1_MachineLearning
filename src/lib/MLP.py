import math
import random
from lib.value import Value
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import lib.lossfunc as lf
from sklearn.utils import shuffle
from tqdm import tqdm



class Weight:
    def __init__(self, method, seed, upper=None, lower=None, mean=None, variance=None):
        self.method = method.lower()
        self.upper = upper
        self.lower = lower
        self.mean = mean
        self.variance = variance
        self.seed = seed
        self.rng = random.Random(seed)

    def __call__(self):
        # Bobot yang dihasilkan berdasarkan metode yang dipilih
        if self.method == "uniform":
            if self.lower is None or self.upper is None:
                raise ValueError("Lower and upper bounds must be provided for uniform distribution.")
            return self.rng.uniform(self.lower, self.upper)
        
        elif self.method == "normal":
            if self.mean is None or self.variance is None:
                raise ValueError("Mean and variance must be provided for normal distribution.")
            stddev = math.sqrt(self.variance)
            return self.rng.gauss(self.mean, stddev)
        
        elif self.method == "zero":
            return 0
        
        # elif self.method == "he":  
        #     stddev = math.sqrt(2.0)
        #     return self.rng.gauss(0, stddev)
        
        # elif self.method == "xavier":  
        #     stddev = math.sqrt(1.0)
        #     return self.rng.gauss(0, stddev)
        
        else:
            raise ValueError("Invalid weight initialization method.")

            raise ValueError("Invalid weight initialization method.")


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin, activation="tanh",weight: Weight=None, biasW: Weight=None):
        raw_weights = [weight() for _ in range(nin)] if weight is not None else [random.uniform(-1.0, 1.0) for _ in range(nin)]
        self.w = [Value(w) for w in raw_weights]
        raw_bias = biasW() if biasW is not None else 1.0
        self.b = Value(raw_bias)
        # print("biasW: ", self.b)
        self.activation = activation.lower()

    def __call__(self, x_input_forward):
        act = sum((wi * xi for wi, xi in zip(self.w, x_input_forward)), self.b)
        # act = np.dot(self.w, x_input_forward) + self.b

        if self.activation == "relu":
            return act.relu()
        elif self.activation == "linear":
            return act
        elif self.activation == "tanh":
            return act.tanh()
        elif self.activation == "sigmoid":
            return act.sigmoid()
        elif self.activation == "softmax":
            return act
        else:
            raise ValueError()

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.activation.capitalize()}Neuron({len(self.w)})"

class Layer(Module):
    def __init__(self, nin, nout, activation="tanh", weight: Weight=None, biasW: Weight=None):
        self.neurons = [Neuron(nin, activation=activation, weight=weight, biasW=biasW) for _ in range(nout)]
        self.activation = activation

    def __call__(self, x_input_forward):
        if self.activation == "softmax":
            raw_activations = [sum((wi * xi for wi, xi in zip(n.w, x_input_forward)), n.b) for n in self.neurons]
            return Value.softmax(raw_activations)
        else:
            out = [n(x_input_forward) for n in self.neurons] 
            return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin, nouts, activations=None, weight: Weight=None, biasW: Weight=None):
        #jika activation tidak diisi maka akan diisi dengan nilai default yaitu tanh
        if activations is None:
            activations = ["tanh"] * len(nouts)

        if len(activations) != len(nouts):
            raise ValueError("activations list must match the number of layers (nouts).")

        sz = [nin] + nouts
        print("sz: " ,sz)
        self.layers = [
            Layer(sz[i], sz[i + 1], activation=activations[i],weight=weight, biasW=biasW) for i in range(len(nouts))
        ]

        self.weight = weight
        self.biasW = biasW
        self.inputlayer = nin
        self.trainloss = []
        self.validloss=[]

    def __call__(self, x_input_forward):
        x_input_forward = np.array(x_input_forward)
        for layer in self.layers:
            x_input_forward = layer(x_input_forward)
        return x_input_forward
    
   
    def fit(self,epoch = 50,lossfunc = "MSE",learning_rate = 0.01,x=None,y=None,x_val=None,y_val = None):
        progress_bar = tqdm(range(epoch), desc="Training", unit="epoch")  # Progress bar
        for i in progress_bar:
            #Forward
            val_loss = None
            ypred = [self(x_input_forward) for x_input_forward in x]
            
            #Sum rumus MSE
            if lossfunc == "MSE":
                loss = lf.mean_squared_error(y_pred=ypred,y_true=y)
            elif lossfunc == "BCE":
                loss = lf.binary_crossentropy(y_pred=ypred,y_true=y)
            elif lossfunc == "CCE":
                loss = lf.categorical_crossentropy(y_pred=ypred,y_true=y)
            else:
                raise ValueError("Invalid loss function")
            #Todo, Loss function yg lain            

            self.trainloss.append(loss.data)
            #flush bobot w
            self.zero_grad()

            ##Backward
            loss.backward()

            #gradient descent
      
            for p in self.parameters():
                #W + -lr*deltaW
                p.data += -1 *learning_rate * p.grad
            
                # === VALIDATION LOSS ===
            if x_val is not None and y_val is not None:
                val_pred = [self(x_input_forward) for x_input_forward in x_val]
                if lossfunc == "MSE":
                    val_loss = lf.mean_squared_error(y_true=y_val,y_pred=val_pred)
                elif lossfunc == "BCE":
                    val_loss = lf.binary_crossentropy(y_true=y_val,y_pred=val_pred)
                elif lossfunc == "CCE":
                    val_loss = lf.categorical_crossentropy(y_true=y_val,y_pred=val_pred)
                else:
                    raise ValueError("Invalid loss function")
                
                self.validloss.append(val_loss.data)
            
            if val_loss is not None:
                progress_bar.set_postfix({"Train Loss": loss.data, "Val Loss": val_loss.data})
            else:
                progress_bar.set_postfix({"Train Loss": loss.data})



        return self


    def fit_minibatch(self, epoch=50, lossfunc="MSE", learning_rate=0.01, x=None, y=None, x_val=None, y_val=None, batch_size=10,reg_lambda=0.01):
        progress_bar = tqdm(range(epoch), desc="Training", unit="epoch")  # Progress bar
        
        for i in progress_bar:
            x, y = shuffle(x, y)

            batch_losses = []
            
            # mini-batch
            for j in range(0, len(x), batch_size):
                x_batch = x[j:j + batch_size]
                y_batch = y[j:j + batch_size]

                # FF
                progress_bar.set_description(f"Epoch {i+1}/{epoch} - Forwarding | Minibatch - {j}/{len(x)}")
                ypred = [self(x_input_forward) for x_input_forward in x_batch]

                # loss
                if lossfunc == "MSE":
                    loss = lf.mean_squared_error(y_pred=ypred,y_true=y)
                elif lossfunc == "BCE":
                    loss = lf.binary_crossentropy(y_pred=ypred,y_true=y)
                elif lossfunc == "CCE":
                    loss = lf.categorical_crossentropy(y_pred=ypred,y_true=y)
                else:
                    raise ValueError("Invalid loss function")
                
                batch_losses.append(loss.data)

                # zero gradients sebelum backward
                progress_bar.set_description(f"Epoch {i+1}/{epoch} - Backpropagating")
                self.zero_grad()

                # NN
                loss.backward()

        

                for p in self.parameters():
                    #W + -lr*deltaW
                    p.data += -1 *learning_rate * p.grad

            #  rata-rata loss per epoch
            avg_loss = np.mean(batch_losses)
            self.trainloss.append(avg_loss)

            # === VALIDATION LOSS ===
            val_loss = None
            if x_val is not None and y_val is not None:
                val_pred = [self(x_input_forward) for x_input_forward in x_val]
                if lossfunc == "MSE":
                    val_loss = lf.mean_squared_error(y_true=y_val,y_pred=val_pred)
                elif lossfunc == "BCE":
                    val_loss = lf.binary_crossentropy(y_true=y_val,y_pred=val_pred)
                elif lossfunc == "CCE":
                    val_loss = lf.categorical_crossentropy(y_true=y_val,y_pred=val_pred)
                else:
                    raise ValueError("Invalid loss function")
                
                self.validloss.append(val_loss.data)

            if val_loss is not None:
                progress_bar.set_postfix({"Train Loss": avg_loss, "Val Loss": val_loss.data})
            else:
                progress_bar.set_postfix({"Train Loss": avg_loss})

        return self

    def save(self, filepath):
        model_state = {
            'input_size': self.inputlayer,
            'layer_sizes': [len(layer.neurons) for layer in self.layers],
            'activations': [layer.activation for layer in self.layers],
            'weights': [[[w.data for w in neuron.w] for neuron in layer.neurons] for layer in self.layers],
            'biases': [[neuron.b.data for neuron in layer.neurons] for layer in self.layers],
            'weight_init': {
                'method': self.weight.method,
                'seed': self.weight.seed,
                'lower': self.weight.lower,
                'upper': self.weight.upper,
                'mean': self.weight.mean,
                'variance': self.weight.variance
            },
            'bias_init': {
                'method': self.biasW.method,
                'seed': self.biasW.seed,
                'lower': self.biasW.lower,
                'upper': self.biasW.upper,
                'mean': self.biasW.mean,
                'variance': self.biasW.variance
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
    
        seed_weight = model_state['weight_init']['seed']
        seed_bias = model_state['bias_init']['seed']
        
        weight_init = Weight(
            model_state['weight_init']['method'],
            seed_weight,
            model_state['input_size'],
            lower=model_state['weight_init']['lower'],
            upper=model_state['weight_init']['upper'],
            mean=model_state['weight_init']['mean'],
            variance=model_state['weight_init']['variance']
        )
        
        bias_init = Weight(
            model_state['bias_init']['method'],
            seed_bias,
            model_state['input_size'],
            lower=model_state['bias_init']['lower'],
            upper=model_state['bias_init']['upper'],
            mean=model_state['bias_init']['mean'],
            variance=model_state['bias_init']['variance']
        )
        
        # Buat ulang model
        model = cls(
            model_state['input_size'],
            model_state['layer_sizes'],
            activations=model_state['activations'],
            weight=weight_init,
            biasW=bias_init
        )
        
        # Restore weights dan bias
        for layer_idx, layer in enumerate(model.layers):
            for neuron_idx, neuron in enumerate(layer.neurons):
                # Restore weights list
                for w_idx, w_value in enumerate(model_state['weights'][layer_idx][neuron_idx]):
                    neuron.w[w_idx] = Value(w_value)
                # Restore bias
                neuron.b = Value(model_state['biases'][layer_idx][neuron_idx])
        
        return model

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def show_grad_distribution(self, x=None):
        if x is not None:
            print(f"\nLayer {x + 1}:")
            for idx, neuron in enumerate(self.layers[x].neurons):
                delta_w = " | ".join(f"ΔW{i+1}={w.grad:.4f}" for i, w in enumerate(neuron.w))  
                print(f"  ΔW di Neuron H{idx + 1}: {delta_w}")
            delta_wb = self.layers[x].neurons[0].b.grad  
            print(f"  ΔWb: {delta_wb:.4f}")
        else:
            for layer_idx, layer in enumerate(self.layers):
                print(f"\n=== Layer {layer_idx + 1} ===")
                for neuron_idx, neuron in enumerate(layer.neurons):
                    delta_w = " | ".join(f"ΔW{i+1}={w.grad:.4f}" for i, w in enumerate(neuron.w))
                    print(f"  ΔW di Neuron H{neuron_idx + 1}: {delta_w}")
                delta_wb = layer.neurons[0].b.grad  
                print(f"  ΔWb: {delta_wb:.4f}")


    def plot_grad_distribution(self, layers=None):
        all_gradients = []
        plt.figure(figsize=(8, 5))
        if layers is not None:
            plt.title(f"Distribusi ΔW pada Layer {layers}")
            for neuron in self.layers[layers].neurons:
                all_gradients.extend([w.grad for w in neuron.w])
        else:
            plt.title("Distribusi ΔW di semua layer")
            for layer in self.layers:
                for neuron in layer.neurons:
                    all_gradients.extend([w.grad for w in neuron.w])  

        all_gradients = np.array(all_gradients)

        sns.histplot(all_gradients, bins=10, kde=True, color="blue")  
        plt.xlabel("ΔW Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    def show_W_distribution(self, x=None):
        if x is not None:
            print(f"\nLayer {x + 1}:")
            for idx, neuron in enumerate(self.layers[x].neurons):
                weights = " | ".join(f"W{i+1}={w.data:.4f}" for i, w in enumerate(neuron.w))  
            bias = self.layers[x].neurons[0].b.data  
            print(f"  Wb: {bias:.4f}")
        else:
            for layer_idx, layer in enumerate(self.layers):
                print(f"\n=== Layer {layer_idx + 1} ===")
                for neuron_idx, neuron in enumerate(layer.neurons):
                    weights = " | ".join(f"W{i+1}={w.data:.4f}" for i, w in enumerate(neuron.w))
                    print(f"  W di Neuron H{neuron_idx + 1}: {weights}")
                bias = layer.neurons[0].b.data  
                print(f"  Wb: {bias:.4f}")


    def plot_W_distribution(self,layers=None):
        all_weights = []
        plt.figure(figsize=(8, 5))
        if layers is not None:
            plt.title(f"Distribusi bobot pada Layer {layers}")
            for neuron in self.layers[layers].neurons:
                all_weights.extend([w.data for w in neuron.w])
                
        else:
            plt.title("Distribusi bobot di semua layer")
            for layer in self.layers:
                for neuron in layer.neurons:
                    all_weights.extend([w.data for w in neuron.w])  

        all_weights = np.array(all_weights)

        sns.histplot(all_weights, bins=10, kde=True, color="blue")  
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    


    #pake fungsi ini kalo mau dapet info predict pakai W setelah fit
    def predict(self,x,showinfo = False):
        out =  [self(x_input_forward) for x_input_forward in x]
        if showinfo :
            for y_batch in out:
                y_list = [y_batch] if isinstance(y_batch, Value) else y_batch  #isinstance handle single neuron
                for i, y in enumerate(y_list):
                    print(f"Y{i}: {y.data}", end=" ")
                print()
        return out



    def plot_loss(self):
        plt.plot(self.trainloss, label="Training Loss",color = "red")
        plt.plot(self.validloss,label = "Validation Loss",color = "blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()






    

        
    def __repr__(self):
        return f"inputx : {self.inputlayer} MLP of [{', '.join(str(layer) for layer in self.layers)}]"
