import math as math


class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += (out.grad * 1.0)  #rumus chain rule turunan
            other.grad += (out.grad * 1.0)  #rumus chain rule turunan
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad  #rumus chain rule turunan
            other.grad += self.data * out.grad  #rumus chain rule turunan
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad  #rumus chain rule turunan
        out._backward = _backward

        return out

     #aktivasi relu
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')


        #Turunan ReLU
        def _backward():
            self.grad += (1 if self.data > 0 else 0) * out.grad
        out._backward = _backward
        return out


    #aktivasi tanh
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')


        #Turunan tanh
        def _backward():
          self.grad += (1-t**2) * out.grad #rumus chain rule turunan

        out._backward = _backward
        return out
    
    def sigmoid(self):
        x = self.data
        s = 1/(1+math.exp(-x))
        out = Value(s, (self,), 'sigmoid')

        #Turunan sigmoid
        def _backward():        
            self.grad += (out.data * (1 - out.data)) * out.grad
        out._backward = _backward
        return out
    
    @staticmethod
    def softmax(values):
        # ambil max val untuk stabilitas numerik (tidak overflow)
        max_val = max(v.data for v in values)
        exps = [Value(math.exp(v.data - max_val), (v,), 'exp') for v in values]
         
        sum_exp = sum(exp.data for exp in exps)
        
        # hitung output softmax
        softmax_outputs = []
        for i, exp in enumerate(exps):
            #exp(x_i) / Σexp(x_j)
            out = Value(exp.data / sum_exp, (exp,) + tuple(values), 'softmax')
            
            def _backward(i=i, exp=exp):
                def backward():
                    for j, v in enumerate(values):
                        if j == i:  # ketika input yang produce output
                            # softmax_i * (1 - softmax_i)
                            v.grad += out.data * (1 - out.data) * out.grad
                        else:  # ketika input yang tidak produce output
                            #-softmax_i * softmax_j
                            v.grad += -out.data * exps[j].data / sum_exp * out.grad
                return backward
            
            out._backward = _backward(i, exp)
            softmax_outputs.append(out)
        
        return softmax_outputs



    def backward(self):
        # topological sort buat backward propagation
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()



    #aritmatika primitif

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def log(self, base=math.e):
        epsilon = 1e-7
        self.data = max(epsilon, self.data)
        out = Value(math.log(self.data, base), (self,), 'log')
        
        def _backward():
            self.grad += (1 / (self.data * math.log(base))) * out.grad
        out._backward = _backward
        
        return out