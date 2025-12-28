import numpy as np

import graphviz

class Value:
    def __init__(self, data, name, prev=set(), op = None, grad = 0):
        self.data = data
        self._prev = prev
        self._op = op 
        self.grad = grad
        self.name = name

    def __str__(self):
        return f"{self.name}=Value(data={self.data})"
    
    def __repr__(self):
        return f"{self.name}=Value(data={self.data})"
    
    def __add__(self, other):
        return Value(self.data + other.data, name="(" + self.name + " + " + other.name + ")", prev = set((self, other)), op = '+')
    
    def __mul__(self, other):
        return Value(self.data * other.data, name = self.name + other.name, prev = set((self, other)), op = '*')
    
    def tanh(self):
        x = self.data
        return Value((np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)), name="tanh(" + self.name + ")", prev = set([self]), op = 'tanh')

def trace(val):
    if val._op == None:
        return set([val]), set()
    
    nodes = set([val])
    edges = set((prev, val) for prev in val._prev)
    

    for prev in val._prev:
        prev_nodes, prev_edges = trace(prev)
        nodes.update(prev_nodes)
        edges.update(prev_edges)
    
    return nodes, edges

def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(filename='04_result', format='svg', graph_attr={
                           'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        dot.node(name=uid,
                label=f'{{ {n.name} | data: {n.data:.4f} | grad: {n.grad:.4f} }}',
                shape='record')
        if n._op:
            # if this value is a result of some operation, create an "op" node for the operation
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to the node of the operation
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the "op" node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

class Perceptron:
    def __init__(self, w1, w2, b):
        self.w1 = Value(w1, "w1")
        self.w2 = Value(w2, "w2")
        self.b = Value(b, "b")

    def forward(self, x1, x2):
        x1 = Value(x1, 'x1')
        x2 = Value(x2, 'x2')

        x1w1 = x1*self.w1
        x2w2 = x2*self.w2
        added = x1w1 + x2w2
        logit = added + self.b
        logit.name = 'logit'

        L = logit.tanh()
        L.name = 'L'

        #Backward
        L.grad = 1
        
        #the derivative of tanh with plugged in logit.data as x
        logit.grad = 4/(np.exp(logit.data) + np.exp(-logit.data))**2

        #logit = b + added => dlogit/db = 1, dL/db = dlogit/db * dL/dlogit
        self.b.grad = logit.grad
        #same as b
        added.grad = logit.grad

        #added = x1w1 + x2w2 => dadded/dx1w1 = 1
        x1w1.grad = added.grad
        x2w2.grad = added.grad

        #x1w1 = x1*w1
        x1.grad = self.w1.data * x1w1.grad
        self.w1.grad = x1.data * x1w1.grad

        #x2w2 = x2*w2
        x2.grad = self.w2.data * x2w2.grad
        self.w2.grad = x2.data * x2w2.grad



        return L
    

def main():
    p = Perceptron(-3, 1, 6.8813735870195432)
    l = p.forward(2, 0)
    draw_dot(l).render(directory='./graphviz_output', view=True)


if __name__ == '__main__':
    main()