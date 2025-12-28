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
    dot = graphviz.Digraph(filename='02_result', format='svg', graph_attr={
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

        logit = (x1*self.w1 + x2*self.w2 + self.b)
        logit.name = 'logit'
        return logit
    

def main():
    p = Perceptron(-3, 1, 6.7)
    l = p.forward(2, 0)
    draw_dot(l).render(directory='./graphviz_output', view=True)


if __name__ == '__main__':
    main()