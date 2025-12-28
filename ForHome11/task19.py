import numpy as np

import graphviz
import numbers

class Value:
    def __init__(self, data, name, prev=[], op = None, grad = 0, backward=0):
        self.data = data
        self._prev = prev
        self._op = op 
        self.grad = grad
        self.name = name
        self._backward = backward

    def __str__(self):
        return f"{self.name}=Value(data={self.data})"
    
    def __repr__(self):
        return f"{self.name}=Value(data={self.data})"
    
    def __add__(self, other):
        if isinstance(other, numbers.Number):
            other = Value(other, 'temp')
        return Value(self.data + other.data, name="(" + self.name + " + " + other.name + ")", prev = [self, other], op = '+')
    
    def __radd__(self, other):
        if isinstance(other, numbers.Number):
            other = Value(other, 'temp')
        return Value(self.data + other.data, name="(" + self.name + " + " + other.name + ")", prev = [self, other], op = '+')
    

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            other = Value(other, 'temp')
        return Value(self.data * other.data, name = self.name + other.name, prev = [self, other], op = '*')
    
    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            other = Value(other, 'temp')
        return Value(self.data * other.data, name = self.name + other.name, prev = [self, other], op = '*')
    

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            other = Value(other, 'temp')
        return self * (other ** (-1))

    def exp(self):
        return Value(np.exp(self.data), name = "e**" + self.name, prev = [self], op = 'exp')
    
    def __pow__(self, other):
        if isinstance(other, numbers.Number):
            other = Value(other, 'temp')
        return Value(self.data ** other.data, name=self.name + "**" + other.name, prev=[self, other], op = '**')
    

    def tanh(self):
        x = self
        # return Value((np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)), name="tanh(" + self.name + ")", prev = set([self]), op = 'tanh')
        return (x.exp() + (-1)*((-1)*x).exp())/(x.exp() + ((-1)*x).exp())
        

    def top_sort(self):

        def get_all_nodes(node):
            result = []
            stack = [node]
            visited = set()

            while len(stack) > 0:
                popped = stack.pop()

                if popped in visited:
                    continue

                visited.add(popped)
                result.append(popped)
                for prev in popped._prev:
                    stack.append(prev)
            return result

        def dependencies_are_removed(removed, dependencies):
            for dependency in dependencies:
                if dependency not in removed:
                    return False
            return True

        l = get_all_nodes(self)
        result = []
        removed = set()
        while len(l) > 0:
            to_remove = []
            for element in l:
                if dependencies_are_removed(removed, element._prev):
                    result.append(element)
                    removed.add(element)
                    to_remove.append(element)
            for element in to_remove:
                l.remove(element)
            
        return result

    


    def backward(self):
        #z = x + y -> dz/dx = 1 && dz/dy = 1
        l = self.top_sort()

        #We are assuming there is only one last
        #node and it is the loss function.

        last = l[-1]
        #dL/dL = 1
        last._backward = 1

        while len(l) > 0:
            last = l.pop()
            last.grad = last._backward

            for index in range(len(last._prev)):
                if last._op == '+':
                    last._prev[index]._backward += last.grad * 1
                elif last._op == '*':
                    last._prev[index]._backward += last.grad * last._prev[index-1].data

                elif last._op == 'tanh':
                    x = last._prev[index].data
                    last._prev[index]._backward += last.grad * (4/(np.exp(x) + np.exp(-x))**2)

                elif last._op == '**':
                    # last = prev[0] ** prev[1]
                    if index == 0:
                        last._prev[index]._backward += last.grad * (last._prev[1].data * (last._prev[0].data ** (last._prev[1].data - 1)))
                    elif index == 1:
                        last._prev[index]._backward += last.grad * np.log(last._prev[0].data) * (last._prev[0].data ** last._prev[1].data)

                elif last._op == 'exp':
                    last._prev[index]._backward += last.grad * np.exp(last._prev[index].data)


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
    dot = graphviz.Digraph(filename='06_result', format='svg', graph_attr={
                           'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        try:
            
            dot.node(name=uid,
                    label=f'{{ {n.name} | data: {n.data:.4f} | grad: {n.grad:.4f} }}',
                    shape='record')
        except:
            # print(f'{{ {n.name} | data: {n.data:.4f} | grad: {n.grad:.4f} }}')
            print(n.name)
            print(n.data)
            print(n.grad)
            return
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

        return L
    

def main() -> None:
    p = Perceptron(-3, 1, 6.8813735870195432)
    l = p.forward(2, 0)

    l.backward()

    draw_dot(l).render(directory='./graphviz_output', view=True)


if __name__ == '__main__':
    main()