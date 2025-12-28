import graphviz

class Value:
    def __init__(self, data, prev=set(), op = None, grad = 0):
        self.data = data
        self._prev = prev
        self._op = op 
        self.grad = grad

    def __str__(self):
        return f"Value(data={self.data})"
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        return Value(self.data + other.data, prev = set((self, other)), op = '+')
    
    def __mul__(self, other):
        return Value(self.data * other.data, prev = set((self, other)), op = '*')
    

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
    dot = graphviz.Digraph(filename='01_result', format='svg', graph_attr={
                           'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    node_name = 'a'
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        dot.node(name=uid,
                label=f'{{ {node_name} | data: {n.data:.4f} | grad: {n.grad:.4f} }}',
                shape='record')
        if n._op:
            # if this value is a result of some operation, create an "op" node for the operation
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to the node of the operation
            dot.edge(uid + n._op, uid)

        node_name = chr(ord(node_name) + 1)

    for n1, n2 in edges:
        # connect n1 to the "op" node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

def manual_der(a,b,c,d,e,f,L):
    # L = L -> dL/dL = 1
    L.grad = 1

    # L = f * d -> dL/df = d 
    f.grad = d.data

    # L = f * d -> dL/dd = f
    d.grad = f.data

    # d = e + c -> dd/de = 1 -> dL/de = dL/dd * dd/de
    e.grad = d.grad * 1

    # d = e + c -> dd/dc = 1 -> dL/dc = dL/dd * dd/dc
    c.grad = d.grad * 1

    #e = a * b -> de/db = a -> dL/db = dL/de * de/db
    b.grad = e.grad * a.data

    #e = a * b -> de/da = b -> dL/da = dL/de * de/da
    a.grad = e.grad * b.data

    return a,b,c,d,e,f,L

def main() -> None:
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    e = a*b
    d = e+c
    f = Value(-2.0) 
    L = f * d

    a,b,c,d,e,f,L = manual_der(a,b,c,d,e,f,L)

    print(f"Old L = {L}")

    a.data += -a.grad * 1e-2
    b.data += -b.grad * 1e-2
    c.data += -c.grad * 1e-2
    f.data += -f.grad * 1e-2

    e = a*b
    d = e+c
    L = f * d


    
    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    # draw_dot(L).render(directory='./graphviz_output', view=True)

    print(f"New L = {L}")
    #In the example it becomes -7.28 but I think
    #it's incorrect because L should become smaller
    #and not smaller by absolute value

if __name__ == '__main__':
    main()