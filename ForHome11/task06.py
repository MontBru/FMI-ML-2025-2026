class Value:
    def __init__(self, data, prev=set(), op = None):
        self.data = data
        self._prev = prev
        self._op = op 

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


def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    
    nodes, edges = trace(x)
    print('x')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(y)
    print('y')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(z)
    print('z')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(result)
    print('result')
    print(f'{nodes=}')
    print(f'{edges=}')

if __name__ == '__main__':
    main()