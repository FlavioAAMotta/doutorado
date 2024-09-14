# node.py

import random

class Node:
    def evaluate(self, variables):
        pass

    def mutate(self, max_depth):
        pass

    def copy(self):
        pass

class VariableNode(Node):
    def __init__(self, name):
        self.name = name

    def evaluate(self, variables):
        return variables.get(self.name, 0)  # Retorna 0 se a variável não estiver definida

    def mutate(self, max_depth):
        if random.random() < 0.1:
            return random.choice([
                VariableNode(random.choice(['x', 'y', 'z', 'w'])),
                ConstantNode(random.randint(1, 100))
            ])
        else:
            return self.copy()

    def copy(self):
        return VariableNode(self.name)

class ConstantNode(Node):
    def __init__(self, value):
        self.value = value

    def evaluate(self, variables):
        return self.value

    def mutate(self, max_depth):
        if random.random() < 0.1:
            return ConstantNode(random.randint(1, 100))
        else:
            return self.copy()

    def copy(self):
        return ConstantNode(self.value)

class OperatorNode(Node):
    def __init__(self, operator, left, right=None):
        self.operator = operator
        self.left = left  # Pode ser Node
        self.right = right  # Pode ser Node ou None

    def evaluate(self, variables):
        if self.operator == 'not':
            return not self.left.evaluate(variables)
        else:
            left_val = self.left.evaluate(variables)
            right_val = self.right.evaluate(variables)
            if self.operator == 'and':
                return left_val and right_val
            elif self.operator == 'or':
                return left_val or right_val
            elif self.operator == '==':
                return left_val == right_val
            elif self.operator == '!=':
                return left_val != right_val
            elif self.operator == '<':
                return left_val < right_val
            elif self.operator == '>':
                return left_val > right_val
            elif self.operator == '<=':
                return left_val <= right_val
            elif self.operator == '>=':
                return left_val >= right_val
            else:
                raise ValueError(f"Operador desconhecido: {self.operator}")

    def mutate(self, max_depth):
        if random.random() < 0.1:
            return generate_random_expression(max_depth)
        else:
            new_left = self.left.mutate(max_depth - 1)
            new_right = self.right.mutate(max_depth - 1) if self.right else None
            return OperatorNode(self.operator, new_left, new_right)

    def copy(self):
        return OperatorNode(
            self.operator,
            self.left.copy(),
            self.right.copy() if self.right else None
        )
