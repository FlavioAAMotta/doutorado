# node.py

class Node:
    def evaluate(self, variables):
        pass

    def copy(self):
        pass

class VariableNode(Node):
    def __init__(self, name):
        self.name = name

    def evaluate(self, variables):
        return variables.get(self.name, 0)  # Returns 0 if the variable is not found

    def copy(self):
        return VariableNode(self.name)

    def __str__(self):
        return self.name

class ConstantNode(Node):
    def __init__(self, value):
        self.value = value

    def evaluate(self, variables):
        return self.value

    def copy(self):
        return ConstantNode(self.value)

    def __str__(self):
        return str(self.value)

class OperatorNode(Node):
    def __init__(self, operator, left, right=None):
        self.operator = operator
        self.left = left  # Could be a Node
        self.right = right  # Could be a Node or None

    def evaluate(self, variables):
        if self.operator == 'not':
            return not self.left.evaluate(variables)
        else:
            left_val = self.left.evaluate(variables)
            right_val = self.right.evaluate(variables)
            try:
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
                    return left_val >= right_val  # This should be left_val >= right_val
                else:
                    raise ValueError(f"Unknown operator: {self.operator}")
            except Exception as e:
                print(f"Error in evaluate: {e}")
                return 0  # Return 0 in case of error

    def copy(self):
        return OperatorNode(
            self.operator,
            self.left.copy(),
            self.right.copy() if self.right else None
        )

    def __str__(self):
        if self.operator == 'not':
            return f"(not {self.left})"
        else:
            return f"({self.left} {self.operator} {self.right})"
