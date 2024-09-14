# generate.py

import random
from node import VariableNode, ConstantNode, OperatorNode

def generate_random_expression(max_depth):
    if max_depth == 0:
        return generate_terminal()
    else:
        choice = random.randint(1, 3)
        if choice == 1:
            return generate_terminal()
        elif choice == 2:
            return generate_unary_operator(max_depth)
        else:
            return generate_binary_operator(max_depth)

def generate_terminal():
    if random.choice([True, False]):
        return VariableNode(random.choice(['x', 'y', 'z', 'w']))
    else:
        return ConstantNode(random.randint(1, 100))

def generate_unary_operator(max_depth):
    operand = generate_random_expression(max_depth - 1)
    return OperatorNode('not', operand)

def generate_binary_operator(max_depth):
    operator = random.choice(['and', 'or', '==', '!=', '<', '>', '<=', '>='])
    left = generate_random_expression(max_depth - 1)
    right = generate_random_expression(max_depth - 1)
    return OperatorNode(operator, left, right)

def crossover(parent1, parent2):
    parent1_copy = parent1.copy()
    parent2_copy = parent2.copy()

    nodes1 = get_all_nodes(parent1_copy)
    nodes2 = get_all_nodes(parent2_copy)

    node1, parent1_of_node1, is_left_child1 = random.choice(nodes1)
    node2, parent2_of_node2, is_left_child2 = random.choice(nodes2)

    swap_nodes(node1, parent1_of_node1, is_left_child1, node2, parent2_of_node2, is_left_child2)

    return parent1_copy, parent2_copy

def get_all_nodes(node, parent=None, is_left_child=None):
    nodes = [(node, parent, is_left_child)]
    if isinstance(node, OperatorNode):
        nodes.extend(get_all_nodes(node.left, node, 'left'))
        if node.right:
            nodes.extend(get_all_nodes(node.right, node, 'right'))
    return nodes

def swap_nodes(node1, parent1, is_left_child1, node2, parent2, is_left_child2):
    # Substitui node1 em parent1 por node2, e vice-versa
    if parent1 is None:
        # node1 é a raiz da árvore parent1_copy
        temp = node1.copy()
        node1.__dict__.update(node2.copy().__dict__)
        node1.__class__ = node2.__class__
    else:
        if is_left_child1 == 'left':
            parent1.left = node2.copy()
        elif is_left_child1 == 'right':
            parent1.right = node2.copy()

    if parent2 is None:
        # node2 é a raiz da árvore parent2_copy
        temp = node2.copy()
        node2.__dict__.update(node1.copy().__dict__)
        node2.__class__ = node1.__class__
    else:
        if is_left_child2 == 'left':
            parent2.left = node1.copy()
        elif is_left_child2 == 'right':
            parent2.right = node1.copy()

def mutate(individual, max_depth):
    if random.random() < 0.1:
        # Substituir o indivíduo inteiro por uma nova expressão
        return generate_random_expression(max_depth)
    else:
        # Recursivamente tentar mutar subnós
        if isinstance(individual, OperatorNode):
            individual.left = mutate(individual.left, max_depth - 1)
            if individual.right:
                individual.right = mutate(individual.right, max_depth - 1)
        elif isinstance(individual, VariableNode) or isinstance(individual, ConstantNode):
            # Possibilidade de mutar um terminal
            if random.random() < 0.1:
                return generate_random_expression(max_depth)
        return individual