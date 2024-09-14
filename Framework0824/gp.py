# gp.py

import random
from node import VariableNode, ConstantNode, OperatorNode
from generate import generate_random_expression, crossover, mutate

def evaluate_individual(individual, variables):
    try:
        result = individual.evaluate(variables)
        return int(bool(result))  # Converte True/False para 1/0
    except Exception:
        return 0  # Atribui fitness 0 em caso de erro

def print_expression(node):
    if isinstance(node, VariableNode):
        return node.name
    elif isinstance(node, ConstantNode):
        return str(node.value)
    elif isinstance(node, OperatorNode):
        if node.operator == 'not':
            return f"not ({print_expression(node.left)})"
        else:
            left_expr = print_expression(node.left)
            right_expr = print_expression(node.right)
            return f"({left_expr} {node.operator} {right_expr})"
    else:
        return ""

def genetic_programming(population_size, generations, max_depth, variables):
    # Inicializa a população
    population = [generate_random_expression(max_depth) for _ in range(population_size)]

    for generation in range(generations):
        # Avalia os indivíduos
        fitness_scores = []
        for individual in population:
            fitness = evaluate_individual(individual, variables)
            fitness_scores.append((fitness, individual))

        # Ordena a população com base no fitness
        fitness_scores.sort(reverse=True, key=lambda x: x[0])

        print(f"Geração {generation}: Melhor fitness = {fitness_scores[0][0]}")

        # Seleção: seleciona a metade superior da população
        survivors = [individual for (fitness, individual) in fitness_scores[:population_size // 2]]

        # Gera novos indivíduos por meio de cruzamento e mutação
        new_population = survivors.copy()
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(survivors, 2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, max_depth)
            child2 = mutate(child2, max_depth)
            new_population.extend([child1, child2])

        # Trunca para o tamanho da população
        population = new_population[:population_size]

    # Retorna o melhor indivíduo após a evolução
    best_individual = fitness_scores[0][1]
    return best_individual

if __name__ == "__main__":
    variables = {'x': 5, 'y': 10, 'z': 3, 'w': 7}
    population_size = 10
    generations = 5
    max_depth = 4

    best = genetic_programming(population_size, generations, max_depth, variables)
    print("Melhor indivíduo:")
    print(print_expression(best))
    print("Avaliação com variáveis:", evaluate_individual(best, variables))
