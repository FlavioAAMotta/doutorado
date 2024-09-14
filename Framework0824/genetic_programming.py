import random
import pandas as pd
from grammar import get_expression
from asteval import Interpreter

random.seed(42)

aeval = Interpreter()

def readArqs(pop):
    PATH = "../../eval/"
    name_space = []
    arq = open(PATH + "nsJanelas_" + pop + ".txt", "r")
    for line in arq:
        name_space.append(sorted(list(map(int, line.split()))))
    arqPath = lambda name: f"{PATH}/{name}_{pop}.txt"
    arq_acessos = pd.read_csv(
        arqPath("access"), low_memory=False, sep=" ", index_col="NameSpace"
    )
    arq_classes = pd.read_csv(
        arqPath("target"), low_memory=False, sep=" ", index_col="NameSpace"
    )
    arq_vol_bytes = pd.read_csv(
        arqPath("vol_bytes"), low_memory=False, sep=" ", index_col="NameSpace"
    )
    return arq_acessos, arq_classes, arq_vol_bytes

def get_fitness(population, week, set):
    fitness = []
    for individual in population:
        individual_fitness = evaluate_individual(individual, week, set)
        if individual_fitness < 0:  # Evita fitness negativos devido a overflow
            individual_fitness = float("inf")
        fitness.append(individual_fitness)
    return fitness

def get_final_fitness(individual, week, set):
    individual_fitness = evaluate_individual(individual, week, set)
    if individual_fitness < 0:
        individual_fitness = float("inf")
    return individual_fitness

def get_number_in_string(string):
    number = ""
    for char in string:
        if char.isdigit():
            number += char
    return int(number)

def evaluate_individual(raw_individual, week, set):
    fitness = 0
    for i, key in enumerate(set.keys()):
        individual_index = get_number_in_string(key)
        individual = get_variable_values(raw_individual, set, individual_index)
        individual = remove_invalid_operations(individual)
        individual = remove_leading_zeros(individual)
        individual_value = abs(execute_expression(individual))
        fitness += (individual_value - set['label' + str(individual_index)]) ** 2
    return fitness / len(set)

def remove_leading_zeros(individual):
    while " 0" in individual:
        individual = individual.replace(" 0", " ")
    return individual

def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def execute_expression(individual):
    try:
        return aeval(individual)
    except:
        return float("inf")  # Penaliza expressões inválidas

def remove_invalid_operations(individual):
    individual = individual.replace("/0", "/1").replace("%0", "%1").replace("^0", "^1")
    return individual

def get_variable_values(individual, training_set, individual_index):
    variables = {'x': "x", 'y': "y", 'z': "z", 'w': "w"}
    for var in variables:
        individual = individual.replace(var, str(training_set[var + str(individual_index)]))
    return individual

def generate_random_population(populationSize):
    return [get_expression(random.randint(2, 4)) for _ in range(populationSize)]

def mutate(population, week, access):
    new_population = []
    for raw_individual in population:
        while True:
            if random.randint(0, 100) > 50:
                new_population.append(raw_individual)
                break
            individual = list(raw_individual)
            if len(individual) == 1:
                new_population.append(get_expression(3))
                break
            mutate_point = random.randint(1, len(individual) - 1)
            individual[mutate_point] = get_expression(1)
            individual = "".join(individual)
            if check_expression(individual, week, access):
                new_population.append(individual)
                break
    return new_population

def crossover(population, week, access):
    new_population = []
    for i in range(0, len(population), 2):
        individual1 = population[i]
        individual2 = population[i + 1] if i + 1 < len(population) else population[i]
        new_population.extend([individual1, individual2])
        if len(individual1) > 1 and len(individual2) > 1:
            sub_expression1 = get_sub_expression(individual1, week, access)
            sub_expression2 = get_sub_expression(individual2, week, access)
            new_population.append(individual1.replace(sub_expression1, sub_expression2))
            new_population.append(individual2.replace(sub_expression2, sub_expression1))
    return new_population

def get_sub_expression(individual, week, access):
    if len(individual) == 1:
        return individual
    for _ in range(50):
        start_point = random.randint(1, len(individual) - 1)
        sub_expression_size = random.randint(1, len(individual) - start_point)
        sub_expression = individual[start_point:start_point + sub_expression_size]
        if check_expression(sub_expression, week, access):
            return sub_expression
    return individual

def check_expression(individual, week, access):
    try:
        if not check_letters_and_numbers(individual):
            return False
        individual = get_variable_values(individual, access, 0)
        individual = remove_invalid_operations(individual)
        individual = remove_leading_zeros(individual)
        execute_expression(individual)
        return True
    except:
        return False

def check_letters_and_numbers(individual):
    for i in range(len(individual) - 1):
        if is_number(individual[i]) and is_variable(individual[i + 1]):
            return False
        if is_variable(individual[i]) and (is_number(individual[i + 1]) or is_variable(individual[i + 1])):
            return False
    return True

def is_variable(string):
    return string in ["x", "y", "z", "w"]

def select(population, fitness, tournament_size=3):
    new_population = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitness)), tournament_size)
        best_individual = min(tournament, key=lambda ind_fit: ind_fit[1])
        new_population.append(best_individual[0])
    return new_population

def genetic_algorithm(populationSize, generations, week, access):
    population = generate_random_population(populationSize)
    fitness = get_fitness(population, week, access)
    print(f"Generation 0 best fitness: {min(fitness)} individual: {population[fitness.index(min(fitness))]}")
    for i in range(generations):
        population = select(population, fitness)
        population = crossover(population, week, access)
        population = mutate(population, week, access)
        fitness = get_fitness(population, week, access)
        print(f"Generation {i+1} best fitness: {min(fitness)} individual: {population[fitness.index(min(fitness))]}")
    
    validation_fitness = get_fitness(population, week, access)
    best_individual = population[validation_fitness.index(min(validation_fitness))]
    print(f"Best individual: {best_individual}")

    test_fitness = get_final_fitness(best_individual, week, access)
    print(f"Test fitness: {test_fitness}")