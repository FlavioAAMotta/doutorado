# genetic_programming_classifier.py

import random
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from node import VariableNode, ConstantNode, OperatorNode
from generate import generate_random_expression, crossover, mutate
from cost_calculation import calculate_costs  # Certifique-se de importar a função
import pandas as pd

class GeneticProgrammingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, population_size=50, generations=100, max_depth=5, random_state=None):
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.random_state = random_state
        self.best_individual_ = None
        self.variables_ = None

    def fit(self, X, y, vol_bytes, acc_fut):
        # Definir a semente aleatória
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        # Armazenar os nomes das variáveis
        self.variables_ = X.columns.tolist()

        # Combinar os dados em um único DataFrame
        data = X.copy()
        data['label'] = y
        data['vol_bytes'] = vol_bytes
        data['acc_fut'] = acc_fut

        # Executar o algoritmo de programação genética
        self.best_individual_ = self._genetic_programming(
            data=data,
            variables=self.variables_
        )
        return self

    def predict(self, X):
        predictions = []
        for data_point in X.values:
            variables = dict(zip(self.variables_, data_point))
            result = self.best_individual_.evaluate(variables)
            prediction = int(bool(result))
            predictions.append(prediction)
        return np.array(predictions)

    def predict_proba(self, X):
        proba = []
        for data_point in X.values:
            variables = dict(zip(self.variables_, data_point))
            result = self.best_individual_.evaluate(variables)
            probability = float(bool(result))
            proba.append([1 - probability, probability])
        return np.array(proba)

    def _genetic_programming(self, data, variables):
        # Inicializa a população
        population = [generate_random_expression(self.max_depth, variables) for _ in range(self.population_size)]

        for generation in range(self.generations):
            # Avalia o fitness de cada indivíduo
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_individual(individual, data, variables)
                fitness_scores.append((fitness, individual))

            # Ordena a população com base no fitness
            fitness_scores.sort(reverse=True, key=lambda x: x[0])

            best_fitness = fitness_scores[0][0]
            best_individual = fitness_scores[0][1]
            print(f"Geração {generation}: Melhor fitness = {best_fitness}")
            print(f"Melhor indivíduo: {best_individual}")

            # Seleciona os melhores indivíduos
            survivors = [individual for (fitness, individual) in fitness_scores[:self.population_size // 2]]

            # Gera nova população através de cruzamento e mutação
            new_population = survivors.copy()
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(survivors, 2)
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1, self.max_depth, variables)
                child2 = mutate(child2, self.max_depth, variables)
                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

        # Retorna o melhor indivíduo
        best_individual = fitness_scores[0][1]
        return best_individual

    def _evaluate_individual(self, individual, data, variables):
        # Avaliar o indivíduo em todos os dados
        df = data.copy()
        predictions = []
        for index, row in df.iterrows():
            vars_dict = row[variables].to_dict()
            try:
                result = individual.evaluate(vars_dict)
                prediction = int(bool(result))
            except Exception:
                prediction = 0  # Penaliza o indivíduo em caso de erro
            predictions.append(prediction)
        df['pred'] = predictions

        # Calcular o custo usando a função 'calculate_costs'
        costs = calculate_costs(df)
        cost_ml = costs['cost ml']  # Extrair o custo total das predições do modelo

        # Definir o fitness como o inverso do custo (adicionar epsilon para evitar divisão por zero)
        epsilon = 1e-6
        fitness = 1 / (cost_ml + epsilon)
        return fitness