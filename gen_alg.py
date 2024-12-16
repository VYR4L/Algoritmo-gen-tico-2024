import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class GeneticAlgorithm:
    '''
    Classe que implementa um algoritmo genético para otimização de funções
    de múltiplas variáveis.

    '''
    # Construtor para inicialização de variáveis
    def __init__(self, max_generations, population_size, base_function, mutation_rate):
        '''
        Construtor da classe GeneticAlgorithm.

        :param max_generations: Número máximo de gerações.
        :param population_size: Tamanho da população.
        :param base_function: Função a ser otimizada.
        :param mutation_rate: Taxa de mutação.

        :return: None
        '''
        self.max_generations:int = max_generations
        self.population_size:int = population_size
        self.base_function = base_function
        self.mutation_rate = mutation_rate
        self.elitism_rate = self.population_size * 0.005

    def box_muller(self, mean, std):
        '''
        Função que implementa o método de Box-Muller para geração de números
        aleatórios com distribuição normal.

        :param mean: Média da distribuição normal.
        :param std: Desvio padrão da distribuição normal.

        :return: Número aleatório com distribuição normal.
        '''
        u1 = random.uniform(0, 1)
        u2 = random.uniform(0, 1)

        z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)

        return z1 * std + mean

    def gausian_mutation(self, individual):
        '''
        Função que implementa a mutação gaussiana.

        :param individual: Indivíduo a ser mutado.
        
        :return: Indivíduo mutado.
        '''
        z = self.box_muller(0, 1)
        mutated_individual = individual + z * self.mutation_rate
        return np.clip(mutated_individual, -2, 4)

    
    def arithmetical_crossover(self, parent1, parent2, beta):
        '''
        Função que implementa o crossover aritmético.

        :param parent1: Primeiro pai.
        :param parent2: Segundo pai.
        :param beta: Peso do cruzamento.

        :return: Dois filhos gerados pelo cruzamento aritmético.
        '''
        child1 = beta * parent1 + (1 - beta) * parent2
        child2 = beta * parent2 + (1 - beta) * parent1
        return np.clip(child1, -2, 4), np.clip(child2, -2, 4)

    
    def apply_crossover(self, population, fitness):
        '''
        Função que aplica o crossover aritmético.
        
        :param population: População atual.
        :param fitness: Fitness da população.

        :return: Nova população gerada pelo crossover aritmético.
        '''
        new_population = []
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(population, fitness)
            parent2 = self.tournament_selection(population, fitness)
            
            if random.random() < 0.7:  # Probabilidade de crossover (por exemplo, 70%)
                beta = random.random()  # Peso aleatório para o cruzamento
                child1, child2 = self.arithmetical_crossover(parent1, parent2, beta)
            else:  # Sem cruzamento, os pais seguem para a próxima geração
                child1, child2 = parent1, parent2
            
            new_population.extend([child1, child2])
        return np.array(new_population[:self.population_size])

    def tournament_selection(self, population, fitness):
        '''
        Função que implementa a seleção por torneio.

        :param population: População atual.
        :param fitness: Fitness da população.

        :return: Indivíduo selecionado pelo torneio.
        '''
        tournament_size = 3

        tournament = np.random.choice(len(population), tournament_size, replace=False)
        fitness_tournament = fitness[tournament]
        winner = tournament[np.argmax(fitness_tournament)]
        return population[winner]
    
    def calculate_fitness(self, population):
        '''
        Função que calcula o fitness da população.

        :param population: População atual.

        :return: Fitness da população.
        '''
        fitness = np.zeros(len(population))

        for i, individual in enumerate(population):
            fitness[i] = self.base_function(*individual)
        
        if np.min(fitness) < 0:
            for i, individual in enumerate(population):
                fitness[i] = fitness[i] + np.abs(np.min(fitness))
        return fitness
    
    def generate_initial_population(self):
        '''
        Função que gera a população inicial.

        :return: None
        '''
        self.population = [[4, -2]]
        for _ in range(self.population_size - 1):
            x = random.uniform(-2, 4)
            y = random.uniform(-2, 4)
            self.population.append([x, y])
        self.population = np.array(self.population)


    def apply_elitism(self, population, fitness):
        '''
        Função que aplica o elitismo.

        :param population: População atual.
        :param fitness: Fitness da população.

        :return: Indivíduos selecionados pelo elitismo.
        '''
        elite_indices = np.argsort(fitness)[-int(self.elitism_rate):]
        return population[elite_indices]

    def apply_mutation(self, population):
        '''
        Função que aplica a mutação.

        :param population: População atual.

        :return: População mutada.
        '''
        for i in range(len(population)):
            if random.random() < self.mutation_rate:  # Probabilidade de mutação
                population[i] = self.gausian_mutation(population[i])
        return population

    def show_best_solution(self):
        # Mostrar melhor solução
        best_solution = self.population[np.argmax(self.fitness)]
        print(f'Best solution: {best_solution}')
        print(f'Fitness: {self.base_function(*best_solution)}')

    def plot_3d_graphic(self):
        # Plotar gráfico 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(self.population[:, 0], self.population[:, 1], self.fitness, cmap=cm.coolwarm)
        plt.show()

    def plot_2d_graphic(self):
        # Gerar uma malha de pontos dentro do intervalo permitido
        x = np.linspace(-2, 4, 100)  # Intervalo de x com 100 divisões
        y = np.linspace(-2, 4, 100)  # Intervalo de y com 100 divisões
        X, Y = np.meshgrid(x, y)  # Criar malha de pontos

        # Calcular o fitness para cada ponto da malha
        calc_fitness_vect = np.vectorize(self.base_function)  # Vetoriza a função fitness
        Z = calc_fitness_vect(X, Y)  # Calcula o fitness para cada ponto (X, Y)

        # Plotar o mapa de calor
        plt.figure(figsize=(8, 6))
        plt.imshow(Z, extent=[-2, 4, -2, 4], origin='lower', cmap='hot', aspect='auto')
        plt.colorbar(label='Fitness')  # Barra de cores representando o fitness

        # Plotar a população atual sobre o mapa de calor
        plt.scatter(self.population[:, 0], self.population[:, 1], c='blue', marker='o', label='População Atual')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f"Mapa de Calor do Fitness - Geração {len(self.best_fitness)}")
        plt.legend()
        plt.show()


    def plot_fitness_evolution(self):
        # Plotar evolução do fitness
        plt.plot(self.best_fitness)
        plt.show()

    def run(self):
        self.generate_initial_population()
        self.best_fitness = []
        
        for generation in range(self.max_generations):
            # Calcula o fitness
            self.fitness = self.calculate_fitness(self.population)
            
            # Guarda o melhor fitness
            self.best_fitness.append(np.max(self.fitness))
            
            # Elitismo
            elite = self.apply_elitism(self.population, self.fitness)
            
            # Seleção e crossover
            new_population = self.apply_crossover(self.population, self.fitness)
            
            # Mutação
            new_population = self.apply_mutation(new_population)
            
            # Substituir população antiga pela nova
            self.population = np.vstack((new_population, elite))[:self.population_size]
            self.population = np.unique(self.population, axis=0)

            # Recalcular fitness após remover duplicados
            self.fitness = self.calculate_fitness(self.population)
            
            print(f"Geração {generation + 1} - Melhor Fitness: {self.best_fitness[-1]}")

        print("Fim do algoritmo genético")
        self.show_best_solution()
        self.plot_3d_graphic()
        self.plot_2d_graphic()
        self.plot_fitness_evolution()

