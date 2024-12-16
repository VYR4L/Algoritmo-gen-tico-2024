from gen_alg import GeneticAlgorithm


base_function = lambda x, y : (1 - x)**2 + 100 * (y - x**2)**2

try:
    population_size = int(input('Digite o tamanho da população: '))
    max_generations = int(input('Digite o número máximo de gerações: '))
    mutation_rate = float(input('Digite a taxa de mutação: '))
except ValueError:
    print('Valores inválidos. Utilizando valores padrão.')
    population_size = 100
    mutation_rate = 0.02

gen_alg = GeneticAlgorithm(base_function=base_function, max_generations=max_generations, population_size=population_size, mutation_rate=mutation_rate)

gen_alg.run()