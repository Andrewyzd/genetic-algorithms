# Import packages
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt


# Class City
# A list of 7 cities
class City:
    cityList = ["Georgetown", "Alor Setar", "Kangar", "Shah Alam", "Kota Bahru", "Ipoh", "Kuala Terengganu"]

# Class distance
# A list of distance between two different cities respectively
class Distance:
    distanceBetweenCity = {
        tuple(["Georgetown", "Alor Setar"]): 122,
        tuple(["Georgetown", "Kangar"]): 171,
        tuple(["Georgetown", "Shah Alam"]): 364,
        tuple(["Georgetown", "Kota Bahru"]): 343,
        tuple(["Georgetown", "Ipoh"]): 160,
        tuple(["Georgetown", "Kuala Terengganu"]): 477,

        tuple(["Alor Setar", "Georgetown"]): 122,
        tuple(["Alor Setar", "Kangar"]): 46,
        tuple(["Alor Setar", "Shah Alam"]): 444,
        tuple(["Alor Setar", "Kota Bahru"]): 359,
        tuple(["Alor Setar", "Ipoh"]): 240,
        tuple(["Alor Setar", "Kuala Terengganu"]): 463,

        tuple(["Kangar", "Georgetown"]): 172,
        tuple(["Kangar", "Alor Setar"]): 49,
        tuple(["Kangar", "Shah Alam"]): 494,
        tuple(["Kangar", "Kota Bahru"]): 409,
        tuple(["Kangar", "Ipoh"]): 291,
        tuple(["Kangar", "Kuala Terengganu"]): 513,

        tuple(["Shah Alam", "Georgetown"]): 363,
        tuple(["Shah Alam", "Alor Setar"]): 443,
        tuple(["Shah Alam", "Kangar"]): 492,
        tuple(["Shah Alam", "Kota Bahru"]): 461,
        tuple(["Shah Alam", "Ipoh"]): 211,
        tuple(["Shah Alam", "Kuala Terengganu"]): 473,

        tuple(["Kota Bahru", "Georgetown"]): 343,
        tuple(["Kota Bahru", "Alor Setar"]): 357,
        tuple(["Kota Bahru", "Kangar"]): 406,
        tuple(["Kota Bahru", "Shah Alam"]): 462,
        tuple(["Kota Bahru", "Ipoh"]): 341,
        tuple(["Kota Bahru", "Kuala Terengganu"]): 165,

        tuple(["Ipoh", "Georgetown"]): 160,
        tuple(["Ipoh", "Alor Setar"]): 240,
        tuple(["Ipoh", "Kangar"]): 289,
        tuple(["Ipoh", "Shah Alam"]): 205,
        tuple(["Ipoh", "Kota Bahru"]): 350,
        tuple(["Ipoh", "Kuala Terengganu"]): 367,

        tuple(["Kuala Terengganu", "Georgetown"]): 447,
        tuple(["Kuala Terengganu", "Alor Setar"]): 462,
        tuple(["Kuala Terengganu", "Kangar"]): 511,
        tuple(["Kuala Terengganu", "Shah Alam"]): 460,
        tuple(["Kuala Terengganu", "Kota Bahru"]): 165,
        tuple(["Kuala Terengganu", "Ipoh"]): 369
    }

# Class Fitness
# To calculate the fitness of the distance of a route
class Fitness:
    def __init__(self, cities_list):
        self.cities = cities_list
        self.cityDistance = Distance.distanceBetweenCity
        self.distances = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distances == 0:
            pathDistance = 0
            for i in range(0, len(self.cities)):
                fromCity = self.cities[i]
                toCity = None
                if i + 1 < len(self.cities):
                    toCity = self.cities[i+1]
                else:
                    toCity = self.cities[0]
                pathDistance += self.mapCityDistance(fromCity, toCity)
            self.distances = pathDistance
        return self.distances

    def mapCityDistance(self, fromcity, toCity):
        distance = 0
        for k, ch in self.cityDistance.items():
            if k[0] == fromcity and k[1] == toCity:
                distance = ch
                break;

        return distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


# Function to generate the route randomly
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


# Function to create a population with seven route
def initialRoutePopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        generatedRoute = createRoute(cityList)
        population.append(generatedRoute)

    return population


# Function to return the sorted fitness of the route
def rankRoutes(population, needSorted):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()

    if needSorted:
        return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)
    else:
        return fitnessResults


# Function to perform selection
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


# Function for mating pool
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])

    return matingpool


# Function for breed the offspring from two parent chromosomes
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []


    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))


    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)


    for i in range(startGene, endGene):
            childP1.append(parent1[i])


    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2

    return child


# Function to breed the population
def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)

    return children


# Function to perform the mutation between the gene of a single chromosomes
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))


            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1

    return individual


# Function to mutate the chromosomes of a population
def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


# Function to generate new generation based on selection, crossover, and mutation
def nextGeneration(populationRoute, eliteSize, mutationRate):
    popRanked = rankRoutes(populationRoute, True)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(populationRoute, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


# Function to display all the seven route in a population
def displayInitialPopulation(initialpopulationRoute):
    routesResult = rankRoutes(initialpopulationRoute, False)

    print('Initial possible route with distance: ')
    for i in range(0, len(initialpopulationRoute)):
        routes = len(initialpopulationRoute[i])
        for j in range(0, routes):
            print(initialpopulationRoute[i][j], end=" -> ")

        print(initialpopulationRoute[i][0], end=" = ")
        print(round(1 / routesResult[i], 0), "KM")
        print("")

    print("Initial distance: " + str(round(1 / rankRoutes(initialpopulationRoute, True)[0][1], 0)) + "KM")


# Function to display the best route with the shortest distance
def displayFinalRoute(bestRoute, pop):
    for x in range(0, len(bestRoute)):
        print(bestRoute[x], end=" -> ")

    print(bestRoute[0], end=" = ")

    print(round(1 / rankRoutes(pop, True)[0][1], 0), "KM")


# Function for genetic algorithms
def geneticAlgorithm(population, populationSize, eliteSize, mutationRate, generations):
    pop = initialRoutePopulation(populationSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop, True)[0][1])

    displayInitialPopulation(pop)


    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop, True)[0][1])


    bestRouteIndex = rankRoutes(pop, True)[0][0]
    bestRoute = pop[bestRouteIndex]

    print("")
    print("Final Route with distance: ")
    displayFinalRoute(bestRoute, pop)


    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

    return bestRoute


# Execute the genetic algorithms
exit = False;

while not exit:
    cityList = City.cityList
    # prompt user to input the parameters
    try:
        populationSize = int(input("Enter the population size: "))
        mutationRate = float(input("Enter the mutation rate: "))
        generation = int(input("Enter the generation: "))
        eliteSize = int(input("Enter the elite size: "))
        while eliteSize > 6:
            print("The elite size cannot be more than or equal to zero because only 7 cities in each individual (chromosomes).")
            eliteSize = int(input("Please enter your elite size again: "))
    except ValueError:
        print("Invalid input! Please enter again!")
        continue

    # passing parameter to genetic algorithms
    geneticAlgorithm(cityList, populationSize, eliteSize, mutationRate, generation)

    # ask user whether to continue or not
    decision = input("Do you want to continue? [y / n]: ")
    while decision != "y" and decision != "Y" and decision != "n" and decision != "N":
        decision = input("Please enter y or n: ")

    if decision == "y" or decision == "Y":
        exit = False

    elif decision == "n" or decision == "N":
        exit = True
