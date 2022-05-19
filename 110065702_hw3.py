import numpy as np

# you must use python 3.6, 3.7, 3.8, 3.9 for sourcedefender
import sourcedefender
import random
from HomeworkFramework import Function


parameterPool = [[0.3, 0.25],[0.3, 0.3],[0.35, 0.25],[0.35, 0.35],[0.35, 0.4],[0.4, 0.25],[0.4, 0.3],[0.4, 0.35],[0.4, 0.4],[0.4, 0.45],[0.4, 0.5],[0.4, 0.55],[0.4, 0.6],[0.45, 0.45],[0.45, 0.5],[0.45, 0.55],[0.45, 0.75]]
# parameterPool = [[0.4,0.1],[0.4,0.9],[0.3,0.2]]

def rand1Bin(i, popsize, pop, dimensions, F, crossp, best):
    idxs = [idx for idx in range(popsize) if idx != i]
    a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
    mutant = np.clip(a + F * (b - c), 0, 1) # np.clip aim to restrict the value to [0,1]
    cross_points = np.random.rand(dimensions) < crossp # cross_points is an array like [False  True False False True False .....]
    if not np.any(cross_points):
        cross_points[np.random.randint(0, dimensions)] = True # avoid False for all dimension
    return np.where(cross_points, mutant, pop[i]) # generate new candidate base on cross_points

def rand2Bin(i, popsize, pop, dimensions, F, crossp, best):
    idxs = [idx for idx in range(popsize) if idx != i]
    a, b, c, d, e = pop[np.random.choice(idxs, 5, replace = False)]
    mutant = np.clip(a + F * (b - c + d - e), 0, 1) # np.clip aim to restrict the value to [0,1]
    cross_points = np.random.rand(dimensions) < crossp # cross_points is an array like [False  True False False True False .....]
    if not np.any(cross_points):
        cross_points[np.random.randint(0, dimensions)] = True # avoid False for all dimension
    return np.where(cross_points, mutant, pop[i]) # generate new candidate base on cross_points

def currentToRand1(i, popsize, pop, dimensions, F, crossp, best):
    idxs = [idx for idx in range(popsize) if idx != i]
    a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
    mutant = a + F * (b - c)
    newpop =  pop[i] + random.uniform(0, 1)*(mutant-pop[i])
    newpop = np.clip(newpop, 0, 1)
    return newpop

def best1Bin(i, popsize, pop, dimensions, F, crossp, best):
    idxs = [idx for idx in range(popsize) if idx != i]
    a, b = pop[np.random.choice(idxs, 2, replace = False)]
    mutant = np.clip(best + F * (a - b), 0, 1) # np.clip aim to restrict the value to [0,1]
    cross_points = np.random.rand(dimensions) < crossp # cross_points is an array like [False  True False False True False .....]
    if not np.any(cross_points):
        cross_points[np.random.randint(0, dimensions)] = True # avoid False for all dimension
    return np.where(cross_points, mutant, pop[i]) # generate new candidate base on cross_points

def best2Bin(i, popsize, pop, dimensions, F, crossp, best):
    idxs = [idx for idx in range(popsize) if idx != i]
    a, b, c, d = pop[np.random.choice(idxs, 4, replace = False)]
    mutant = np.clip(best + F * (a - b + c - d), 0, 1) # np.clip aim to restrict the value to [0,1]
    cross_points = np.random.rand(dimensions) < crossp # cross_points is an array like [False  True False False True False .....]
    if not np.any(cross_points):
        cross_points[np.random.randint(0, dimensions)] = True # avoid False for all dimension
    return np.where(cross_points, mutant, pop[i]) # generate new candidate base on cross_points

def randToBest1Bin(i, popsize, pop, dimensions, F, crossp, best):
    idxs = [idx for idx in range(popsize) if idx != i]
    a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
    mutant = np.clip(a + F * (b - c + best - a), 0, 1) # np.clip aim to restrict the value to [0,1]
    cross_points = np.random.rand(dimensions) < crossp # cross_points is an array like [False  True False False True False .....]
    if not np.any(cross_points):
        cross_points[np.random.randint(0, dimensions)] = True # avoid False for all dimension
    return np.where(cross_points, mutant, pop[i]) # generate new candidate base on cross_points

functionSet = [rand1Bin,rand2Bin,best1Bin,best2Bin,randToBest1Bin,currentToRand1]

# source of this function: https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/ 

def de(function, bounds,  popsize, parameterPool):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions) #Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([function(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop[best_idx]
    best_denorm = pop_denorm[best_idx]
    while True:
        for i in range(popsize):
            parameter = parameterPool[random.randrange(len(parameterPool))]
            trial = []
            trial_denorm = []
            f = []
            trial.append(rand1Bin(i, popsize, pop, dimensions, parameter[0], parameter[1], best))
            trial.append(rand2Bin(i, popsize, pop, dimensions, parameter[0], parameter[1], best))
            trial.append(currentToRand1(i, popsize, pop, dimensions, parameter[0], parameter[1], best))
            trial_denorm.append(min_b + trial[0] * diff)
            trial_denorm.append(min_b + trial[1] * diff)
            trial_denorm.append(min_b + trial[2] * diff)
            f.append(function(trial_denorm[0]))
            f.append(function(trial_denorm[1]))
            f.append(function(trial_denorm[2]))
            minIdx = np.argmin(np.array(f))
            if f[minIdx] < fitness[i]:
                fitness[i] = f[minIdx]
                pop[i] = trial[minIdx]
                if f[minIdx] < fitness[best_idx]:
                    best_idx = i
                    best_denorm = trial_denorm[minIdx]  
                    best = trial[minIdx] 
        yield best_denorm, fitness[best_idx]


class DE_optimizer(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func) # must have this init to work normally

        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)

        self.target_func = target_func

        self.generation = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)
        self.popsize = 20
        self.parameterPool = parameterPool

    def evalFunc(self, solution):
        returnValue = self.f.evaluate(self.target_func, solution)
        if returnValue == "ReachFunctionLimit":
            print("ReachFunctionLimit")
            exit() 
        else:
            return returnValue

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def run(self, FES): # main part for your implementation
        boundList = []
        for _ in range(self.dim):
            boundList.append((self.lower, self.upper))
        de_Generator = de(self.evalFunc, boundList, self.popsize, self.parameterPool)
        self.generation+=1
        while self.generation*self.popsize*3 < FES:
            # print('=====================FE=====================')
            # print(self.generation*self.popsize)
            self.optimal_solution, self.optimal_value = next(de_Generator)
            self.generation+=1
            # print("optimal: {}\n".format(self.get_optimal()[1]))

            

if __name__ == '__main__':
    func_num = 1
    fes = 0

    # op = DE_optimizer(1)
    # print(op.evalFunc([0.5,-0.5,0.5,-0.5,0.5,-0.5]))


    # function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000 
        else:
            fes = 2500

        # you should implement your optimizer
        op = DE_optimizer(func_num)
        op.run(fes)
        
        best_input, best_value = op.get_optimal()
        # print(best_input, best_value)
        print(best_value)
        
        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1 
