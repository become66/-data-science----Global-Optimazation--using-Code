# project 內容:
這是清大的碩班課程資料科學的第三次作業，對應到Global Optimazation的單元，內容是給我們未知的4個function，每個function有限制的evaluation上限，我們要運用有限的evaluation次數，設計演算法來盡力找到function的global minimum，詳細內容可參見Data Science HW3.pptx，並且我是選擇實現CoDE演算法來完成這次作業


# 前言:
- 程式中Differential Evolution function的part是修改自這個帥氣的法國男人的網站，
    
    [A tutorial on Differential Evolution with Python](https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/)

    ```python
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
                functionChoiced = functionSet[random.randrange(len(functionSet))]
                trial = functionChoiced(i, popsize, pop, dimensions, parameter[0], parameter[1], best)
                trial_denorm = min_b + trial * diff
                f = function(trial_denorm)
                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = trial
                    if f < fitness[best_idx]:
                        best_idx = i
                        best_denorm = trial_denorm  
                        best = trial 
            yield best_denorm, fitness[best_idx]
    ```

# 各個branch的內容:
### main: 
助教給的initial code
### Differential_Evolution
modify帥氣法國男人網站的程式，變成此project可以使用的形式，運用Differential_Evolution來進行Global Optimazation
### CoDE: 
原始論文的方法，每個generation的每個candidate都會使用random從parameter pool中選擇的parameter，並使用每個(程式中用三個)mutaion-crossover function，從中挑選最佳的與原本的candidate比較，若較好則取代原本的candidate
### CoDE-single_parameter_pool-random_choice_mutation_function_for_each_candidate:
由於測試效果最好，所以為最終上傳的版本，只有一個parameter pool，並且對於每個generation的每一個candidate，都random選一個mutaion-crossover function和parameter pool中的一組parameter來進行mutation-crossover，與原本的candidate比較，若較好則取代原本的candidate
### CoDE-multiple_parameter-random_choice_mutation_function_for_each_candidate:
對於每一個mutaion-crossover function有其專屬的parameter pool(針對function測試挑選過的parameter們)，對於每個generation的每一個candidate，都random選一個mutaion-crossover function來進行mutation-crossover，與原本的candidate比較，若較好則取代原本的candidate
### tuning_parameters
對不同的mutaion-crossover function進行實驗，找出他們較好的parameter
### test_different_function_set
測試比較不同比例的mutaion-crossover function(某些function有較高機率被選中)進行Global Optimazation的效果
