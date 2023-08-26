from multitaskoptimizations.multiobjectiveopt import run_bayesian_optimization
from multitaskoptimizations.nsgaopt import run_nsga_optimization

def main():
    bPoints, bResult = run_bayesian_optimization()
    gPoints, gResult = run_nsga_optimization()

    print("Optimization results:")
    print("-------------------------------Params A to W -------------------------------")
    for i in range(len(bPoints)):
        param = chr(ord("A") + i)
        print(param, ": ", "Bayesian", round(bPoints[i].item(), 3), "|Genetic", round(gPoints[i], 3))

    print("-------------------------------Output-------------------------------")
    print("DT: ", "Bayesian", round(bResult[0], 3), "|Genetic", gResult[0])
    print("r: ", "Bayesian", round(bResult[1], 3), "|Genetic", gResult[1])
    print("R0: ", "Bayesian", round(bResult[2], 3), "|Genetic", gResult[2])
    print("TL: ", "Bayesian", round(bResult[3], 3), "|Genetic", gResult[3])

    print("-------------------------------Comparison-------------------------------")
    from scipy.stats import ranksums
    print("Wilcoxon rank-sum test:")
    print("Parameters", ranksums(bPoints, gPoints))
    print("Outputs,", ranksums(bResult, gResult))


if __name__ == "__main__":
    main()