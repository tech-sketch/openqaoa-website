# Fermionic QAOA

This notebook provides a brief introduction to Fermionic QAOA (FQAOA). 
It shows how this technique is implemented in the OpenQAOA workflow by solving the constrained quadratic optimization problem, an NP-hard problem.

## A brief introduction

We present an implementation of a novel algorithm designed for solving combinatorial optimization problems with constraints, utilizing the principles of quantum computing. The algorithm, known as the FQAOA [[1](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.5.023071)
, [2](https://arxiv.org/pdf/2312.04710)]
, introduces a significant enhancement over traditional methods by leveraging fermion particle number preservation. This intrinsic property allows the algorithm to enforce constraints naturally throughout the optimization process, addressing a critical challenge in many combinatorial problems.

### Key Features
- Constraint Handling: In contrast to conventional approaches, which treat constraints as soft constraints in the cost function, FQAOA enforces constraints intrinsically by preserving fermion particle numbers, thereby enhancing the overall performance of the optimization algorithm.

- Design of FQAOA Ansatz: In this algorithm,
the mixer is designed so that any classical state can be reached by its multiple actions.
The initial state is set to a ground state of the mixer Hamiltonian satisfying the constraints of the problem.

- Adiabatic Evolution: FQAOA effectively reduces to quantum adiabatic computation in the large limit of circuit depth, $p$, offering improved performance even for shallow circuits by optimizing parameters starting from fixed angles determined by Trotterized quantum adiabatic evolution.

- Performance Advantage: Extensive numerical simulations demonstrate that FQAOA offers substantial performance benefits over existing methods, particularly in portfolio optimization problems.

- Broad Applicability: The Hamiltonian design guideline benefits QAOA and extends to other algorithms like Grover adaptive search and quantum phase estimation, making it a versatile tool for solving constrained combinatorial optimization problems.

This notebook describes the implementation of FQAOA, illustrates its application through an example portfolio optimization problem, and provides insight into FQAOA's superior performance in constrained combinatorial optimization tasks.

### Quadratic Constrained Binary Optimization Problems
The constrained combinatorial optimization problem for a quadratic binary cost function $C_{\vec x}$ can be written in the following form：

$${\vec x}^{\rm opt} = \arg \min_{\vec x} C_{\vec x}\qquad {\rm s.t.} \quad\sum_{i=1}^{N} x_i = M,$$

with bit string ${\vec x}\in \{0,1\}^N$, where ${\vec x}^{\rm opt}$ is the optimal solution.
This problem can be replaced by the minimum eigenvalue problem in the following steps.

1. map the cost function $C_{\vec x}$ to the cost Hamiltonian $\hat{\cal H}_C$ by $x_i\rightarrow \hat{n}_i$:

    $$C_{\vec x} = \sum_i \mu_i x_{i}+\sum_{i,j} \sigma_{i,j} x_{i}x_{j}\quad\mapsto\quad \hat{\cal H}_C=\sum_i \mu_i \hat{n}_{i}+\sum_{i,j} \sigma_{i,j} \hat{n}_{i_1}\hat{n}_{i_2},$$
    
    where $\hat{n}_i = \hat{c}^\dagger_i\hat{c}_i$ is number operator and $\hat{c}_i^\dagger (\hat{c}_i)$ is creation (annihilation) operator of fermion at $i$-th site.

2. formulate eigenvalue problems for combinatorial optimization problem under the constraint:

    $$\hat{\cal H}_C|x_1x_2\cdots x_N\rangle = C_{\vec x}|x_1x_2\cdots x_N\rangle,$$
    
    $$\sum_{i=1}^{N} \hat{n}_i|x_1x_2\cdots x_N\rangle = M|x_1x_2\cdots x_N\rangle,$$
    
    where $|x_1x_2\cdots x_N\rangle=(\hat{c}^\dagger_1)^{x_1}(\hat{c}^\dagger_2)^{x_2}\cdots (\hat{c}^\dagger_N)^{x_N}|{\rm vac}\rangle$ is fermionic basis state and $|\rm vac\rangle$ is vacuum satisfying $\hat{c}_i|\rm vac\rangle=0$.

3. optimize FQAOA ansatz:

$$|\psi_p({\vec \gamma}^{\rm opt}, {\vec \beta}^{\rm opt})\rangle
= \left[\prod_{j=1}^pU(\hat{\cal H}_M,\beta_j^{\rm opt}){U}(\hat{\cal H}_C,\gamma_j^{\rm opt})\right]\hat{U}_{\rm init}|{\rm vac}\rangle$$

by

$$C_p({\vec \gamma}^{\rm opt}, {\vec \beta}^{\rm opt})=\min_{{\vec \gamma}, {\vec \beta}}C_p({\vec \gamma},{\vec \beta}),$$

$$\qquad{\rm where \quad}C_p({\vec \gamma}, {\vec \beta}) = \langle\psi_p({\vec \gamma}, {\vec \beta})|\hat{\cal H}_C|\psi_p({\vec \gamma}, {\vec \beta})\rangle.$$

The variational parameters $({\vec \gamma}^{\rm opt}, {\vec \beta}^{\rm opt})$ give the lowest cost value at QAOA level $p$.



```python
%matplotlib notebook
%matplotlib inline

# Import the libraries needed to employ the QAOA and FQAOA quantum algorithm using OpenQAOA
from openqaoa import FQAOA
from openqaoa import QAOA

# method to covnert a docplex model to a qubo problem
from openqaoa.problems import PortfolioOptimization

# Import external libraries to present an manipulate the data
import pandas as pd
import matplotlib.pyplot as plt
```

## Portfolio Optimization

In the following, the [portfolio optimization problem](https://en.wikipedia.org/wiki/Portfolio_optimization) is taken as a constrained quadratic optimization problem.
Start by creating an instance of the portfolio optimization problem, using the `random_instance` method of the `PortfolioOptimization`.


```python
# create a problem instance for portfolio optimization
num_assets = 8 # number of decision variables
budget = 4 # constraint on the sum of decision variables
problem = PortfolioOptimization.random_instance(num_assets=num_assets, budget=budget, penalty = None).qubo
```

##  Solving the problem

The simplest QAOA and FQAOA workflows.


```python
# conventional QAOA workflow
qaoa = QAOA()
qaoa.compile(problem = problem)
qaoa.optimize()
```


```python
# FQAOA workflow
fqaoa = FQAOA()
fqaoa.compile(problem = problem, n_fermions = budget)
fqaoa.optimize()
```

## Performance Evaluation of FQAOA
To evaluate the performance of FQAOA, we show expectation value of costs. 


```python
labels = ['QAOA', 'FQAOA']

# plot cost history
fig, ax = plt.subplots()
for i, result in enumerate([qaoa.result, fqaoa.result]):
    result.plot_cost(ax=ax, color=f'C{i}', label=labels[i])
ax.grid(True)
plt.show()
```


    
![png](/img/fqaoa_steps.png)
    



```python
# evaluate optimized expectation values of the cost
exp_cost_dict = {
    'Method': labels,
    r'Optimized Cost $\langle C_{\vec x} \rangle$':[qaoa.result.optimized['cost'], fqaoa.result.optimized['cost']]
}
df = pd.DataFrame(exp_cost_dict)
print('optimized expectation values of the cost')
display(df)
```

the optimized expectation values of the cost


| Method | Optimized Cost $\langle C_{\vec x} \rangle$ |
|--------|----------------|
| QAOA   | 4.512233       |
| FQAOA  | -0.482998      |



```python
# Print the best 5 solutions
qaoa_lowest5_dict = qaoa.result.lowest_cost_bitstrings(5)
fqaoa_lowest5_dict = fqaoa.result.lowest_cost_bitstrings(5)

qaoa_dict = {
    r'Bitstring, $\boldsymbol{x}$': qaoa_lowest5_dict['solutions_bitstrings'],
    r'Cost, $C_{\vec x}$': qaoa_lowest5_dict['bitstrings_energies'],
    labels[0]: qaoa_lowest5_dict['probabilities'],
    labels[1]: fqaoa_lowest5_dict['probabilities']
}
df = pd.DataFrame(qaoa_dict)
print('probabilities of finding the best five optimal solutions')
display(df)
```

the probabilities of finding the best five optimal solutions by using QAOA and FQAOA

| Index | Bitstring, $\vec x$ | Cost, $C_{\vec x}$| QAOA     | FQAOA    |
|-------|---------------------|-------------------|----------|----------|
| 0     | 11100001            | -1.371650         | 0.005642 | 0.009979 |
| 1     | 11101000            | -1.193696         | 0.006019 | 0.024488 |
| 2     | 10101001            | -1.080295         | 0.006139 | 0.192331 |
| 3     | 10110001            | -1.079240         | 0.005780 | 0.044356 |
| 4     | 11100100            | -0.949303         | 0.006036 | 0.020335 |


References
----------
1. T. Yoshioka, K. Sasada, Y. Nakano, and K. Fujii, [Phys. Rev. Research 5, 023071 (2023).](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.5.023071), [arXiv:2301.10756 [quant-ph]](https://arxiv.org/pdf/2301.10756).
2. T. Yoshioka, K. Sasada, Y. Nakano, and K. Fujii, [2023 IEEE International Conference on Quantum Computing and Engineering (QCE) 1, 300-306 (2023).](https://ieeexplore.ieee.org/document/10313662), [arXiv:2312.04710 [quant-ph]](https://arxiv.org/pdf/2312.04710).
