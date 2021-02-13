# Genetic TSP

Genetic algorithm to solve the travelling salesman problem (TSP):

> _"Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city and returns to the origin city?"_ — From [Wikipedia](https://en.wikipedia.org/wiki/Travelling_salesman_problem)




## Images

<p align="center">
    <img width="400" height="300" src="images/cities.png">
    <img width="400" height="300" src="images/convergence.png">
    <img width="400" height="300" src="images/best_path.png">
</p>



## Installation

To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

If using Conda, you can also create an environment with the requirements:

```bash
conda env create -f environment.yml
```

By default the environment name is `genetic-TSP`. To activate it run:

```bash
conda activate genetic-TSP
````


## Usage

Run the menu with the following command:

```python
python -m genetic_TSP 
```
