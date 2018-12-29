Welcome to the third project for the [Deep Reinforcement Learning Nanodegree](https://eu.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) at [Udacity](https://eu.udacity.com).

# Background

This project requires the build of a Reinforcement Learning Agent, using [PyTorch](https://pytorch.org) that can play tennis by controlling two agents in an simlator.

![tennis](tennis.png)

The environment has two agents, representing the two players. Each agent receives a set of observations (where the **observation space** is a 24 dimensional array that includes informations about the position and velocity of the agent and the ball), and interacts with the environment with a 2 dimensional action vector (**action space** is 2 dimensional) representing the movement of the racket.

When one agent hits the ball and this lands in the opponent's space there is a reward of +0.1. If one agent misses the ball and this hits the ground in his court, the agent gets a ngative reward of -0.01 and the episode ends. When the episode is finished the score for the whole episode is the maximum between the two scores realized by the two agents.

The challenge is to train a model where each of th actors only use their own information for action determination so that the average score over 100 consecutive episodes is higher than 0.5.

The environment for simulation is provided as an Unity application and for convenience there is a Python wrapper that permits interacting with this environment.

# Organisation

The main code for this project is included in one Jupiter Notebook: [Project 3 - Collaboration and Competition.ipynb](Project%203%20-%20Collaboration%20and%20Competition.ipynb).

A number of supporting source files are included and these are listed in the jupyter notebook in the appendix.

For convenience we have a PDF export of the notebook's content that includes all the results of running the cells, plots, etc. under [Report.pdf](Report.pdf). I recommend using this document for an offline reading. 

The execution of the notebook is straightforward, all cells that perform training have been commented and the results are included in the notebook text. The analysis of the data is based on loading the pre-saved run results. Every such cell saves the result of training in two folders:

* [models/](models/) contains the trainined PyTorch models for the NN used by Agents
* [results/](results/) contains the statistics from the training process (average score per episode, running averages, execution time, etc.) that I use later when creating plots of the training performance

# Installation

If you wish to run the notebook on you own system you will first need to comnfigure it.

It is highly recommended that you use Anaconda and setup a dedicated environment for this purpose:

```
conda create --name drlnd python=3.6
source activate drlnd
```
Perform a minimal install of OpenAI gym:
```
pip install gym
```
Next install `classic_control` and `box2d`:
```
pip install gym[classic_control]
pip install gym[box2d]
```
Clone the main repository from the Deep Reinforcement Learning course and install dependencies:
```
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
cd ../..
```
Now clone this repository:
```
git clone https://github.com/sonelu/drlnd.git
```
You will now need to save into the drlnd directory the Unity Environment. Depending on you platform you will have to use:

**Twenty Agents Environment**
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)


Before using the jupyter notebook register a new kernel in IPython for the `dlrnd` enviuronment:
```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
Now you can start Jupyter and navigate to the [Project 3 - Collaboration and Competition.ipynb](Project%203%20-%20Collaboration%20and%20Competition.ipynb) notebook. Follow the instructions about identifying the correct Unity application, then run the cells as you like.
