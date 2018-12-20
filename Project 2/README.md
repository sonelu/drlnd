Welcome to the second project for the [Deep Reinforcement Learning Nanodegree](https://eu.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) at [Udacity](https://eu.udacity.com).

# Background

This project requires the build of a Reinforcement Learning Agent, using [PyTorch](https://pytorch.org) that can ncontrol a 2 DOF arm to follow a moving target.

![reacher](reacher.gif)

The Agent receives a reward of +0.4 (the original text at Udacity wrongly states the reward per step is +0.1) for maintaining the end effector within the target bubble. The **observation space** consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each **action** is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. The environment considers an episode of being formed of 1001 steps. The challenge is to train the agent so that the average return over 100 episodes is above 30. When using the environment with 20 agents (see below) the return per episode is the average of the returns over the 20 agents.

The environment for simulation is provided as an Unity application and for convenience there is a Python wrapper that permits interacting with this environment.

# Organisation

All the code for this project is included in one Jupiter Notebook: [Project 2 - Continuous Control.ipynb](Project%202%20-%20Continuous%20Control.ipynb). 

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

**One Agent Environment**
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

**Twenty Agents Environment**
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)


Before using the jupyter notebook register a new kernel in IPython for the `dlrnd` enviuronment:
```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
Now you can start Jupyter and navigate to the [Project 2 - Continuous Control.ipynb](Project%202%20-%20Continuous%20Control.ipynb) notebook. Follow the instructions about identifying the correct Unity application, then run the cells as you like.
