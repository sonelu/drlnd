Welcome to the first project for the [Deep Reinforcement Learning Nanodegree](https://eu.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) at [Udacity](https://eu.udacity.com).

# Background

The first project requires the build of a Reinforcement Learning Agent, using [PyTorch](https://pytorch.org) that can navigate a simulated world and collect yellow bananas while avoiding blue bananas.

The Agent receives a reward of +1 for collecting yellow bananas and a -1 for a blue banana. The environmnent provides a **state space** of size 37 (that includes Agent's velocity as well as information about the bananas in the field of view) and the Agent can interact with the environment by chosing one of 4 actions avaialable in the **action space**: `0` move forward, `1` move backward, `2` turn left and `3` turn right. The problem is considered solved when the Agent scores an average of +13 or more over 100 episodes. There is a "benchmark" solution showcased in the project instructions that solves the problem in 1700 episodes. It is expected that our solution will provide better performance.

The environment for simulation is provided as an Unity application and for convenience there is a Python wrapper that permits interacting with this environment.

# Organisation

All the code for this project is included in one Jupiter Notebook: [Project 1 - Navigation.ipynb](Project%201%20-%20Navigation.ipynb). 

For convenience we have a PDF export of the notebook's content that includes all the results of running the cells, plots, etc. under [Report.pdf](Report.pdf). I recommend using this document for an offline reading. 

The execution of the notebook might take a significant amount of time and I have placed notes before cells that perform training of the networks. If you prefer you might skip the execution of those cells and use the data that I have generated. Every such cell saves the result of training in two folders:

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

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Before using the jupyter notebook register a new kernel in IPython for the `dlrnd` enviuronment:
```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
Now you can start Jupyter and navigate to the [Project 1 - Navigation.ipynb](Project%201%20-%20Navigation.ipynb) notebook. Follow the instructions about identifying the correct Unity application, then run the cells as you like.
