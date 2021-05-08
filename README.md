# Atarai
This software package provides utilities and functionality for performing Deep Reinforcement Learning on Atari games.
## Structure
Each of the following files provide the following functionality:
* Agent- handles on the actions to take given a specified state, this interacts with the set strategy and neural network to return the appropraite action
* Strategy- communicates to the agent whether to perform a random or optimal action depending on the current training episode and given parameters
* Environment handler- handles the transmission of states and actions between the GYM environment and the agent. Depending on the game, certain
pre-processing techniques (condensing, state manipulation, colour correction) are performed on the state returned by the GYM environment before
being returned to the agent
* Policy- contains all the neural networks, QValues and policy gradient objects needed to optimise the agent 
* Replay memory- stores all experiences currently undertaken by the agent and returns these in batches to optimise the agent's performance
* Results handler- accepts all results given to the agent and plots these for the user in appropriate graphs as well as saving the final results to an Excel file
* Optimal game parameters- stores all the parameters for each set game, which are used when training the agent for a given game, and verifies each parameter
to ensure it is in the correct format
* Main- connects all elements of the model and performs a training loop that optimises the model for a given game
## Setup 
How to run:
1. Install the following required modules:
	* PyTorch
	* Torchvision
	* Numpy
	* Gym
	* Atari_py
	* Pandas
	* Json
	* Matplotlib
	* Openpyxl
	* Xlsxwriter
2. To create a diagram of the neural network, GraphVis executables must be installed and set as an OS environment variable path.
	   If this is not desirable, set the variable 'show_neural_network' to False.
3. Of time of writing (April 2020) atari_py has an error where the file 'ale.py' is not included in the module package, this can be solved by downloading the module
	   from the atari_py github page, rather than downloading through PIP.
4. Now that everything is installed, the program can be executed by running main.py. The game to run can be changed by setting the variable 
	   'default_atari_game' to one of the avaliable atari games (each game must have an 'optimal_game_parameter' namedtuple assigned to it).
5. A series of parameters for the different games have been set in the 'optimal_game_parameters' file, thescan be edited and rune can be edited and run.
## Performance
#### Break out
![Break Out Performance](https://media.giphy.com/media/csj5usnHg2u6rUwhzv/giphy.gif)
#### Enduro
![Enduro Performance](https://media3.giphy.com/media/xRCi9lbXAsm8gLMDW8/giphy.gif)
#### Pacman
![Pacman Performance](https://media1.giphy.com/media/YgEn2czh567NtVWlCU/giphy.gif)
