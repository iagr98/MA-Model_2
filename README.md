## Separator Model

## Getting started

0D Separator Model. Documentation are published here: https://arxiv.org/abs/2406.01528

Model is written in a script style.

Further infos --> contact: song.zhai@avt.rwth-aachen.de

## Getting Started
These instructions will guide you in setting up the project on your local machine for development and testing purposes. If working on an AVT terminalserver, please ensure to read the wiki guidlines: https://wiki.avt.rwth-aachen.de/de/02_Einrichtungen/IT/Alle/Python-Nutzung


### Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.9 or higher

### Installation

To set up the Anlagensteuerung with a local environment, follow these steps:

1. Clone the repository to your local machine:
   ```bash
   git clone git@git.rwth-aachen.de:avt-fvt/private/separator_model_0d.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Separator_model
   ```
3. Create a virtual environment to isolate the project dependencies:
   ```bash
   python -m venv env
   ```
4. Activate the virtual environment:
   ```bash
   env\Scripts\activate
   ```
5. Install the required dependencies using `requirements.txt` (these might differ depending on the plugins you use):
   ```bash
   conda env create -f simulation_env.yml
   ```

### Setup & Usage

### V1.0
   ## Parameter setup
      1. constants.py
      2. Properties.properties_xxx.py
   ## Calculation of one simulation
      1. main_script_fun.py
         run_mainscript(input)
         Compare --> script for multiple simulations...
   ## Calculation of multiple operating conditons
      Grid: wrapper_grid_calc.py
      LHS: wrapper_LHS.calc.py
   ## Postprocessing - hardcoded
      1. postprocessing.py

### V2.0
   ## Parameter setup
      1. Go to "settings" folder
      2. Modify modules xxx.py
   ## Calculation of one simulation
      1. search test_script in scripts folder and copy paste it to V2.0 folder
      2. SeparatorModel --> takes all parameter in initialization
      3. SeparatorModel.solve_ODE() --> solve set ODE system
   ## Calculation of multiple operating conditons
      Grid: wrapper_grid_calc.py
      LHS: wrapper_LHS.calc.py
   ## Postprocessing - hardcoded
      1. postprocessing.py


## Authors
- @song.zhai - Initial version

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
Cite and further reads in this [paper](https://arxiv.org/abs/2406.01528).
