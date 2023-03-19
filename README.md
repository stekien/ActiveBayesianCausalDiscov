# ActiveBayesianCausalDiscov

The Python Codes, Figures and data objects contained in the folders are the ones used to produce
the numerical results in my masters thesis: Active Bayesian Causal Discovery with Gaussian Process Newtworks.

My masters thesis and a presentation about it are also included in the repo.

For every result or figure, all data and the code, which was used to produce the results, is stored in the folders. 

Quick start: Open the "bivariate_main.py" file, minimize the class definition, adjust parameters and run the program.

Example on how to navigate to a .py file and reproduce some results:
  1. Go to the folder "..\Daten und Code\Bivariate_Example3.2" and choose the "2tanh" folder.
  2. The file "main_withObjective.py" is self contained and includes a big class definition 
     followed by parameter specifications and plots.
  3. Set the parameters of interest and run the program. 
  
 Attention: The codes which output the optimization objective are very computation heavy.
            The programs that loop through a lot of random seeds also take very long.
