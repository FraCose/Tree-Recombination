---------------------------------------------------
This Repository contains the Algorithms explained in<br />
Cosentino, Abate, Oberhauser <br />
"Markov Chain Approximations to Stochastic Differential Equations by Recombination on Lattice Trees"<br />
---------------------------------------------------

The files are divided in the following way:<br />
- create_grid_*.py are the libraries with the code of the Algorithm presented in the cited work for the respective model.<br />
- recombination.py contains the algorithms relative to the reduction of the measure presented in <br />
COSENTINO, OBERHAUSER, ABATE - "A randomized algorithm to reduce the support of discrete measures",<br />
NeurIPS 2020, Available at https://github.com/FraCose/Recombination_Random_Algos
- main_*.py are the files to be run for the respective model. It will save some files in the folder "Results_*".
- plots_*.ipynb are the python notebooks to be run to obtain the plots of the paper. <br />
 Before running the ipython notebooks you HAVE TO run the main_*.py file. <br />

----------------------------------------------------
Special notes to run the experiments
----------------------------------------------------
- main_*.py are set to use a cuda gpu for the Monte Carlo computation. If a cuda gpu is not available you can use the function<br />
simulate_*() (uncomment/comment the relative lines in main_*.py)<br />
- changing the parameter FIG of the function build_tree() in create_grid_*.py it is possible to have a live dynamic plot of <br />
the lattice considered.<br />
- play with the list n = [] in main_*.py to reduce the running time.<br />

---------------------------------------------------
Funding
---------------------------------------------------
The authors want to thank The Alan Turing Institute and the University of Oxford<br /> 
for the financial support given. FC is supported by The Alan Turing Institute, [TU/C/000021],<br />
under the EPSRC Grant No. EP/N510129/1. HO is supported by the EPSRC grant Datasig<br />
[EP/S026347/1], The Alan Turing Institute, and the Oxford-Man Institute.
