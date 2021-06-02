---------------------------------------------------
This Repository contains the Algorithms explained in<br />
Cosentino, Abate, Oberhauser <br />
"Multinomial Multivariate Universal Lattice for SDE approximations"<br />
---------------------------------------------------

The files are divided in the following way:<br />
- create_grid.py is the library with the code of the Algorithm presented in the cited work.<br />
- recombination.py contains the algorithms relative to the reduction of the measure presented in <br />
COSENTINO, OBERHAUSER, ABATE - "A randomized algorithm to reduce the support of discrete measures",<br />
NeurIPS 2020, Available at https://github.com/FraCose/Recombination_Random_Algos
- main.py is the file to be run. It will save some files in the folder "Results"
- plots.ipynb is the python notebook to be run to obtain the plots of the paper. <br />
 Before running the ipython notebooks you HAVE TO run the main.py file. <br />

----------------------------------------------------
Special Note to run the experiments
----------------------------------------------------
- main.py is set to use a cuda gpu for the Monte Carlo computation. If a cuda gpu is not available you can use the function<br />
simulate_HM() (uncomment/comment the relative lines in main.py)<br />
- changing the parameter FIG of the function build_tree() in create_grid.py it is possible to have a live dynamic plot of <br />
the lattice considered.<br />
- play with the list n = [] in main.py to reduce the running time.<br />

---------------------------------------------------
Funding
---------------------------------------------------
The authors want to thank The Alan Turing Institute and the University of Oxford<br /> 
for the financial support given. FC is supported by The Alan Turing Institute, [TU/C/000021],<br />
under the EPSRC Grant No. EP/N510129/1. HO is supported by the EPSRC grant Datasig<br />
[EP/S026347/1], The Alan Turing Institute, and the Oxford-Man Institute.
