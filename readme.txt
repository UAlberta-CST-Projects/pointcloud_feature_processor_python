Requirements:
Either use the included env file to recreate the conda env, or just install the packages listed in the pip section of the yaml file.
Also python must be at least version 3.8

Note: You need to run the program from the conda env mentioned above, this can be done either from the command line via conda or through your ide if supported

Requirement install instructions:
1. If you do not already have anaconda installed on your computer get it here https://www.anaconda.com/products/individual
Linux: After installing from the link above or your package manager, your terminals should start with "(base)"
Windows: After installing from the link above, you should have a new terminal application available from the start menu called "anaconda"
2. In your respective anaconda enabled terminal navigate to this project's folder
3. Create the conda environment with the following command
> conda env create --file condaenv.yaml
4. Whenever you want to run the project you must ensure the environment is activated with 
> conda activate pointcloudenv
and run the project with either of the usage instructions below.

Usages:
1. (Recommended) You can simply run the compute_features.py file and it will guide you through the process.
> python compute_features.py
You can also run compute_features.py through pycharm if you properly set the python environment for this project in pycharm.
Note(legacy interface): When asked for which features you would like, you can specify multiple options by inputting the associated numbers space separated. (e.g. 0 1 3)

2. You can use the basic cli to run it from the terminal. Although not all features are available via cli at this point in time.
> python compute_features.py --file=S.las --gradient --gfield=z
> python compute_features.py --file=S.las --roughness --rradius=0.2 --zdiff --diffradius=0.2

The available options:
--gradient --gfield=<x,y,z>
--roughness --rradius=<x.x>
--density --dradius=<x.x> --unprecise
--zdiff --diffradius=<x.x>
--file=<filepath>

The output file will be the original name of your file with "processed" added onto the end.

To compute gradient you must specify gfield.
To compute roughness you must specify rradius.
To compute density you must specify dradius.
Density also offers the --unprecise option. See the relevant code section in the operations.py file for details.
To compute zdiff you must specify diffradius.
You may compute any combination of the three simultaneously if you specify the correct options.

--file must refer to either an absolute path or a relative path from the running directory.

*For implementation and developer details please see the related section in the matlab tutorial document as well as the comments in the code itself.
