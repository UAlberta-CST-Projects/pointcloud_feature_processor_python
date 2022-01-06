Requirements:
Either use the included env file to recreate the conda env, or just install the packages listed in the pip section of the yaml file.
Also python must be at least version 3.8

Note: You need to run the program from the conda env mentioned above, this can be done either from the command line via conda or through you ide if supported

Usages:
1. You can simply run the compute_features.py file and it will guide you through the process.

2. You can use the basic cli to run it from the terminal.
Example> python cc_standalone.py --file=S.las --gradient --gfield=z

The available options:
--gradient --gfield=<x,y,z>
--roughness --rradius=<x.x>
--density --dradius=<x.x> --unprecise
--zdiff --diffradius=<x.x>
--file=<filepath>

To compute gradient you must specify gfield.
To compute roughness you must specify rradius.
To compute density you must specify dradius.
Density also offers the --unprecise option. See the relevant code section in the operations.py file for details.
To compute zdiff you must specify diffradius.
You may compute any combination of the three simultaneously if you specify the correct options.

--file must refer to either an absolute path or a relative path from the running directory.