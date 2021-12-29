Requirements:
Either use the included env file to recreate the conda env, or just install the packages listed in the pip section of the yaml file.
Also python must be at least version 3.8

Usages:
1. You can simply run the compute_features.py file and it will guide you through the process.

2. You can use the basic cli to run it from the terminal.
Example> python cc_standalone.py --file=S.las --gradient --gfield=z

To compute gradient you must specify gfield.
To compute roughness you must specify rradius.
To compute density you must specify dradius.
Density also offers the --unprecise option. See the relevant code section in the operations.py file for details.
You may compute any combination of the three simultaneously if you specify the correct options.

--file must refer to either an absolute path or a relative path from the running directory.