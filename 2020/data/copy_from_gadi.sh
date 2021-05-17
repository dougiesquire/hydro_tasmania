#!/bin/bash
#SBATCH -p io
#SBATCH --time=04:00:00
#SBATCH --mem=8gb

rsync -avPS ds0092@gadi-dm.nci.org.au:/g/data/v14/ds0092/active_projects/hydro_tasmania/2020/data/*.zarr.zip .
