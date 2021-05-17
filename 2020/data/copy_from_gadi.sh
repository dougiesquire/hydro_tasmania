#!/bin/bash
#SBATCH -p io
#SBATCH --time=04:00:00
#SBATCH --mem=8gb

# Tranfer small files first
rsync -avPS --max-size=1G  ds0092@gadi-dm.nci.org.au:/g/data/v14/ds0092/active_projects/hydro_tasmania/2020/data/*.zarr.zip

# Then the rest
rsync -avPS ds0092@gadi-dm.nci.org.au:/g/data/v14/ds0092/active_projects/hydro_tasmania/2020/data/*.zarr.zip .
