# ki Sprint 43A: Baseline predictive models for the BEAN dataset
The code and result figures for sprint 43a are available in this repository. You'll have to bring your own data if you want to rerun the models.

## Installation Instructions
The easiest way to manage a Python installation is to use Anaconda. 

First install a Python 3.x miniconda on your system: https://docs.conda.io/en/latest/miniconda.html

Once it is installed, open a miniconda shell/terminal, and execute the following command to install all the necessary packages:

```bash
conda install pip cython -y
pip install -r requirements.in
```

## Data
To be able to run the code in this repository, you'll have to make a `data/` folder and put the following files in it:
- `DataRequest_Sanchez-Alonso_12.10.20.xlsx`
- `BEAN_testing_training_n130.xlsx`

## Code Organization
The files that begin with `01_*`, `02_*`, etc are the main scripts. They should be run in order:

- `01_import_data.py` - Once you stick your data into the `data/` folder, you can run this script with `python 01_import_data.py`. 
It will convert the raw data into a machine learning ready format.

- `02_run_ml_models.py` - This will run all of the models, and save all of the results. It takes about 1 to 2 hours. Run with `python 02_run_ml_models.py`.

- `03_visualize.ipynb` - This is a Jupyter notebook. To run, you'll have to execute `jupyterlab` in your terminal. You will then see an address pop-up. 
Copy and paste that into your browser and you'll see the notebook pop-up. You can re-execute all of its cells to update the results. 
For more on JupyterLab see: https://jupyterlab.readthedocs.io/en/stable/

The rest of the files have various code utilities used by the numbered scripts. They are imported when necessary. 