# Local-in-time optimal mixing project

## Setup

To download repository *associated with this branch*, 
`git clone https://github.com/cjm715/lit.git --branch sinflow --single-branch`

System Requirements:

- [Python 3.6](https://wiki.python.org/moin/BeginnersGuide/Download)
- [FFTW](http://www.fftw.org/)
- [ffmpeg](https://www.ffmpeg.org/)

Python packages: 
If you use the `pip` python module manager, you can install all required packages by using the command `pip install -r requirements.txt` within the root directory of this repository. Otherwise, just make sure that the following python modules are installed in your python distribution:

- jupyter
- numpy
- pyfftw 
- matplotlib

## Usage

To load `sinflow.ipynb`, you must start a jupyter notebook session by running the command `jupyter notebook sinflow.ipynb` from the root directory of the repository. Alternatively, you can download the repository by clicking the green button at the top right of this page that says `Clone or download` and then click `Download zip`. Lastly, unzip folder locally.


## Testing
You will need pytest module installed. Using pip package manager, use the command `pip install pytest`. 

To run pytest, use the command `python -m pytest -v -s`