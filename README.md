# Local-in-time optimal mixing project

## Setup

To download repository *associated with this branch*, 
`git clone https://github.com/cjm715/lit.git --branch sinflow --single-branch`

System Requirements:

- Python 3.6
- FFTW
- ffmpeg

Python packages: 
All required packages can be install by using the command `pip install -r requirements.txt` within this repository folder.
- jupyter
- numpy
- pyfftw 
- matplotlib

## Usage

To load `sinflow.ipynb`, you must start a jupyter notebook session by running the command `jupyter notebook sinflow.ipynb` from the root directory of the repository.


## Testing
You will need pytest module installed. Using pip package manager, use the command `pip install pytest`.

To run pytest, use the command `python -m pytest -v -s`