import fluxtools
from simulations import lit_enstrophy_sim

if __name__ == "__main__":
    python_file = 'lit_enstrophy.py'
    flux_folder_name = 'lit_enstrophy2/'
    dict_of_parameter_lists = {'Pe': [8.0, 16.0, 32.0, 64.0, 128.0, 256.0,
                                      512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0],
                               'N': [512]}

    fluxtools.run_python_file(
        python_file, dict_of_parameter_lists, flux_folder_name)
