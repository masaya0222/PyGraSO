import os
import yaml
import logging
import argparse

from pysoc_deriv.parser import decode_gaussian_parser
from pysoc_deriv.preprocessing import extract_info
from pysoc_deriv.calc_soc import (
    calc_soc_s0t1,
    calc_soc_s0t1_deriv,
    calc_soc_s1t1,
    calc_soc_s1t1_deriv,
)


def merge_configs(defaults: dict, loaded_config: dict) -> dict:
    """
    Merge loaded configuration into default configuration.
    If a key exists in both, the loaded configuration overwrites the default.
    """
    config = defaults.copy()
    config.update(loaded_config)
    return config


def load_config(file_path: str) -> dict:
    """Loads a YAML configuration file."""
    try:
        if not (file_path.endswith(".yaml") or file_path.endswith(".yml")):
            raise ValueError("Only .yaml or .yml files are supported.")

        with open(file_path, "r") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError("The YAML file must be a dictionary-like structure.")

        return config
    except Exception as e:
        print(f"Failed to load the configuration file: {e}")
        return {}


def check_file_exist(config):
    """Check if the needed files exist."""
    # check xyz file
    file_key = "xyz_file"
    if not (file_key in config):
        raise KeyError(f"Error : '{file_key}' are missing from config")
    if not (os.path.exists(config[file_key])):
        raise FileNotFoundError(f"Error: '{config[file_key]}' do not exist.")

    if config.get("dump"):
        check_list = ["json_file", "npz_file"]
    else:
        check_list = ["log_file", "rwf_file"]

    # check about triplet
    for file_name in check_list:
        file_key = f"triplet_{file_name}"
        if not (file_key in config):
            raise KeyError(f"Error : '{file_key}' are missing from config")

        if not (os.path.exists(config[file_key])):
            raise FileNotFoundError(f"Error: '{config[file_key]}' do not exist.")

    if not (config["is_ground"]):
        # check about singlet
        for file_name in check_list:
            file_key = f"singlet_{file_name}"
            if not (file_key in config):
                raise KeyError(f"Error : '{file_key}' are missing from config")

            if not (os.path.exists(config[file_key])):
                raise FileNotFoundError(f"Error: '{config[file_key]}' do not exist.")


def read_xyz(xyz_file):
    """Read xyz file and return atoms and coordinates"""
    with open(xyz_file, mode="r") as f:
        lines = [line.strip() for line in f.readlines()]
    atoms = []
    coordinates = []
    for line in lines:
        tmp = line.strip().split()
        if len(tmp) == 0:
            continue
        atoms.append(tmp[0])
        coordinates.append([float(tmp[1]), float(tmp[2]), float(tmp[3])])
    return atoms, coordinates


def main():
    parser = argparse.ArgumentParser(
        description="Run the main program with a specified configuration file."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    default_config = {
        "zeff_type": "orca",
        "deriv": True,
        "dump": False,
        "triplet": "",
        "singlet": "",
        "xyz_file": "",
    }

    config_file = args.config
    loaded_config = load_config(config_file)
    config = merge_configs(default_config, loaded_config)

    logging.info(f"Loaded configuration:{config_file}")
    for key, value in config.items():
        logging.info(f"{key}: {value}, {type(value)}")
    config["is_ground"] = not (bool(int(config.get("singlet")[1:])))
    check_file_exist(config)

    if not (config.get("dump", True)):
        triplet_mol_name = os.path.splitext(config["triplet_log_file"])[0]
        extract_info(
            triplet_mol_name,
            config["triplet_log_file"],
            config["triplet_rwf_file"],
            deriv=config["deriv"],
        )
        logging.info("Extracted triplet state infomation")
        config["triplet_json_file"] = f"{triplet_mol_name}_log.json"
        config["triplet_npz_file"] = f"{triplet_mol_name}_mat.npz"
        if not (config["is_ground"]):
            singlet_mol_name = os.path.splitext(config["singlet_log_file"])[0]
            extract_info(
                singlet_mol_name,
                config["singlet_log_file"],
                config["singlet_rwf_file"],
                deriv=config["deriv"],
            )
            logging.info("Extracted singlet state information")
            config["singlet_json_file"] = f"{singlet_mol_name}_log.json"
            config["singlet_npz_file"] = f"{singlet_mol_name}_mat.npz"
    triplet_parser = decode_gaussian_parser(
        config["triplet_json_file"], config["triplet_npz_file"]
    )

    atoms, coordinates = read_xyz(config["xyz_file"])

    if config["is_ground"]:
        soc_sntn = calc_soc_s0t1(
            atoms,
            coordinates,
            triplet_parser,
            basis=triplet_parser._basis,
            Z=config["zeff_type"],
        )
    else:
        singlet_parser = decode_gaussian_parser(
            config["singlet_json_file"], config["singlet_npz_file"]
        )
        soc_sntn = calc_soc_s1t1(
            atoms,
            coordinates,
            singlet_parser,
            triplet_parser,
            basis=triplet_parser._basis,
            Z=config["zeff_type"],
        )

    soc_tnsn = soc_sntn.conj()

    # Print the SOCME to file
    soc_tnsn_file = f"soc_{config['triplet']}_{config['singlet']}.data"
    with open(soc_tnsn_file, mode="w") as f:
        tmp = ""
        for i in range(3):
            tmp += f"{soc_tnsn[i].real:15.6e} {soc_tnsn[i].imag:15.6e}"
        f.write(tmp)

    if config["deriv"]:
        if config["is_ground"]:
            vsoc_sntn = calc_soc_s0t1_deriv(
                atoms,
                coordinates,
                triplet_parser,
                basis=triplet_parser._basis,
                Z=config["zeff_type"],
            )
        else:
            vsoc_sntn = calc_soc_s1t1_deriv(
                atoms,
                coordinates,
                singlet_parser,
                triplet_parser,
                basis=triplet_parser._basis,
                Z=config["zeff_type"],
            )

        vsoc_tnsn = vsoc_sntn.conj()

        # Print the SOC derivative to file.
        vsoc_tnsn_file = f"vsoc_{config['triplet']}_{config['singlet']}.data"
        with open(vsoc_tnsn_file, mode="w") as f:
            for i in range(vsoc_tnsn.shape[0]):
                for j in range(vsoc_tnsn.shape[1]):
                    tmp = ""
                    for k in range(3):
                        tmp += f"{vsoc_tnsn[i, j, k].real:15.6e} {vsoc_tnsn[i, j, k].imag:15.6e}"
                    f.write(tmp + "\n")


if __name__ == "__main__":
    main()
