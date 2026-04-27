#!/usr/bin/env python3

import os
import yaml
import logging
import argparse

from pygraso.parser import decode_gaussian_parser
from pygraso.preprocessing import extract_info
from pygraso.calc_soc import (
    calc_soc_s0t1,
    calc_soc_s0t1_deriv,
    calc_soc_s1t1,
    calc_soc_s1t1_deriv,
)
from pygraso.calc_nac import calc_nac_tntm, calc_nac_t0tm, calc_nac_tnt0


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
    if config["coupling_type"] == "SOC":
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
                    raise FileNotFoundError(
                        f"Error: '{config[file_key]}' do not exist."
                    )
    elif config["coupling_type"] == "NAC":
        if bool(int(config.get("state1")[1:])):
            for file_name in check_list:
                file_key = f"state1_{file_name}"
                if not (file_key in config):
                    raise KeyError(f"Error : '{file_key}' are missing from config")

                if not (os.path.exists(config[file_key])):
                    raise FileNotFoundError(
                        f"Error: '{config[file_key]}' do not exist."
                    )

        if bool(int(config.get("state2")[1:])):
            for file_name in check_list:
                file_key = f"state2_{file_name}"
                if not (file_key in config):
                    raise KeyError(f"Error : '{file_key}' are missing from config")

                if not (os.path.exists(config[file_key])):
                    raise FileNotFoundError(
                        f"Error: '{config[file_key]}' do not exist."
                    )


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
        "coupling_type": "SOC",
        "method": 1,
        "dump": False,
        "charge": 0,
        "xyz_file": "",
    }

    soc_default_config = {
        "zeff_type": "orca",
        "deriv": True,
        "triplet": "",
        "singlet": "",
    }
    nac_default_config = {
        "state1": "",
        "state2": "",
    }

    config_file = args.config
    loaded_config = load_config(config_file)
    config = merge_configs(default_config, loaded_config)
    coupling_type = config.get("coupling_type")
    if coupling_type == "SOC":
        config = merge_configs(soc_default_config, config)
    if coupling_type == "NAC":
        config = merge_configs(nac_default_config, config)

    logging.info(f"Loaded configuration:{config_file}")
    for key, value in config.items():
        logging.info(f"{key}: {value}, {type(value)}")

    if coupling_type == "SOC":
        config["is_ground"] = not (bool(int(config.get("singlet")[1:])))
        check_file_exist(config)

        if not (config.get("dump", True)):
            triplet_mol_name = os.path.splitext(config["triplet_log_file"])[0]
            extract_info(
                triplet_mol_name,
                config["triplet_log_file"],
                config["triplet_rwf_file"],
                method=config["method"],
                deriv=config["deriv"],
                state=int(config["triplet"][1:]),
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
                    method=config["method"],
                    deriv=config["deriv"],
                    state=int(config["singlet"][1:]),
                )
                logging.info("Extracted singlet state information")
                config["singlet_json_file"] = f"{singlet_mol_name}_log.json"
                config["singlet_npz_file"] = f"{singlet_mol_name}_mat.npz"
        triplet_parser = decode_gaussian_parser(
            config["triplet_json_file"], config["triplet_npz_file"]
        )

        atoms, coordinates = read_xyz(config["xyz_file"])
        charge = int(config["charge"])

        if config["is_ground"]:
            soc_sntn = calc_soc_s0t1(
                atoms,
                coordinates,
                charge,
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
                charge,
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
                    charge,
                    triplet_parser,
                    basis=triplet_parser._basis,
                    Z=config["zeff_type"],
                )
            else:
                vsoc_sntn = calc_soc_s1t1_deriv(
                    atoms,
                    coordinates,
                    charge,
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
    elif coupling_type == "NAC":
        check_file_exist(config)
        is_state1_ground = not (bool(int(config.get("state1")[1:])))
        is_state2_ground = not (bool(int(config.get("state2")[1:])))
        if not (config.get("dump", True)):
            if not (is_state1_ground):
                state1_mol_name = os.path.splitext(config["state1_log_file"])[0]
                extract_info(
                    state1_mol_name,
                    config["state1_log_file"],
                    config["state1_rwf_file"],
                    method=config["method"],
                    deriv=True,
                    state=int(config["state1"][1:]),
                )
                logging.info("Extracted state1 state infomation")
                config["state1_json_file"] = f"{state1_mol_name}_log.json"
                config["state1_npz_file"] = f"{state1_mol_name}_mat.npz"
            if not (is_state2_ground):
                state2_mol_name = os.path.splitext(config["state2_log_file"])[0]
                extract_info(
                    state2_mol_name,
                    config["state2_log_file"],
                    config["state2_rwf_file"],
                    method=config["method"],
                    deriv=True,
                    state=int(config["state2"][1:]),
                )
                logging.info("Extracted state2 state information")
                config["state2_json_file"] = f"{state2_mol_name}_log.json"
                config["state2_npz_file"] = f"{state2_mol_name}_mat.npz"

        if not (is_state1_ground):
            state1_parser = decode_gaussian_parser(
                config["state1_json_file"], config["state1_npz_file"]
            )
        if not (is_state2_ground):
            state2_parser = decode_gaussian_parser(
                config["state2_json_file"], config["state2_npz_file"]
            )

        atoms, coordinates = read_xyz(config["xyz_file"])
        charge = int(config["charge"])

        if not (is_state1_ground) and not (is_state2_ground):
            nac_tntm = calc_nac_tntm(
                atoms,
                coordinates,
                charge,
                state1_parser,
                state2_parser,
                basis=state1_parser._basis,
            )
        elif is_state1_ground and not (is_state2_ground):
            nac_tntm = calc_nac_t0tm(
                atoms, coordinates, charge, state2_parser, basis=state2_parser._basis
            )
        elif not (is_state1_ground) and is_state2_ground:
            nac_tntm = calc_nac_tnt0(
                atoms, coordinates, charge, state1_parser, basis=state1_parser._basis
            )
        else:
            raise ValueError("One of two states must be excited state")

        # Print the NACME to file
        nac_tntm_file = f"NAC_{config['state1']}_{config['state2']}.data"
        with open(nac_tntm_file, mode="w") as f:
            for i in range(nac_tntm.shape[0]):
                tmp = f"{nac_tntm[i, 0]:+.5f} {nac_tntm[i, 1]:+.5f} {nac_tntm[i, 2]:+.5f}\n"
                f.write(tmp)


if __name__ == "__main__":
    main()
