import os
import subprocess
import numpy as np

from .parser import gaussian_perser
from .calc_ao_element import calc_ao_element
from .calc_soc import calc_soc_s0t1, calc_soc_s1t1


def generate_input_file(
    inp_file_name,
    atoms,
    coordinates,
    mol_name,
    do_triplet=False,
    calc_type="td",
    functional="PBE1PBE",
    basis="6-31G",
    old_chk="",
):
    with open(inp_file_name, "w") as f:
        if old_chk:
            f.write(f"%oldchk={old_chk}\n")
            guess = "Guess=Read"
        else:
            guess = ""
        if do_triplet:
            triplet = "Triplet,"
        else:
            triplet = ""
        root = f"#p {calc_type}({triplet}nstates=6,root=1,conver=6) {guess} {functional}/{basis} 6D 10F nosymm GFInput scf=tight"
        f.write(f"""%Chk={mol_name}.chk
%Mem=2GB
%rwf={mol_name}.rwf
%NProcShared=32
{root}

title

0 1
""")
        for atom, coord in zip(atoms, coordinates):
            f.write(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
        f.write("\n")


def perturb_coordinates(base_coords, atom_idx, axis, delta):
    new_coords = [list(coord) for coord in base_coords]
    new_coords[atom_idx][axis] += delta
    return new_coords


def five_point_derivative(energies, step_size):
    """
    Calculate numerical derivative using five point differential method

    Parameters:
    - energies: list of float
        list of energies at each point [E(-2delta), E(-delta)), E(+delta), E(+2delta)]
    - step_size: float
        step width delta

    Returns:
    - derivative: float
        result of numerical derivatlve
    """
    if len(energies) != 4:
        raise ValueError("fThe length of energies must be 4, but {len(energies)=}")

    coefficients = [
        1 / (12 * step_size),
        -8 / (12 * step_size),
        +8 / (12 * step_size),
        -1 / (12 * step_size),
    ]
    derivative = sum(c * e for c, e in zip(coefficients, energies))
    return derivative


class numerical_deriv:
    def __init__(
        self,
        mol_name,
        atoms,
        coordinates,
        step_size=0.001,
        calc_dir="num",
        calc_type="td",
        basis="6-31G",
    ):
        self.mol_name = mol_name
        self.atoms = atoms
        self.coordinates = coordinates
        self.basis = basis
        self.natoms = len(atoms)
        self.step_size = step_size
        self.calc_dir = calc_dir
        self.calc_type = calc_type
        os.makedirs(self.calc_dir, exist_ok=True)

    def execute_gaussian(self):
        axes = ["x", "y", "z"]
        delta_list = [
            -2 * self.step_size,
            -self.step_size,
            +self.step_size,
            +2 * self.step_size,
        ]
        for axis_idx, axis in enumerate(axes):
            for atom_idx in range(self.natoms):
                for delta in delta_list:
                    pert_mol_name_s1 = os.path.join(
                        self.calc_dir,
                        f"{self.mol_name}_s1_{axis}_atom{atom_idx:03d}_delta{delta:+.4f}",
                    )
                    pert_s1_inp_file_name = f"{pert_mol_name_s1}.gjf"
                    pert_s1_log_file_name = f"{pert_mol_name_s1}.log"
                    pert_s1_rwf_file_name = f"{pert_mol_name_s1}.rwf"
                    if (
                        os.path.isfile(pert_s1_inp_file_name)
                        and os.path.isfile(pert_s1_log_file_name)
                        and os.path.isfile(pert_s1_rwf_file_name)
                    ):
                        continue
                    pert_coords = perturb_coordinates(
                        self.coordinates, atom_idx, axis_idx, delta
                    )
                    generate_input_file(
                        pert_s1_inp_file_name,
                        self.atoms,
                        pert_coords,
                        pert_mol_name_s1,
                        do_triplet=False,
                        calc_type=self.calc_type,
                        basis=self.basis,
                    )
                    try:
                        subprocess.run(["g16", pert_s1_inp_file_name], check=True)
                    except subprocess.CalledProcessError as e:
                        raise ValueError(f"Error in Gaussian execution: {e}")

        for axis_idx, axis in enumerate(axes):
            for atom_idx in range(self.natoms):
                for delta in delta_list:
                    pert_mol_name_t1 = os.path.join(
                        self.calc_dir,
                        f"{self.mol_name}_t1_{axis}_atom{atom_idx:03d}_delta{delta:+.4f}",
                    )
                    pert_t1_inp_file_name = f"{pert_mol_name_t1}.gjf"
                    pert_t1_log_file_name = f"{pert_mol_name_t1}.log"
                    pert_t1_rwf_file_name = f"{pert_mol_name_t1}.rwf"
                    if (
                        os.path.isfile(pert_t1_inp_file_name)
                        and os.path.isfile(pert_t1_log_file_name)
                        and os.path.isfile(pert_t1_rwf_file_name)
                    ):
                        continue
                    pert_coords = perturb_coordinates(
                        self.coordinates, atom_idx, axis_idx, delta
                    )
                    generate_input_file(
                        pert_t1_inp_file_name,
                        self.atoms,
                        pert_coords,
                        pert_mol_name_t1,
                        do_triplet=True,
                        calc_type=self.calc_type,
                        basis=self.basis,
                    )
                    try:
                        subprocess.run(["g16", pert_t1_inp_file_name], check=True)
                    except subprocess.CalledProcessError as e:
                        raise ValueError(f"Error in Gaussian execution: {e}")

    def execute_num_deriv(self, property_name, target_state="s1"):
        allow_property_names = [
            "mo_coeff",
            "xy_coeff",
            "tdip",
            "ao_soc",
            "soc_s0t1",
            "soc_s1t1",
        ]
        if not (property_name in allow_property_names):
            raise ValueError(f"Can't find {property_name}")

        self.execute_gaussian()
        axes = ["x", "y", "z"]
        delta_list = [
            -2 * self.step_size,
            -self.step_size,
            +self.step_size,
            +2 * self.step_size,
        ]
        gradients = [[] for _ in range(self.natoms)]
        for axis_idx, axis in enumerate(axes):
            for atom_idx in range(self.natoms):
                properties = []
                for delta in delta_list:
                    pert_mol_name = os.path.join(
                        self.calc_dir,
                        f"{self.mol_name}_{target_state}_{axis}_atom{atom_idx:03d}_delta{delta:+.4f}",
                    )
                    pert_inp_file_name = f"{pert_mol_name}.gjf"
                    pert_log_file_name = f"{pert_mol_name}.log"
                    pert_rwf_file_name = f"{pert_mol_name}.rwf"

                    g_parser = gaussian_perser(pert_log_file_name, pert_rwf_file_name)
                    pert_coords = perturb_coordinates(
                        self.coordinates, atom_idx, axis_idx, delta
                    )
                    if property_name == "mo_coeff":
                        prop = g_parser.get_mo_coeff()
                    if property_name == "xy_coeff":
                        prop = g_parser.get_xy_coeff()
                        if isinstance(prop, tuple):
                            self.calc_type = "td"
                            prop = np.hstack((prop[0], prop[1]))
                    if property_name == "tdip":
                        prop = g_parser.get_tdip()
                    if property_name == "ao_soc":
                        ao_calculator = calc_ao_element(
                            self.atoms, pert_coords, basis=self.basis
                        )
                        prop = ao_calculator.get_ao_soc()
                    if property_name == "soc_s0t1":
                        pert_mol_name_t1 = os.path.join(
                            self.calc_dir,
                            f"{self.mol_name}_t1_{axis}_atom{atom_idx:03d}_delta{delta:+.4f}",
                        )
                        pert_t1_inp_file_name = f"{pert_mol_name_t1}.gjf"
                        pert_t1_log_file_name = f"{pert_mol_name_t1}.log"
                        pert_t1_rwf_file_name = f"{pert_mol_name_t1}.rwf"
                        prop = calc_soc_s0t1(
                            self.atoms,
                            pert_coords,
                            pert_t1_log_file_name,
                            pert_t1_rwf_file_name,
                            basis=self.basis,
                        )
                    if property_name == "soc_s1t1":
                        pert_mol_name_t1 = os.path.join(
                            self.calc_dir,
                            f"{self.mol_name}_t1_{axis}_atom{atom_idx:03d}_delta{delta:+.4f}",
                        )
                        pert_t1_inp_file_name = f"{pert_mol_name_t1}.gjf"
                        pert_t1_log_file_name = f"{pert_mol_name_t1}.log"
                        pert_t1_rwf_file_name = f"{pert_mol_name_t1}.rwf"

                        pert_mol_name_s1 = os.path.join(
                            self.calc_dir,
                            f"{self.mol_name}_s1_{axis}_atom{atom_idx:03d}_delta{delta:+.4f}",
                        )
                        pert_s1_inp_file_name = f"{pert_mol_name_s1}.gjf"
                        pert_s1_log_file_name = f"{pert_mol_name_s1}.log"
                        pert_s1_rwf_file_name = f"{pert_mol_name_s1}.rwf"
                        prop = calc_soc_s1t1(
                            self.atoms,
                            pert_coords,
                            pert_s1_log_file_name,
                            pert_s1_rwf_file_name,
                            pert_t1_log_file_name,
                            pert_t1_rwf_file_name,
                            basis=self.basis,
                        )
                    properties.append(prop)
                grad = five_point_derivative(properties, self.step_size)
                gradients[atom_idx].append(grad)
        gradients = np.array(gradients)
        a2b = 1.8897259886  # angstrom to bohr
        gradients /= a2b
        if property_name == "xy_coeff" and self.calc_type == "td":
            nva = gradients.shape[-1] // 2
            return gradients[:, :, :, :nva], gradients[:, :, :, nva:]
        return gradients
