import os
import re
from pathlib import Path
import subprocess
import numpy as np
import periodictable
import json
import itertools

CHUNK_SIZE = 10**6


def flatten_to_symmetric(flat_array, nbasis):
    """
    Recover a two-dimensional symmetric matrix from one-dimensional upper triangular matrix data.

    Parameters:
        flat_array (np.ndarray): One-dimensional array containing only the upper triangular component.

    Returns:
        np.ndarray: Symmetric two-dimensional square matrix.
    """
    symmetric_matrix = np.zeros((nbasis, nbasis))
    indices = np.tril_indices(nbasis)
    symmetric_matrix[indices] = flat_array
    symmetric_matrix += np.tril(symmetric_matrix, -1).T
    return symmetric_matrix


def read_block(lines, start, nrow, ncol):
    tmp = []
    start += 1
    for j in range((ncol + 4) // 5):
        tmp1 = []
        for k in range(nrow):
            tmp1.append(
                [float(s.replace("D", "E")) for s in lines[start + k].split()[1:]]
            )
        tmp1 = np.array(tmp1).T
        tmp.extend(tmp1)
        start += nrow + 1
    tmp = np.array(tmp).T
    return tmp


def read_block2(lines, start, nrow, ncol):
    tmp = []
    for j in range((ncol + 4) // 5):
        tmp1 = []
        for k in range(nrow):
            tmp1.append([float(s.replace("D", "E")) for s in lines[start + k].split()])
        tmp1 = np.array(tmp1).T
        tmp.extend(tmp1)
        start += nrow + 1
    tmp = np.array(tmp).T
    return tmp


def search_by_chunk(file_name, key_list, method="start"):
    if isinstance(key_list, str):
        key_list = [key_list]
    method_list = ["start", "re"]
    assert method in method_list, f"method must be {method_list}, but {method}"

    iline = 0
    found = False
    with open(file_name, mode="r") as f:
        while chunk := list(itertools.islice(f, CHUNK_SIZE)):
            for key in key_list:
                if method == "start":
                    if any(line.startswith(key) for line in chunk):
                        found = True
                        break
                elif method == "re":
                    if any(re.match(key, line) for line in chunk):
                        found = True
                        break
            if found:
                break
            iline += CHUNK_SIZE

    if not chunk:
        raise ValueError(f"Can't find {key}")

    for i, line in enumerate(chunk):
        for key in key_list:
            if method == "start":
                if line.startswith(key):
                    return iline + i, line
            elif method == "re":
                if re.match(key, line):
                    return iline + i, line
    raise ValueError


def extract_lines_by_chunk(file_name, start_line, end_line):
    extracted_lines = []
    current_line = 0
    with open(file_name, mode="r") as f:
        while chunk := list(itertools.islice(f, CHUNK_SIZE)):
            if current_line + CHUNK_SIZE < start_line:
                current_line += CHUNK_SIZE
                continue
            if current_line >= end_line:
                break
            for i, line in enumerate(chunk):
                if current_line + i >= end_line:
                    break
                if current_line + i >= start_line:
                    extracted_lines.append(line.strip())
            current_line += CHUNK_SIZE

    return extracted_lines


class abstract_parser:
    def __init__(self):
        self.natoms = None
        self.nbasis = None
        self.nfc = None
        self.norb = None
        self.noa = None
        self.nva = None
        self.basis_idx = None
        self.nxy = None

        self._basis = None

        self._ao_ovlp = None
        self._ao_ovlp_deriv = None

        self._mo_coeff = None
        self._mo_coeff_U = None
        self._mo_coeff_deriv = None

        self._x_coeff = None
        self._x_tdens_deriv = None
        self._x_coeff_deriv = None

        self._y_coeff = None
        self._y_tdens_deriv = None
        self._y_coeff_deriv = None

        self._dip = None
        self._tdip = None


class gaussian_perser(abstract_parser):
    def __init__(self, log_file_name, rwf_file_name):
        super().__init__()

        assert os.path.isfile(log_file_name), f"{log_file_name=} doesn't exist"
        self.log_file_name = log_file_name
        assert os.path.isfile(rwf_file_name), f"{rwf_file_name=} doesn't exist"
        self.rwf_file_name = rwf_file_name

        (
            self.natoms,
            self.nbasis,
            self.nfc,
            self.norb,
            self.noa,
            self.nva,
            self.basis_idx,
            self.nxy,
        ) = self.extract_orb_info()
        self.rwf_parser = RWF_parser(self.rwf_file_name)

    def extract_orb_info(self):
        key = " NAtoms="
        natoms = int(
            search_by_chunk(self.log_file_name, key_list=" NAtoms=", method="start")[
                1
            ].split()[1]
        )
        pattern = r" NBasis=\s+\d+\s+NAE=\s+\d+\s+NBE=\s+\d+\s+NFC=\s+\d+\s+NFV=\s+\d+$"
        basis_line = (
            search_by_chunk(self.log_file_name, key_list=pattern, method="re")[1]
            .strip()
            .split()
        )
        nbasis, nfc = int(basis_line[1]), int(basis_line[7])
        pattern = r" NROrb=\s+\d+\s+NOA=\s+\d+\s+NOB=\s+\d+\s+NVA=\s+\d+\s+NVB=\s+\d+$"
        orb_line = (
            search_by_chunk(self.log_file_name, key_list=pattern, method="re")[1]
            .strip()
            .split()
        )
        norb, noa, nob, nva, nvb = (
            int(orb_line[1]),
            int(orb_line[3]),
            int(orb_line[5]),
            int(orb_line[7]),
            int(orb_line[9]),
        )
        assert noa == nob, "NOA and NOB must be same"
        assert nva == nvb, "NVA and NVB must be same"
        assert nbasis == nfc + noa + nva, (
            f"must satisfy following relationship, nbasis == nfc + noa + nva, but {nbasis=} {nfc=} {noa=} {nva=}"
        )

        key = (
            " AO basis set in the form of general basis input (Overlap normalization):"
        )
        start_line = search_by_chunk(self.log_file_name, key_list=key, method="start")[
            0
        ]
        pattern = r"\s*\d+\s+basis functions,\s+\d+\s+primitive gaussians,\s+\d+\s+cartesian basis functions"
        end_line = (
            search_by_chunk(self.log_file_name, key_list=pattern, method="re")[0] - 1
        )
        extract_lines = extract_lines_by_chunk(self.log_file_name, start_line, end_line)
        basis_line_idxs = [0] + [
            i for i, line in enumerate(extract_lines) if line == "****"
        ]
        assert len(basis_line_idxs) == natoms + 1, "length of Basis idx must be natoms"
        basis_idx = []
        basis_num = {"S": 1, "P": 3, "SP": 4, "D": 6, "F": 10}
        for i, idx1 in enumerate(basis_line_idxs[:natoms]):
            idx2 = basis_line_idxs[i + 1]
            assert i + 1 == int(extract_lines[idx1 + 1].split()[0]), "wrong number"
            for j in range(idx1 + 1, idx2):
                if extract_lines[j].split()[0] in basis_num.keys():
                    basis_idx += [i] * basis_num[extract_lines[j].split()[0]]
        assert len(basis_idx) == nbasis, f"{len(basis_idx)=}, {nbasis=}"

        key_list = [" Total Energy, E(CIS/TDA) =", " Total Energy, E(TD-HF/TD-DFT) ="]
        line = search_by_chunk(self.log_file_name, key_list=key_list, method="start")[1]
        nxy = 0
        if line.startswith(key_list[0]):
            nxy = 1
        elif line.startswith(key_list[1]):
            nxy = 2

        return natoms, nbasis, nfc, norb, noa, nva, basis_idx, nxy

    def read_basis(self):
        if self._basis is not None:
            return self._basis
        atoms_key = " Center     Atomic      Atomic             Coordinates (Angstroms)"
        atoms_key_line = (
            search_by_chunk(self.log_file_name, key_list=atoms_key, method="start")[0]
            + 3
        )
        atoms_num = [
            int(line.split()[1])
            for line in extract_lines_by_chunk(
                self.log_file_name, atoms_key_line, atoms_key_line + self.natoms
            )
        ]
        atoms_symbol = [
            periodictable.elements[atom_num].symbol for atom_num in atoms_num
        ]
        key = (
            " AO basis set in the form of general basis input (Overlap normalization):"
        )
        start_line = search_by_chunk(self.log_file_name, key_list=key, method="start")[
            0
        ]
        pattern = r"\s*\d+\s+basis functions,\s+\d+\s+primitive gaussians,\s+\d+\s+cartesian basis functions"
        end_line = (
            search_by_chunk(self.log_file_name, key_list=pattern, method="re")[0] - 1
        )
        extract_lines = extract_lines_by_chunk(self.log_file_name, start_line, end_line)
        tmp_line = [0] + [
            i for i, line in enumerate(extract_lines) if line.startswith("****")
        ]
        assert len(tmp_line) - 1 == self.natoms, "number of basis doesn't match "
        basis_line = [(tmp_line[i] + 1, tmp_line[i + 1]) for i in range(self.natoms)]
        basis = dict()
        angular_sym2num = {"S": 0, "P": 1, "D": 2, "F": 3}
        for i in range(self.natoms):
            if atoms_symbol[i] in basis:
                continue
            basis[atoms_symbol[i]] = []
            now_line = basis_line[i][0] + 1
            while now_line < basis_line[i][1]:
                tmp = extract_lines[now_line].split()
                angular, nctr = tmp[0], int(tmp[1])
                if len(angular) == 1:
                    angular_num = angular_sym2num[angular]
                    coeff = read_block2(extract_lines, now_line + 1, nctr, 2)
                    basis[atoms_symbol[i]].append([angular_num] + coeff.tolist())
                elif angular == "SP":
                    coeff = read_block2(extract_lines, now_line + 1, nctr, 3)
                    basis[atoms_symbol[i]].append([0] + coeff[:, [0, 1]].tolist())
                    basis[atoms_symbol[i]].append([1] + coeff[:, [0, 2]].tolist())
                now_line += 1 + nctr
        self._basis = basis
        return self._basis

    def get_mo_coeff(self):
        if self._mo_coeff is not None:
            return self._mo_coeff
        try:
            mo_coeff = self.rwf_parser.parse(self.rwf_parser.MOA_COEFFS)
            mo_coeff = mo_coeff[: self.nbasis**2]
            mo_coeff = np.array([float(c.replace("D", "E")) for c in mo_coeff]).reshape(
                self.nbasis, self.nbasis
            )
            self._mo_coeff = mo_coeff
        except Exception as e:
            raise ValueError(f"Error occur while reading mo_coeff:{e}")
        return self._mo_coeff

    def get_ao_ovlp(self):
        if self._ao_ovlp is not None:
            return self._ao_ovlp
        try:
            num_AO = self.nbasis * (self.nbasis + 1) // 2
            ao_ovlp = self.rwf_parser.parse(self.rwf_parser.AO_OVERLAP, num_AO)
            ao_ovlp = [float(c.replace("D", "E")) for c in ao_ovlp]
            ao_ovlp = flatten_to_symmetric(ao_ovlp, self.nbasis)
            self._ao_ovlp = ao_ovlp
        except Exception as e:
            raise ValueError(f"Error occur while reading ao_ovlp:{e}")
        return self._ao_ovlp

    def get_ao_ovlp_deriv(self):
        if self._ao_ovlp_deriv is not None:
            self._ao_ovlp_deriv
        try:
            num_AO = self.nbasis * (self.nbasis + 1) // 2
            # ao_ovlp = self.rwf_parser.parse(self.rwf_parser.AO_OVERLAP, num_AO)
            # ao_ovlp = [float(c.replace("D", "E")) for c in ao_ovlp]
            # ao_ovlp = flatten_to_symmetric(ao_ovlp, self.nbasis)

            num_AO_deriv = num_AO * 3
            ao_ovlp_deriv = self.rwf_parser.parse(
                self.rwf_parser.AO_OVERLAP_DERIV, num_AO_deriv
            )
            ao_ovlp_deriv = [float(c.replace("D", "E")) for c in ao_ovlp_deriv]
            ao_ovlp_deriv_element = np.array(
                [
                    flatten_to_symmetric(
                        ao_ovlp_deriv[i * num_AO : (i + 1) * num_AO], self.nbasis
                    )
                    for i in range(3)
                ]
            )
            ao_ovlp_deriv = np.zeros((self.natoms, 3, self.nbasis, self.nbasis))
            for i in range(3):
                for j in range(self.nbasis):
                    for k in range(j + 1):
                        v = ao_ovlp_deriv_element[i, j, k]
                        if self.basis_idx[j] == self.basis_idx[k]:
                            continue
                        else:
                            ao_ovlp_deriv[self.basis_idx[j], i, j, k] = ao_ovlp_deriv[
                                self.basis_idx[j], i, k, j
                            ] = -v
                            ao_ovlp_deriv[self.basis_idx[k], i, j, k] = ao_ovlp_deriv[
                                self.basis_idx[k], i, k, j
                            ] = +v

            self._ao_ovlp_deriv = ao_ovlp_deriv
        except Exception as e:
            raise ValueError(f"Error occur while reading ao_ovlp_deriv:{e}")
        return self._ao_ovlp_deriv

    def get_mo_coeff_U(self):
        if self._mo_coeff_U is not None:
            return self._mo_coeff_U
        try:
            num_unique = self.nbasis * (self.nbasis + 1) // 2
            mo_f1 = self.rwf_parser.parse(
                self.rwf_parser.F1_CPHF, (3 + self.natoms * 3) * num_unique
            )[3 * num_unique :]
            mo_f1 = np.array([float(c.replace("D", "E")) for c in mo_f1])
            mo_f1 = np.array(
                [
                    flatten_to_symmetric(
                        mo_f1[i * num_unique : (i + 1) * num_unique], self.nbasis
                    )
                    for i in range(3 * self.natoms)
                ]
            )
            mo_f1 = mo_f1.reshape(self.natoms, 3, self.nbasis, self.nbasis)

            mo_energy = self.rwf_parser.parse(self.rwf_parser.MO_ENERGY, self.nbasis)
            mo_energy = np.array([float(c.replace("D", "E")) for c in mo_energy])
            delta = 1e-18
            ao_ovlp_deriv = self.get_ao_ovlp_deriv()
            mo_coeff = self.get_mo_coeff()
            mo_ovlp_deriv = (mo_coeff) @ ao_ovlp_deriv @ (mo_coeff.T)
            tmp1 = mo_f1 - mo_ovlp_deriv * mo_energy[None, None, None, :]
            tmp2 = mo_energy[None, :] - mo_energy[:, None] + delta
            mo_coeff_U = tmp1 / tmp2[None, None, :, :]

            for i in range(self.natoms):
                for j in range(3):
                    for k in range(self.nbasis):
                        mo_coeff_U[i, j, k, k] = -0.5 * mo_ovlp_deriv[i, j, k, k]

            self._mo_coeff_U = mo_coeff_U
        except Exception as e:
            raise ValueError(f"Error occur while reading mo_coeff_U:{e}")
        return self._mo_coeff_U

    def get_mo_coeff_deriv(self):
        if self._mo_coeff_deriv is not None:
            return self._mo_coeff_deriv
        try:
            mo_coeff = self.get_mo_coeff()
            mo_coeff_U = self.get_mo_coeff_U()
            mo_coeff_U = mo_coeff_U.transpose(0, 1, 3, 2)
            mo_coeff_deriv = mo_coeff_U @ mo_coeff

            self._mo_coeff_deriv = mo_coeff_deriv
        except Exception as e:
            raise ValueError(f"Error occur while reading mo_coeff_deriv:{e}")
        return self._mo_coeff_deriv

    def get_xy_coeff(self):
        if self._x_coeff is not None:
            if self.nxy == 1:
                return self._x_coeff
            elif (self.nxy == 2) and (self._y_coeff is not None):
                return self._x_coeff, self._y_coeff
        try:
            xy_coeff = self.rwf_parser.parse(self.rwf_parser.XY_COEFFS)
            xy_coeff = [float(v.replace("D", "E")) for v in "  ".join(xy_coeff).split()]
            dat_length = len(xy_coeff)
            nl = 12
            ndim = self.noa * self.nva
            mseek = int((dat_length - 12) / (ndim * 4 + 1))

            xpy_coeff = np.array(xy_coeff[nl : nl + ndim]).reshape(self.noa, self.nva)
            xpy_coeff = np.vstack((np.zeros((self.nfc, self.nva)), xpy_coeff))

            self._x_coeff = xpy_coeff
            if self.nxy == 2:
                nl += ndim * 2 * mseek
                xmy_coeff = np.array(xy_coeff[nl : nl + ndim]).reshape(
                    self.noa, self.nva
                )
                xmy_coeff = np.vstack((np.zeros((self.nfc, self.nva)), xmy_coeff))
                self._x_coeff = (xpy_coeff + xmy_coeff) / 2.0
                self._y_coeff = (xpy_coeff - xmy_coeff) / 2.0
        except Exception as e:
            raise ValueError(f"Error occur while reading xy_coeff:{e}")
        if self.nxy == 1:
            return self._x_coeff
        elif self.nxy == 2:
            return self._x_coeff, self._y_coeff

    def get_tdens_deriv(self):
        if self._x_tdens_deriv is not None:
            if self.nxy == 1:
                return self._x_tdens_deriv
            if (self.nxy == 2) and (self._y_tdens_deriv is not None):
                return self._x_tdens_deriv, self._y_tdens_deriv
        try:
            if self.nxy == 1:
                pattern = r" \[X\]\(x\) alpha: IBuc=(\d+) IX=\s+1 I=\s+1 J=\s+1:"
            elif self.nxy == 2:
                pattern = r" \[X,Y\]\(x\) alpha: IBuc=(\d+) IX=\s+1 I=\s+1 J=\s+1:"
            end_line = search_by_chunk(self.log_file_name, pattern, method="re")[0]
            stride = (self.nva + 1) * ((self.noa * self.nxy + 4) // 5) + 1
            start_line = end_line - stride * (self.natoms * 3)
            extract_lines = extract_lines_by_chunk(
                self.log_file_name, start_line, end_line
            )
            assert len(extract_lines) > 0, (
                "Can't find CPHF results line in logfile, please set iop(10/33=2)"
            )

            ilines = []
            key = "CPHF results for U (alpha) for IMat="
            for i, line in enumerate(extract_lines):
                if line.startswith(key):
                    ilines.append(i)

            x_coeff_deriv_tmp = []
            for iline in ilines:
                x_coeff_deriv_tmp.append(
                    read_block(
                        extract_lines, iline + 1, self.nva, self.noa * self.nxy
                    ).T
                )
            x_coeff_deriv_tmp = np.array(x_coeff_deriv_tmp)

            if self.nxy == 2:
                y_coeff_deriv_tmp = x_coeff_deriv_tmp[:, self.noa :, :]
                x_coeff_deriv_tmp = x_coeff_deriv_tmp[:, : self.noa, :]

            mo_coeff_U = self.rwf_parser.parse(self.rwf_parser.U_CPHF)
            mo_coeff_U = mo_coeff_U[
                3 * self.nbasis * self.nbasis : (3 + 3 * self.natoms)
                * self.nbasis
                * self.nbasis
            ]
            mo_coeff_U = np.array(
                [float(c.replace("D", "E")) for c in mo_coeff_U]
            ).reshape(self.natoms, 3, self.nbasis, self.nbasis)

            mo_coeff = self.get_mo_coeff()
            mo_coeff_deriv_tmp = (mo_coeff_U @ mo_coeff).reshape(
                self.natoms * 3, self.nbasis, self.nbasis
            )

            mo_coeff_deriv_tmp_i = mo_coeff_deriv_tmp[:, : self.nfc + self.noa, :]
            mo_coeff_deriv_tmp_a = mo_coeff_deriv_tmp[:, self.nfc + self.noa :, :]

            x_coeff = self.get_xy_coeff()
            if self.nxy == 2:
                y_coeff = x_coeff[1]  # Y, dexcitation amplitude
                x_coeff = x_coeff[0]  # X, excitation amplitude

            mo_coeff_i = mo_coeff[: self.nfc + self.noa, :]
            mo_coeff_a = mo_coeff[self.nfc + self.noa :, :]

            CPt = (mo_coeff_i.T) @ x_coeff
            PtCt = (mo_coeff_a.T) @ x_coeff.T

            x_tdens_deriv_1 = (
                -1.0 * (mo_coeff_i[self.nfc :, :].T) @ x_coeff_deriv_tmp @ mo_coeff_a
            )
            x_tdens_deriv_2 = CPt @ mo_coeff_deriv_tmp_a
            x_tdens_deriv_3 = (PtCt @ mo_coeff_deriv_tmp_i).transpose(0, 2, 1)

            x_tdens_deriv = x_tdens_deriv_1 + x_tdens_deriv_2 + x_tdens_deriv_3
            x_tdens_deriv *= 2.0
            x_tdens_deriv = x_tdens_deriv.reshape(
                self.natoms, 3, self.nbasis, self.nbasis
            )

            self._x_tdens_deriv = x_tdens_deriv

            if self.nxy == 2:
                CPt2 = (mo_coeff_a.T) @ y_coeff.T
                PtCt2 = (mo_coeff_i.T) @ y_coeff

                y_tdens_deriv_1 = -1.0 * (
                    mo_coeff_i[self.nfc :, :].T @ y_coeff_deriv_tmp @ mo_coeff_a
                ).transpose(0, 2, 1)
                y_tdens_deriv_2 = CPt2 @ mo_coeff_deriv_tmp_i
                y_tdens_deriv_3 = (PtCt2 @ mo_coeff_deriv_tmp_a).transpose(0, 2, 1)

                y_tdens_deriv = y_tdens_deriv_1 + y_tdens_deriv_2 + y_tdens_deriv_3
                y_tdens_deriv *= 2.0
                y_tdens_deriv = y_tdens_deriv.reshape(
                    self.natoms, 3, self.nbasis, self.nbasis
                )

                self._y_tdens_deriv = y_tdens_deriv
        except Exception as e:
            raise ValueError(f"Error occur while reading tdens_deriv:{e}")
        if self.nxy == 1:
            return self._x_tdens_deriv
        elif self.nxy == 2:
            return self._x_tdens_deriv, self._y_tdens_deriv

    def get_xy_coeff_deriv(self):
        if self._x_coeff_deriv is not None:
            if self.nxy == 1:
                return self._x_coeff_deriv
            elif self.nxy == 2 and (self._y_coeff_deriv is not None):
                return self._x_coeff_deriv, self._y_coeff_deriv
        try:
            x_tdens_deriv = self.get_tdens_deriv()
            if self.nxy == 2:
                y_tdens_deriv = x_tdens_deriv[1]
                x_tdens_deriv = x_tdens_deriv[0]
            x_tdens_deriv = 0.5 * x_tdens_deriv
            mo_coeff = self.get_mo_coeff()
            mo_coeff_deriv = self.get_mo_coeff_deriv()

            x_coeff = self.get_xy_coeff()
            if self.nxy == 2:
                y_coeff = x_coeff[1]  # Y, dexcitation amplitude
                x_coeff = x_coeff[0]  # X, excitation amplitude

            ao_ovlp = self.get_ao_ovlp()

            x_tdens_deriv2 = (
                mo_coeff[: self.nfc + self.noa, :].T
                @ x_coeff
                @ mo_coeff_deriv[:, :, self.nfc + self.noa :, :]
            )
            x_tdens_deriv3 = (
                mo_coeff[self.nfc + self.noa :, :].T
                @ x_coeff.T
                @ mo_coeff_deriv[:, :, : self.nfc + self.noa, :]
            ).transpose(0, 1, 3, 2)
            x_tdens_deriv -= x_tdens_deriv2 + x_tdens_deriv3

            x_coeff_deriv = (
                (mo_coeff[: self.nfc + self.noa, :] @ ao_ovlp)
                @ x_tdens_deriv
                @ (mo_coeff[self.nfc + self.noa :, :] @ ao_ovlp).T
            )
            self._x_coeff_deriv = x_coeff_deriv

            if self.nxy == 2:
                y_tdens_deriv = 0.5 * y_tdens_deriv
                y_tdens_deriv2 = (
                    mo_coeff[self.nfc + self.noa :, :].T
                    @ y_coeff.T
                    @ mo_coeff_deriv[:, :, : self.nfc + self.noa, :]
                )
                y_tdens_deriv3 = (
                    mo_coeff[: self.nfc + self.noa, :].T
                    @ y_coeff
                    @ mo_coeff_deriv[:, :, self.nfc + self.noa :, :]
                ).transpose(0, 1, 3, 2)
                y_tdens_deriv -= y_tdens_deriv2 + y_tdens_deriv3

                y_coeff_deriv = (
                    mo_coeff[self.nfc + self.noa :, :]
                    @ ao_ovlp
                    @ y_tdens_deriv
                    @ (mo_coeff[: self.nfc + self.noa, :] @ ao_ovlp).T
                ).transpose(0, 1, 3, 2)
                self._y_coeff_deriv = y_coeff_deriv

        except Exception as e:
            raise ValueError(f"Error occur while reading xy_coeff_deriv:{e}")
        if self.nxy == 1:
            return self._x_coeff_deriv
        elif self.nxy == 2:
            return self._x_coeff_deriv, self._y_coeff_deriv

    def get_dip(self):
        if self._dip is not None:
            return self._dip
        try:
            key = "Electric dipole moment (input orientation):"
            with open(self.log_file_name, mode="r") as f:
                lines = [line.strip() for line in f.readlines()]
            iline = [i for i, line in enumerate(lines) if line.startswith(key)]
            dip = np.zeros(3)
            for i in range(3):
                dip[i] = float(lines[iline[0] + 4 + i].split()[1].replace("D", "E"))
            self._dip = dip
        except Exception as e:
            raise ValueError(f"Error occur while reading dip:{e}")
        return self._dip

    def get_tdip(self):
        if self._tdip is not None:
            return self._tdip
        try:
            key = "Electronic transition elements"
            with open(self.log_file_name, mode="r") as f:
                lines = [line.strip() for line in f.readlines()]
            iline = [i for i, line in enumerate(lines) if line.startswith(key)]
            tdip = np.zeros(3)
            for i in range(3):
                tdip[i] = float(lines[iline[0] + 3 + i].split()[1].replace("D", "E"))
            self._tdip = tdip
        except Exception as e:
            raise ValueError(f"Error occur while reading tdip:{e}")
        return self._tdip

    def dump_json(self, dump_json_file):
        dump_dir = {
            "natoms": self.natoms,
            "nbasis": self.nbasis,
            "nfc": self.nfc,
            "norb": self.norb,
            "noa": self.noa,
            "nva": self.nva,
            "basis_idx": self.basis_idx,
            "nxy": self.nxy,
            "_basis": self._basis,
        }

        with open(dump_json_file, "w") as f:
            json.dump(dump_dir, f)


class decode_gaussian_parser(abstract_parser):
    def __init__(self, dump_json_file, dump_npz_file):
        with open(dump_json_file, "r") as f:
            dump_dir = json.load(f)
        for key, value in dump_dir.items():
            setattr(self, key, value)

        loaded_mat = np.load(dump_npz_file)
        for key, value in loaded_mat.items():
            setattr(self, "_" + key, value)


class RWF_parser:
    """
    Class for parsing data from gaussian binary files (.rwf and .chk).
    """

    # Path/command to Gaussian's rwfdump.
    RWFDUMP = "rwfdump"

    # A string that we look for in output files that indicates the real data is coming next.
    START_STRING = "Dump of file"

    # Some rwfdump codes.
    MO_ENERGY = "522R"
    MOA_COEFFS = "524R"
    MOB_COEFFS = "526R"
    XY_COEFFS = "635R"
    AO_OVERLAP = "514R"
    AO_OVERLAP_DERIV = "588R"
    U_CPHF = "651R"
    F1_CPHF = "596R"

    def __init__(self, rwf_file_name):
        """
        Constructor for RWF_parser objects.

        :param rwf_file_name: A file to read from. Both .rwf and .chk files are supported.
        """
        self.rwf_file_name = Path(rwf_file_name)

    def get_section(self, code):
        """
        Fetch a given section from a rwf file.

        The sections to parse are given by a code understood by rwfdump. Each code is up to 4 digits, followed by one of the following letters (taken from rwfdump.hlp):
            - I if the data is to be printed as (decimal) integers
            - H for hexadecimal
            - R for real
            - A for ascii

        The data is returned as a list of lines, and will not include the Gaussian preamble.

        :param code: A code identifying a section to extract.
        :return: A tuple of the 'Dump of file' line followed by a list of lines.
        """
        try:
            rwfdump_proc = subprocess.run(
                [self.RWFDUMP, self.rwf_file_name.name, "-", code],
                # Capture both stdout and stderr.
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=str(self.rwf_file_name.parent),
                check=True,
            )

            dumped_data = rwfdump_proc.stdout.split("\n")

            # dumped_data =  str(subprocess.check_output([self.RWFDUMP, self.rwf_file_name, "-", code], universal_newlines = True)).split("\n")
        except subprocess.CalledProcessError:
            # rwfdump failed to run, check to see if the given .rwf file actually exists.
            if not self.rwf_file_name.exists():
                raise Exception(
                    "Required Gaussian .rwf file '{}' does not exist".format(
                        self.rwf_file_name
                    )
                )
            else:
                raise

        # Find start line and remove everything before.
        try:
            start_pos = [
                index
                for index, line in enumerate(dumped_data)
                if self.START_STRING in line
            ][0]
        except IndexError:
            raise Exception(
                "Failed to parse rwfdump output from file '{}', could not find start of dump identified by '{}'".format(
                    self.rwf_file_name, self.START_STRING
                )
            )

        # Next, discard everything before the start line.
        dumped_data = dumped_data[start_pos:]

        # The next line is the header, which we'll keep.
        header = dumped_data.pop(0)

        # Return header and data.
        return header, dumped_data

    def parse(self, code, num_records=-1):
        """
        Fetch and parse given section from a rwf file.

        The sections to parse are given by a code understood by rwfdump. Each code is up to 4 digits, followed by one of the following letters (taken from rwfdump.hlp):
            - I if the data is to be printed as (decimal) integers
            - H for hexadecimal
            - R for real
            - A for ascii

        The dumped data will additionally be parsed into a number of records (a 1D list), based on num_lines and remaining_fields.

        :param code: A code identifying a section to extract.
        :param num_records: The number of records to return from the extracted data. A negative number (the default) will return all available records.
        :return: The processed data (as a 1D list).
        """
        # First, run RWFDUMP to get our data.
        dumped_data = self.get_section(code)[1]

        # Convert the remaining data to linear form.
        records = "  ".join(dumped_data).split()

        # Only keep the number of records we're interested in.
        if num_records > -1:
            records = records[:num_records]

        # And return the data.
        return records
