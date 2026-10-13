"""The perturbation-theory tracer fields of scripts/feasibility/second_layer_rsd.py against the analytic redshift-space
kernels of wstfast.theory.bias: plane-wave inputs must give Z1, Z2 and Z3 (with the cubic operators) exactly."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/feasibility"))

from pt_modulus_noise import SPTFields  # noqa: E402
from second_layer_rsd import HaloRSDFields  # noqa: E402
from wstfast.measure import Lattice  # noqa: E402
from wstfast.theory.bias import cubic_kernel, galaxy_kernel  # noqa: E402

N = 32
BIASES = dict(b1=1.9, b2=-0.6, bG2=-0.6, bGamma3=0.6, b3=0.4, bG3=-0.3, bdG2=0.2, alpha0=0.0)
F = 0.76
BETA = np.array([1.0, BIASES["b1"], BIASES["b2"], BIASES["bG2"], BIASES["bGamma3"]])
CUBIC = np.array([BIASES["b3"], BIASES["bG3"], BIASES["bdG2"]])


def fields(modes):
    lattice = Lattice(N)
    spt = SPTFields.__new__(SPTFields)
    k, kz = 2 * np.pi * np.fft.fftfreq(N), 2 * np.pi * np.fft.rfftfreq(N)
    spt.kvec = np.meshgrid(k, k, kz, indexing="ij", sparse=True)
    spt.k2 = lattice.kmag.astype("f8") ** 2
    spt.k2[0, 0, 0] = 1.0
    spt.n = N
    delta1 = np.zeros((N, N, N // 2 + 1), complex)
    for index, value in modes:
        delta1[index] = value
    spt.delta1 = delta1
    spt.delta2, theta2 = spt._second(delta1)
    spt.delta3 = spt._third(delta1, spt.delta2, theta2)
    return HaloRSDFields(spt, F, BIASES)


def wavevector(index):
    i, j, l = index
    return 2 * np.pi * np.array([np.fft.fftfreq(N)[i], np.fft.fftfreq(N)[j], np.fft.rfftfreq(N)[l]])


@pytest.mark.parametrize("i1, i2", [((1, 2, 1), (3, 0, 2)), ((0, 1, 3), (2, 31, 1)), ((5, 3, 2), (30, 1, 1))])
def test_second_order_matches_z2(i1, i2):
    c1, c2 = 0.7 + 0.2j, -0.4 + 0.9j
    target = tuple((np.array(i1) + np.array(i2)) % N)
    z2 = galaxy_kernel(np.array([wavevector(i1), wavevector(i2)]), F) @ BETA
    assert fields([(i1, c1), (i2, c2)]).s2[target] == pytest.approx(2 * z2 * c1 * c2 / N**3, rel=1e-6)


@pytest.mark.parametrize("i1, i2, i3", [((1, 2, 1), (3, 0, 2), (0, 1, 1)), ((2, 1, 1), (0, 3, 2), (1, 30, 1))])
def test_third_order_matches_z3(i1, i2, i3):
    c1, c2, c3 = 0.7 + 0.2j, -0.4 + 0.9j, 0.3 - 0.5j
    target = tuple((np.array(i1) + np.array(i2) + np.array(i3)) % N)
    vectors = np.array([wavevector(i1), wavevector(i2), wavevector(i3)])
    z3 = galaxy_kernel(vectors, F) @ BETA + cubic_kernel(vectors) @ CUBIC
    s3 = fields([(i1, c1), (i2, c2), (i3, c3)]).s3[target]
    assert s3 == pytest.approx(6 * z3 * c1 * c2 * c3 / N**6, rel=1e-6)


def test_first_order_is_kaiser():
    k = wavevector((1, 2, 1))
    assert fields([((1, 2, 1), 1.0)]).s1[1, 2, 1] == pytest.approx(BIASES["b1"] + F * k[2] ** 2 / (k @ k), rel=1e-6)
