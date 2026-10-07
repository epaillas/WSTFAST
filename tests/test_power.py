"""Checks of the vendored dsc-model matter loop, the lattice binning and the P(k) desilike graph."""

from dataclasses import replace

import numpy as np
import pytest

import wstfast.theory  # noqa: F401  (enables JAX double precision)
from wstfast.measure import Lattice, PowerMultipoles
from wstfast.theory.eft_loop import EFTLoopConvention, LinearSpectrum, Quadrature, loop_integrals, real_space_p13
from wstfast.theory.perturbation import OneLoopMatter, log_interpolator
from wstfast.theory.power import LatticeBinning, default_knodes, matter_power_terms


@pytest.fixture
def power():
    k = np.geomspace(1e-6, 100.0, 4096)
    return LinearSpectrum(k, 1e6 * k / (1 + (k / 0.1) ** 2) ** 2)


def test_p13_matches_analytic_angular_integral(power):
    """dsc-model's test_analytic_p13_and_angular_projection, regulated and not."""
    k = np.array([0.005, 0.02, 0.06, 0.1])
    quadrature = Quadrature(nq=192, nx=48, qmax=2.0)
    for convention in (EFTLoopConvention(), None):
        direct = loop_integrals(k, [0.0], power, 0.0, convention, quadrature, rsd=False)
        analytic = real_space_p13(k, power, convention, quadrature)
        np.testing.assert_allclose(direct["13"][:, 0], analytic, rtol=2e-7)


def test_unregulated_loop_matches_wst_one_loop(power):
    """With convention=None the vendored loop is the SPT loop of the WST model."""
    k = np.array([0.02, 0.1, 0.2])
    vendored = loop_integrals(k, [0.0], power, 0.0, None, replace(Quadrature(), qmin=1e-4), rsd=False)["loop"][:, 0]
    klin = np.geomspace(1e-4, 10.0, 1024)
    wst = np.asarray(OneLoopMatter(k)(log_interpolator(klin, power(klin))))
    np.testing.assert_allclose(vendored, wst, atol=1e-3 * power(k).max())


def test_regulator_is_a_counterterm_at_low_k(power):
    """Lambda only shifts the loop by ~ k^2 P_L where k << Lambda."""
    k = np.array([0.01, 0.02, 0.03])
    regulated, spt = (loop_integrals(k, [0.0], power, 0.0, c, rsd=False)["loop"][:, 0] for c in (EFTLoopConvention(), None))
    ratio = (regulated - spt) / (k**2 * power(k))
    np.testing.assert_allclose(ratio, ratio[0], rtol=2e-2)
    with pytest.raises(ValueError):
        loop_integrals(np.array([0.6]), [0.0], power, 0.0, EFTLoopConvention(), rsd=False)


def test_tree_terms_are_linear(power):
    knodes = default_knodes(0.1)
    terms = matter_power_terms(power.k, power.power, knodes, order="tree")
    np.testing.assert_allclose(terms[0], power(knodes))
    assert np.all(terms[1] == 0.0)


def test_lattice_binning_matches_power_multipoles():
    nmesh, boxsize = 32, 100.0
    lattice = Lattice(nmesh)
    spectra = PowerMultipoles(lattice, boxsize, kmax=0.8)
    binning = LatticeBinning(spectra.edges, boxsize, nmesh, default_knodes(spectra.edges[-1], boxsize), window=None)
    np.testing.assert_allclose(binning.k, spectra.k, rtol=1e-6)
    np.testing.assert_array_equal(binning.nmodes, spectra.nmodes)
    # The spline is exact for a quadratic: binning k^2 gives the bin mean of |k|^2 (monopole of <k^2 1>).
    kphys2 = (lattice.kmag * nmesh / boxsize) ** 2
    expected = spectra(kphys2, np.ones_like(kphys2))[0] / spectra.norm
    np.testing.assert_allclose(binning(binning.knodes**2), expected, rtol=1e-5)
    np.testing.assert_allclose(binning(np.ones_like(binning.knodes), shotnoise=2.0), 3.0)
    # The CIC window and its aliased shot noise tend to 1 at low k, decrease, and agree to O(k^4 H^4).
    cic = LatticeBinning(spectra.edges, boxsize, nmesh, binning.knodes)
    assert np.all(np.diff(cic.noise) < 0) and cic.noise[0] > 0.98
    low = cic.k < 0.5 * np.pi * nmesh / boxsize
    np.testing.assert_allclose(cic.matrix.sum(axis=1)[low], cic.noise[low], rtol=1e-2)


@pytest.mark.slow
def test_power_likelihood_graph_evaluates():
    from desilike import build

    from wstfast.calculators import PowerBasis, PowerTheory, WSTLikelihood, build_cosmology, power_rows

    knodes = default_knodes(0.12)
    edges = 2 * np.pi / 1000.0 * np.arange(0.5, 19.0, 2.0)
    binning = LatticeBinning(edges, 1000.0, 256, knodes)
    basis = PowerBasis(cosmo=build_cosmology(("omega_cdm", "logA")), knodes=knodes, z=0.5)
    theory = PowerTheory(binning, basis=basis, shotnoise=7.45)
    graph = build(theory)
    prediction = np.asarray(graph({}))
    assert prediction.shape == (edges.size - 1,) and np.all(prediction > 0)
    likelihood = WSTLikelihood(theory, prediction, np.diag((0.01 * prediction) ** 2))
    assert float(build(likelihood)({})) == pytest.approx(0.0, abs=1e-8)
    # The basis is normalised to the fiducial amplitude: P_L scales as A_s and the loop as A_s^2.
    basis_graph = build(basis)
    low, high = (basis_graph(dict(logA=value)) for value in (3.0, 3.1))
    np.testing.assert_allclose(np.asarray(low.pk_terms), np.asarray(high.pk_terms), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(power_rows(high)[1]) / np.asarray(power_rows(low)[1]), np.exp(0.2), rtol=1e-10)
