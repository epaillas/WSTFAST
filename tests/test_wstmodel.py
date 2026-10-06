"""Fast checks of the measurement, the theory building blocks and the desilike graph."""

import numpy as np
import pytest
from scipy.special import gammaln

import wstmodel  # noqa: F401
from wstmodel import WSTConfig, select_coefficients
from wstmodel.measure import Lattice, PowerMultipoles, measure_wst
from wstmodel.theory import Assembly, all_coefficients
from wstmodel.theory.perturbation import OneLoopMatter, log_interpolator


def gaussian_field(nmesh=64, seed=1):
    lattice = Lattice(nmesh)
    k = lattice.kmag
    pk = np.where(k > 0, (k + 1e-2) ** -1.5 * np.exp(-((k / 1.5) ** 2)), 0.0)
    noise = np.fft.rfftn(np.random.default_rng(seed).standard_normal((nmesh,) * 3))
    return np.fft.irfftn(noise * np.sqrt(pk), s=(nmesh,) * 3).astype(np.float32), lattice


def test_s1_of_gaussian_field_is_chi_moment():
    delta, lattice = gaussian_field()
    config = WSTConfig(J=2, L=2, sigma0=0.8, cellsize=1.0)
    result = {name: value[0] for name, value in measure_wst(delta, config, lattice=lattice).items()}
    fk = np.fft.rfftn(delta)
    mult = np.full(fk.shape, 2.0)
    mult[..., 0] = 1.0
    mult[..., -1] = 1.0
    for j in range(config.J + 1):
        for ell in range(config.L + 1):
            n = 2 * ell + 1
            weight = lattice.radial(config.sigma0 * 2**j, ell) ** 2
            s2 = np.sum(mult * np.abs(fk) ** 2 * weight) / delta.size**2 / n  # Parseval
            expected = (2 * s2) ** (config.q / 2) * np.exp(gammaln((n + config.q) / 2) - gammaln(n / 2))
            assert result["S1"][j, ell] == pytest.approx(expected, rel=2e-2)
    assert np.isnan(result["S2"][1, 0, 0]) and np.isfinite(result["S2"][0, 1, 1])


def test_one_loop_signs():
    k = np.geomspace(1e-4, 10, 512)
    pk = log_interpolator(k, 2e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.5))
    loop = OneLoopMatter(np.array([0.05, 0.3]), nq=128, nq13=512)
    assert np.all(np.asarray(loop.p22(pk)) > 0)
    assert np.all(np.asarray(loop.p13(pk)) < 0)


def test_assembly_keeps_selection_order():
    config = WSTConfig()
    selected = select_coefficients(config)[::-1]
    assembly = Assembly(config, selected)
    basis = all_coefficients(config)
    n1 = sum(c.kind == "S1" for c in basis)
    s1_terms = np.tile(np.arange(1.0, n1 + 1.0)[:, None], (1, 4)) * np.array([1.0, 0.0, 0.0, 0.0])
    s21_terms = np.ones((len(basis) - n1, 2))
    out = np.asarray(assembly(s1_terms, s21_terms, 0.0, np.zeros(len(assembly.noise_ells))))
    s1_rows = {c: i for i, c in enumerate(c for c in basis if c.kind == "S1")}
    for value, c in zip(out, selected):
        if c.kind == "S1":
            n = 2 * c.ell + 1
            variance = (s1_rows[c] + 1.0) / n
            gamma = np.exp(gammaln((n + config.q) / 2) - gammaln(n / 2))
            assert value == pytest.approx(gamma * (2 * variance) ** (config.q / 2))


@pytest.mark.slow
def test_likelihood_graph_evaluates():
    from desilike import build

    from wstmodel.calculators import WSTLikelihood, WSTTheory, build_cosmology

    config = WSTConfig()
    coefficients = select_coefficients(config)
    theory = WSTTheory(coefficients, config=config, cosmo=build_cosmology(), shotnoise=7.45)
    prediction = np.asarray(build(theory)({}))
    assert prediction.shape == (len(coefficients),) and np.all(np.isfinite(prediction))
    likelihood = WSTLikelihood(theory, prediction, np.diag((0.01 * prediction) ** 2))
    assert float(build(likelihood)({})) == pytest.approx(0.0, abs=1e-8)


def test_half_octave_superset_contains_dyadic_configuration():
    delta, lattice = gaussian_field(nmesh=48)
    dyadic = WSTConfig(J=2, L=2, sigma0=0.8, cellsize=1.0)
    superset = WSTConfig(J=4, L=2, sigma0=0.8, step=2**0.5, min_dj=2, cellsize=1.0)
    qs = [0.5, 0.8]
    a = measure_wst(delta, dyadic, qs=qs, lattice=lattice)
    b = measure_wst(delta, superset, qs=qs, lattice=lattice, spectra=PowerMultipoles(lattice, 48.0, kmax=1.0))
    np.testing.assert_allclose(b["S1"][:, ::2], a["S1"], rtol=1e-5)
    for j1, j2 in dyadic.second_layer_pairs():
        np.testing.assert_allclose(b["S2"][:, 2 * j1, 2 * j2], a["S2"][:, j1, j2], rtol=1e-5)
    # First-layer auto-spectra are stored and positive.
    assert np.all(b["PUU"][:, :, 0] > 0)


def test_torch_backend_matches_numpy():
    torch = pytest.importorskip("torch")
    from wstmodel.measure_torch import TorchLattice, TorchPowerMultipoles, measure_wst_torch, paint_cic_torch
    from wstmodel.quijote import paint_cic

    delta, lattice = gaussian_field(nmesh=32)
    config = WSTConfig(J=3, L=3, L2=2, min_dj=2, sigma0=0.8, step=2**0.5, cellsize=1.0)
    qs = [0.5, 0.8, 2.0]
    reference = measure_wst(delta, config, qs=qs, lattice=lattice, spectra=PowerMultipoles(lattice, 32.0, kmax=1.5))
    tlattice = TorchLattice(32, device=torch.device("cpu"))
    result = measure_wst_torch(delta, config, qs=qs, lattice=tlattice,
                               spectra=TorchPowerMultipoles(tlattice, 32.0, kmax=1.5))
    assert set(result) == set(reference)
    for name in ("S0", "S1", "S2", "Umean", "Pdd", "PUU"):
        np.testing.assert_allclose(result[name], reference[name], rtol=2e-4, atol=1e-9)
    np.testing.assert_allclose(result["PUd"], reference["PUd"], rtol=2e-3, atol=1e-6 * np.abs(reference["PUU"]).max())

    positions = [np.random.default_rng(2).uniform(0, 100.0, (5000, 3))]
    np.testing.assert_allclose(paint_cic_torch(positions, 16, 100.0, device="cpu"), paint_cic(positions, 16, 100.0),
                               atol=1e-5)
