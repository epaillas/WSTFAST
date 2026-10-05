#!/usr/bin/env python
"""Monte Carlo check of the NLO Edgeworth correction to E|X|^q for O(n)-invariant moduli.

X = g + lam T(g g - 1) + lam^2 A g (|g|^2 - n) with g ~ N(0, I_n). The predicted
correction q(q-2) K4 / (8 n (n+2)) + q(q-2)(q-4) (6 K33 + 9 K3v) / (72 n (n+2) (n+4))
must match the Monte Carlo ratio E|X|^q / E_G|X|^q - 1 up to O(lam^4).
"""

import numpy as np

from wst_feasibility import edgeworth_nlo, gaussian_norm_moment


def main(q=0.8, nsamp=2_000_000, nrep=8, seed=0):
    rng = np.random.default_rng(seed)
    for n in (3, 5):
        t = rng.standard_normal((n, n, n))
        t = (t + t.transpose(0, 2, 1)) / 2
        a = rng.standard_normal((n, n)) * 0.3
        for lam in (0.05, 0.1):
            chunks = []
            for _ in range(nrep):
                g = rng.standard_normal((nsamp, n))
                gg = np.einsum("ib,ic->ibc", g, g) - np.eye(n)[None]
                chunks.append(g + lam * np.einsum("abc,ibc->ia", t, gg)
                              + lam**2 * (g @ a.T) * ((g**2).sum(1, keepdims=True) - n))
            x = np.concatenate(chunks)
            x -= x.mean(0)
            npts = len(x)
            cov = x.T @ x / npts
            s2 = np.trace(cov) / n
            z = (x**2).sum(1)
            k4 = (z.var() - 2 * np.sum(cov**2)) / s2**2
            kap = np.einsum("ia,ib,ic->abc", x, x, x, optimize=True) / npts / s2**1.5
            k33 = np.sum(kap**2)
            k3v = np.sum(np.einsum("aac->c", kap) ** 2)
            zq = z ** (q / 2)
            ratio = zq.mean() / gaussian_norm_moment(np.linalg.eigvalsh(cov), q) - 1
            err = zq.std() / np.sqrt(npts) / zq.mean()
            c4, c6 = edgeworth_nlo(n, k4, k33, k3v, q)
            print(f"n={n} lam={lam}: MC {ratio:+.5f} +- {err:.5f}   NLO {c4 + c6:+.5f}")


if __name__ == "__main__":
    main()
