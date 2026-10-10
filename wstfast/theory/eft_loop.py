"""One-loop matter power spectrum of dsc-model (regulated EFT), vendored for real-space P(k) fits.

Copied from the desilike fork used by dsc-model (/Users/epaillas/code/desilike, branch
``codex/ds-refactor-jax`` at cbe2df7f), whose files are untracked upstream:

- ``theories/galaxy_clustering/density_split_eft/operators.py``: the EdS SPT kernel recursion
  and the matter Z_n kernels (``_Kernels``, ``matter_kernel``);
- ``.../integrals.py``: ``LinearSpectrum``, ``Quadrature``, ``radial_rule``, ``loop_integrals``,
  ``sigma2`` and the analytic cross-check ``real_space_p13``;
- ``.../renormalization.py``: ``EFTLoopConvention`` (exp4 regulator at cutoff Lambda).

Only the matter path is kept: the density-split operators, their joint contractions and the
renormalization subtractions (all zero for matter) are dropped. The code is otherwise unchanged,
so results match dsc-model's ``loop_integrals(k, mu, power, f, convention, quadrature, rsd=...)``.
``convention=None`` is unregulated SPT, which is the loop of ``wstfast.theory.perturbation``.
This is a NumPy reference implementation (not JAX-traceable); fits go through an emulator.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
from math import comb, factorial

import numpy as np
from scipy.interpolate import CubicSpline


# --- renormalization.py ---------------------------------------------------------------------

@dataclass(frozen=True)
class EFTLoopConvention:
    """Regulated loop: 13 weighted by exp(-(q/L)^4), 22 by exp(-(p/L)^4 - (r/L)^4 + (k/L)^4)."""

    cutoff: float = .5  # h/Mpc

    def __post_init__(self):
        if not np.isfinite(self.cutoff) or self.cutoff <= 0.:
            raise ValueError('cutoff must be finite and positive, in h/Mpc')

    def as_dict(self):
        return asdict(self)

    def covariance(self, k):
        return np.exp(-(np.asarray(k) / self.cutoff)**4)

    def weights(self, k, p, r):
        # Evaluate the ratio in log space, avoiding 0/0 for internal tails.
        if np.any(np.asarray(k) > self.cutoff):
            raise ValueError('reference output must satisfy k <= cutoff')
        return (np.exp(-(np.asarray(p)/self.cutoff)**4 - (np.asarray(r)/self.cutoff)**4
                       + (np.asarray(k)/self.cutoff)**4), self.covariance(p))


# --- operators.py ---------------------------------------------------------------------------

@lru_cache(None)
def _subsets(mask):
    """Nonempty proper subsets and the symmetry weight for each split."""
    n = mask.bit_count()
    return tuple((sub, mask ^ sub, 1.0 / comb(n, sub.bit_count()))
                 for sub in range(1, mask) if (sub & mask) == sub)


def _divide(numerator, denominator):
    """The continuous zero-group prescription used by the SPT recursion."""
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0.)


def _dot(a, b):
    return np.einsum("...i,...i->...", a, b)


def _product(left, right, operation=np.multiply):
    """Kernel convolution, including every partition and its symmetry weight."""
    out = {}
    for mask in left.keys() | right.keys():
        value = None
        for sub, rest, weight in _subsets(mask):
            if sub not in left or rest not in right:
                continue
            term = weight * operation(left[sub], right[rest])
            value = term if value is None else value + term
        if value is not None:
            out[mask] = value
    return out


class _Kernels:
    def __init__(self, vectors, los):
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim < 2 or vectors.shape[-1] != 3 or vectors.shape[-2] not in (1, 2, 3):
            raise ValueError("vectors must have shape (..., n, 3), with n equal to 1, 2, or 3")
        if not np.all(np.isfinite(vectors)) or np.any(np.sum(vectors**2, axis=-1) == 0.):
            raise ValueError("individual wavevectors must be finite and nonzero")
        los = np.asarray(los, dtype=float)
        if los.shape != (3,) or not np.all(np.isfinite(los)) or np.dot(los, los) == 0.:
            raise ValueError("los must be a finite nonzero three-vector")
        self.los = los / np.linalg.norm(los)
        self.n = vectors.shape[-2]
        self.full = (1 << self.n) - 1
        self.batch_shape = vectors.shape[:-2]
        self.momenta, self.norms, self.parallel = {}, {}, {}
        self.F, self.G = {}, {}
        for mask in range(1, self.full + 1):
            indices = [i for i in range(self.n) if mask & (1 << i)]
            momentum = np.sum(vectors[..., indices, :], axis=-2)
            self.momenta[mask] = momentum
            self.norms[mask] = _dot(momentum, momentum)
            self.parallel[mask] = np.einsum("...i,i->...", momentum, self.los)
            order = len(indices)
            if order == 1:
                self.F[mask] = self.G[mask] = np.ones(self.batch_shape)
                continue
            fs, gs = np.zeros(self.batch_shape), np.zeros(self.batch_shape)
            for sub, rest, weight in _subsets(mask):
                p, q = self.momenta[sub], self.momenta[rest]
                pq = _dot(p, q)
                alpha = 1. + _divide(pq, self.norms[sub])
                beta = _divide(self.norms[mask] * pq, 2. * self.norms[sub] * self.norms[rest])
                fs += weight * self.G[sub] * ((2 * order + 1) * alpha * self.F[rest]
                                            + 2 * beta * self.G[rest])
                gs += weight * self.G[sub] * (3 * alpha * self.F[rest]
                                            + 2 * order * beta * self.G[rest])
            denominator = (2 * order + 3) * (order - 1)
            # A mass/velocity perturbation at exactly zero total momentum has
            # zero kernel.  Enforce the limit before subsequent soft divisions.
            self.F[mask] = np.where(self.norms[mask] == 0., 0., fs / denominator)
            self.G[mask] = np.where(self.norms[mask] == 0., 0., gs / denominator)

    def matter(self, f):
        """Number-density redshift-space Z kernels from the exact mass map."""
        velocity = {mask: _divide(self.parallel[mask], self.norms[mask]) * self.G[mask]
                    for mask in self.momenta}
        output = dict(self.F)
        power = velocity
        for j in range(1, self.n + 1):
            density_power = _product(self.F, power)
            for mask in self.momenta:
                term = power.get(mask, 0.) + density_power.get(mask, 0.)
                output[mask] = output[mask] + (f * self.parallel[mask])**j / factorial(j) * term
            power = _product(power, velocity)
        return output


def _growth(f, rsd):
    if np.ndim(f) or not np.isfinite(f):
        raise ValueError("f must be a finite scalar")
    return float(f) if rsd else 0.


def spt_kernels(vectors):
    """Return symmetrized EdS density F_n and velocity-divergence G_n."""
    work = _Kernels(vectors, (0., 0., 1.))
    return work.F[work.full], work.G[work.full]


def matter_kernel(vectors, f=0., los=(0., 0., 1.), rsd=True):
    """Return the symmetrized matter Z_n (F_n in real space), n <= 3."""
    work = _Kernels(vectors, los)
    return work.matter(_growth(f, rsd))[work.full]


# --- integrals.py ---------------------------------------------------------------------------

class LinearSpectrum:
    """Positive log-log natural spline with explicit power-law tails."""
    def __init__(self,k,power,low_slope=None):
        k,power=np.asarray(k,dtype=float),np.asarray(power,dtype=float)
        if k.ndim!=1 or k.shape!=power.shape or len(k)<4 or np.any(k<=0) or np.any(np.diff(k)<=0) or np.any(power<=0) or not np.isfinite(k).all() or not np.isfinite(power).all():
            raise ValueError('linear spectrum must be positive, finite and strictly ordered')
        self.k,self.power=k.copy(),power.copy()
        self.spline=CubicSpline(np.log(k),np.log(power),bc_type='natural')
        self.low_slope=float(self.spline(np.log(k[0]),1) if low_slope is None else low_slope)
        self.high_slope=float(np.log(power[-1]/power[-2])/np.log(k[-1]/k[-2]))

    def __call__(self,k):
        k=np.asarray(k,dtype=float)
        if np.any(k<=0) or not np.isfinite(k).all():
            raise ValueError('spectrum evaluation requires positive finite k')
        x=np.log(k);lo,hi=np.log(self.k[[0,-1]])
        lp=self.spline(np.clip(x,lo,hi))
        lp=np.where(x<lo,np.log(self.power[0])+self.low_slope*(x-lo),lp)
        lp=np.where(x>hi,np.log(self.power[-1])+self.high_slope*(x-hi),lp)
        return np.exp(lp)


@dataclass(frozen=True)
class Quadrature:
    nq: int = 192
    nx: int = 32
    nphi: int = 12
    qmin: float = 1.e-5
    qmax: float = 10.
    chunk: int = 8

    def __post_init__(self):
        for name in ('nq','nx','nphi','chunk'):
            value=getattr(self,name)
            if isinstance(value,bool) or int(value)!=value or value<1:
                raise ValueError('quadrature sizes must be positive integers')
        if self.nq<16 or self.nx<4 or self.nphi<9 or not 0 < self.qmin < self.qmax or not np.isfinite(self.qmax):
            raise ValueError('insufficient or invalid reference quadrature')

    def as_dict(self):
        return asdict(self)


def radial_rule(qmin,qmax,nq,breaks=()):
    """Piecewise Gauss-Legendre in log p; weights integrate dlog p."""
    if not np.isfinite(qmin) or not np.isfinite(qmax) or not 0<qmin<qmax or isinstance(nq,bool) or int(nq)!=nq or nq<1:
        raise ValueError('invalid radial range or quadrature count')
    edges=np.unique(np.array([qmin,*[x for x in breaks if qmin<x<qmax],qmax]))
    nodes,weights=[],[]
    total=np.log(qmax/qmin)
    for lo,hi in zip(edges[:-1],edges[1:]):
        size=max(8,int(np.ceil(nq*np.log(hi/lo)/total)))
        x,w=np.polynomial.legendre.leggauss(size)
        width=np.log(hi/lo)/2.
        nodes.append(np.exp(np.log(lo)+width*(x+1)))
        weights.append(width*w)
    return np.concatenate(nodes),np.concatenate(weights)


def _vectors(k,mu,p,x,phi):
    # Shapes: external mu, radial p, x, phi, Cartesian coordinate.
    mu=np.asarray(mu)
    khat=np.stack([np.sqrt(1-mu*mu),np.zeros_like(mu),mu],axis=-1)
    perp=np.stack([mu,np.zeros_like(mu),-np.sqrt(1-mu*mu)],axis=-1)
    direction=x[None,:,:,None,None]*khat[:,None,None,None,:]
    direction=direction+np.sqrt(1-x*x)[None,:,:,None,None]*(np.cos(phi)[None,None,None,:,None]*perp[:,None,None,None,:]
        +np.sin(phi)[None,None,None,:,None]*np.array([0.,1.,0.]))
    pv=p[None,:,None,None,None]*direction
    kv=np.broadcast_to(k*khat[:,None,None,None,:],pv.shape)
    return kv,pv


def loop_integrals(k,mu,power,f,convention=EFTLoopConvention(),quadrature=Quadrature(),rsd=True,biased=False):
    """Return separate 22, 13 and combined matter loops, each of shape (nk, nmu).

    convention=None is an explicitly unregulated SPT reference.
    With ``biased`` (wstfast addition), the loops of Eulerian-biased tracers (``wstfast.theory.bias``), as the
    coefficients of the degree-2 bias monomials: shape (nk, nmu, nmono).
    """
    k,mu=np.asarray(k,dtype=float),np.asarray(mu,dtype=float)
    if k.ndim!=1 or mu.ndim!=1 or np.any(k<=0) or np.any(np.abs(mu)>1) or not np.isfinite(k).all() or not np.isfinite(mu).all():
        raise ValueError('finite positive k and physical mu arrays required')
    f=float(f) if rsd else 0.
    if not np.isfinite(f): raise ValueError('growth must be finite')
    if convention is not None and not isinstance(convention,EFTLoopConvention):
        raise ValueError('unknown loop convention')
    if convention is not None and np.max(k)>convention.cutoff:
        raise ValueError('reference output must satisfy k <= cutoff')
    shape=(len(k),len(mu))
    if biased:
        from .bias import galaxy_kernel, monomial_exponents, product_coefficients
        shape=shape+(len(monomial_exponents(2)),)
    results={name:np.zeros(shape) for name in ('22','13','loop')}
    gx,gw=np.polynomial.legendre.leggauss(quadrature.nx)
    phi=2*np.pi*np.arange(quadrature.nphi)/quadrature.nphi
    def fields(vectors):
        return galaxy_kernel(vectors,f) if biased else matter_kernel(vectors,f,rsd=rsd)
    def square(a,b):
        return product_coefficients(a,b) if biased else a*b
    def expand(w):
        return w[...,None] if biased else w
    for ik,ki in enumerate(k):
        p,rw=radial_rule(quadrature.qmin,quadrature.qmax,quadrature.nq,breaks=(ki/2,ki,2*ki))
        for start in range(0,len(p),quadrature.chunk):
            pp=p[start:start+quadrature.chunk]
            rrw=rw[start:start+quadrature.chunk]
            boundary=np.minimum(1.,ki/(2*pp))
            for lower,upper in ((np.zeros_like(pp),boundary),(boundary,np.ones_like(pp))):
                x=lower[:,None]+.5*(upper-lower)[:,None]*(gx+1)
                wx=.5*(upper-lower)[:,None]*gw
                if not np.any(wx):continue
                kv,pv=_vectors(ki,mu,pp,x,phi)
                Pp=power(pp)[None,:,None,None]
                gamma=fields(np.stack([kv,pv,-pv],axis=-2))
                lin=fields(kv[...,None,:])
                i13=6*expand(power(ki)*Pp)*square(lin,gamma)
                if convention is not None:
                    i13=i13*expand(convention.covariance(pp)[None,:,None,None])
                i22=np.zeros_like(i13)
                for sign in (1.,-1.):
                    leg=sign*pv; remainder=kv-leg
                    pr=np.linalg.norm(remainder,axis=-1)
                    K2=fields(np.stack([leg,remainder],axis=-2))
                    # Fold both soft corners onto p=0, then pair p and -p.
                    weight=2*Pp*power(pr)*(pr>pp[None,:,None,None])
                    if convention is not None:
                        weight*=convention.weights(ki,pp[None,:,None,None],pr)[0]
                    i22+=square(K2,K2)*expand(weight)
                weight=rrw[:,None]*pp[:,None]**3*wx/(2*np.pi**2*quadrature.nphi)
                # x now covers [0,1] and the integrand is the parity average;
                # angular normalization is dphi/(2pi) dx, with no extra 1/2.
                weight=expand(weight[None,:,:,None])
                for name,value in (('22',i22),('13',i13),('loop',i22+i13)):
                    results[name][ik]+=np.sum(weight*value,axis=(1,2,3))
    return results


def sigma2(power,convention=EFTLoopConvention(),quadrature=Quadrature()):
    p,w=radial_rule(quadrature.qmin,quadrature.qmax,quadrature.nq)
    return float(np.dot(w,p**3*power(p)*convention.covariance(p))/(2*np.pi**2))


def real_space_p13(k,power,convention=EFTLoopConvention(),quadrature=Quadrature()):
    """Independent analytic angular integral of the EdS F3 kernel."""
    out=[]
    for ki in np.atleast_1d(k):
        p,w=radial_rule(quadrature.qmin,quadrature.qmax,quadrature.nq,breaks=(ki,))
        r=p/ki
        # Evaluate the textbook expression in extended precision to control
        # polynomial cancellations; use its asymptotic series in both tails.
        t=r.astype(np.longdouble)
        with np.errstate(divide='ignore',invalid='ignore'):
            bracket=12/t**2-158+100*t**2-42*t**4+3/t**3*(t*t-1)**3*(7*t*t+2)*np.log(np.abs((1+t)/(1-t)))
        bracket=np.where(np.abs(t-1)<1.e-12,-88.,bracket)
        large=-np.longdouble(488)/5+np.longdouble(96)/(5*t**2)-np.longdouble(160)/(21*t**4)-np.longdouble(1376)/(1155*t**6)
        small=-168+np.longdouble(928)/5*t**2-np.longdouble(4512)/35*t**4+np.longdouble(416)/21*t**6
        bracket=np.where(t>30,large,np.where(t<.02,small,bracket)).astype(float)
        C=1. if convention is None else convention.covariance(p)
        out.append(ki**2*power(ki)/(252*(2*np.pi)**2)*np.dot(w,p*power(p)*C*bracket))
    return np.array(out)
