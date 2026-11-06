import scipy as sp

mmax = 5
nmax = 3

print(f"Neumann roots j'_(m,n) for m=0..{mmax}, n=1..{nmax}:")
for n in range(1, nmax + 1):
    for m in range(0, mmax + 1):
        nm = 10 * n + m
        roots = sp.special.jnp_zeros(m, n)
        print(f"nm={nm}: "+f'{roots[-1]:.16f}')

print(f"\nDirichlet roots j_(m,n) for m=0..{mmax}, n=1..{nmax} (tabulated):")
for n in range(1, nmax + 1):
    for m in range(0, mmax + 1):
        nm = 10 * n + m
        roots = sp.special.jn_zeros(m, n)
        print(f"nm={nm}: "+f'{roots[-1]:.16f}')