# TO-DOs

## MUST-DOs`

- [ ] `SLSQP` algorithm stops with local optimal found
- [ ] Try `pyomo==4.3.11388` for better computational performance (available in debian packages)
- [ ] (Optional) Use an older version of `numpy` (Latest version for ARM debian is `1.13.3`)
- [x] Use an older version of `scipy` (Latest version for ARM debian is `0.19.1`)

## Code Simplification

- [ ] Use `:` instead of `ndarray.take()`
- [ ] Move `opf.makeYbus()` to `powerflow` module

## Performance Improvement

- [ ] Use sparse matrices when necessary
- [x] Added `dg`, `dh` for better convergence

## Issues

### Selection of `method` for `scipy.optimize.minimize()`

- [ ] `'trust-ncg'` cannot handle `bounds` nor `constraints`
- [ ] `'COBYLA'` nor `'TNC'` cannot handle equality `constraints`