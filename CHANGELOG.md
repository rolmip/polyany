## 0.3.0 (2025-08-13)

### Feat

- multiplication and division (#40)

## 0.2.0 (2025-08-06)

### Feat

- partial derivatives (#31)
- zeros polynomial (#29)
- add method to prune empty monomials (#27)
- addition/subtraction (#19)
- **exponents**: sparse representation (#20)
- add method to shift variables (#8)
- add comparison and equality operators (#5)
- add quadratic form creation method (#4)
- evaluate polynomial at a point using __call__
- add univariate polynomial method

### Fix

- **prune**: return zeros polynomial when all coefficients are zero (#30)
- polynomial evaluation without constant terms (#24)
- get python builtin in degree attribute
- remove squeeze in coefficients conversion

### Refactor

- simplify internal representation to full polynomial

### Perf

- full_exponents algorithm (#15)
