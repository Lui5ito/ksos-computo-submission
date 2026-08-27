# Joint learning of RKHS and kernel sum-of-squares functions: a representer theorem and its convex dual

Louis Allain, Sébastien Da Veiga
2026-08-27

### Citation

Submitted to Computo.

### Badges

[![build and
publish](https://github.com/Lui5ito/ksos-computo-submission/actions/workflows/build.yml/badge.svg)](https://github.com/Lui5ito/ksos-computo-submission/actions/workflows/build.yml)
[![reviews](https://img.shields.io/badge/review-report-blue)](https://github.com/Lui5ito/ksos-computo-submission/issues?q=is%3Aopen+is%3Aissue+label%3Areview)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/Lui5ito/ksos-computo-submission)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/Lui5ito/ksos-computo-submission)
[![DOI:10.5072/computo.0000](https://img.shields.io/badge/DOI-10.5072%2Fcomputo.0000-034E79.svg)](https://doi.org/10.5072/computo.0000)
[![Creative Commons
License](https://i.creativecommons.org/l/by/4.0/80x15.png)](http://creativecommons.org/licenses/by/4.0/)

### Authors’ affiliations

- Louis Allain (Safran AI Research, Magny-Les-Hameaux, France, Univ Rennes, Ensai, CNRS, CREST - UMR 9194, F-35000 Rennes, France)
- [Sébastien Da Veiga](https://sites.google.com/view/sebastien-da-veiga/home) (Univ Rennes, Ensai, CNRS, CREST - UMR 9194, F-35000 Rennes, France)

### Abstract

Kernel sum-of-squares (kSoS) models represent a non-negative function as
a quadratic form in an reproducing kernel Hilbert space (RKHS),
guaranteeing non-negativity everywhere. It has emerged as a promising
kernel approach to model non-negative phenomena, leveraging a
representer theorem that makes the problem tractable. This result only
holds for kSoS functions, and with the same RKHS. However, real-world
problems may require to learn general-valued functions *jointly* with
kSoS functions, and may also benefit from different kernels for each
kSoS function for better predictive accuracy. In this work, we thus
consider such wider class of statistical learning problems, where we are
interested in jointly learning $p$ real valued functions and $q$
non-negative functions defined on different RKHS. Building on
representer theorems from traditional kernel methods and kSoS, we
introduce a new generalized representer theorem for that learning
problem. The induced finite-dimensional problem is a semi-definite
program (SDP), easily solvable using off-the-shelf solvers. To scale
better than SDP solvers, we also establish a convex dual formulation,
which writes as an optimization problem with only $\mathcal{O}(n)$
variables and no positive semi-definite constraints. Finally, for
completeness and to highlight the potential of the kSoS framework, we
derive the explicit dual formulations for three kSoS problems previously
introduced in the literature and provide two new problems leveraging the
generalized representer theorem. All our experiments are reproducible
with our accompanying Python code.
