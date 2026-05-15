# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.2] - 2025-06-09

### Added
- Sampling from MPS
- Reordering sites in MPS/MPO
- Fidelity metrics for MPS/MPO

### Changed
- Python version changed from `>=3.11, <3.12` to `>=3.11, <4`
- Implemented faster fermionic Hamiltonian MPO builder

### Fixed
- Bug in existing fermionic Hamiltonian MPO construction

## [0.0.3] - 2025-07-11

### Added
- Quantum Algorithms sub-module
- Quantum backend offload
- Quantum circuit simulator
- Basic TN methods

### Changed
- DMRG now accepts MPO input

### Fixed
- Normalisation fixes in fidelity metrics
- Fixed tensor network constructions from qiskit QuantumCircuits

## [0.0.4] - 2025-07-16

### Added
- QSCI, TE-QSCI, CTE-QSCI quantum algorithms
- HF Suppression and Active Space Selection TN methods
- QI cost functions

### Fixed
- Various bug fixes and optimisations

## [0.0.5] - 2026-03-12

### Added
- Entanglement feature construction
- MPS to circuit mapping methods

### Changed
- Improved contraction/compression methods for MPO/MPS
- ActiveSpaceSelection renamed to MolecularOrbitalOptimisation

### Fixed
- Rewrote circuit to MPO/MPS constructions, no change to functionality

## [0.0.6] - 2026-03-14

### Fixed
- Necessary bug fix to mps to circuit middle out method

## [0.0.7] - 2026-03-19

### Changed
- Python version relaxed and dependencies updated
- Replaced poetry with uv

## [0.0.8] - 2026-03-19

### Changed
- Removed unused Symmer dependency

## [0.0.9] - 2026-03-26

### Changed
- Python version restricted to < 3.12 due to dependency issues

### Fixed
- Construction of cost function for MO optimisation method

### Added
- Log depth MPS to circuit mapping
- Verifier Circuit class

## [0.0.10] - 2026-04-10

### Changed
- Default DMRG solver changed to Block2

### Fixed
- Bug in exp_pauli_to_circuit builder

### Added
- Parent Hamiltonian class
- Noise modelling functionality
- Error mitigation functionality
- TNQEM method

## [0.0.11] - 2026-04-24

### Changed
- Modified correct handling of sparse vs. dense tensors throughout

### Removed
- TN4QA_DMRG and fermionic DMRG handling removed, replaced by DMRG class which wraps block2

## [0.0.12] - 2026-04-24

### Changed
- Minor necessary bug fix in MOO class

## [0.0.13] - 2026-04-24

### Changed
- Minor necessary bug fix in MOO class


## [0.0.14] - 2026-04-24

### Changed
- Minor necessary bug fix in MOO class

## [0.0.15] - 2026-04-24

### Changed
- Minor necessary bug fix in MOO class

## [0.0.16] - 2026-04-25

### Changed
- Minor necessary bug fix in MOO class

## [0.0.17] - 2026-04-25

### Changed
- Minor necessary bug fix in MOO class

## [0.0.18] - 2026-04-30

### Changed
- Update grad descent loop in MO optimisation
- Enable arbitrary initial state in DMRG

## [0.0.19] - 2026-05-14

### Changed
- Initialise DMRG with HF state in TNMOO
