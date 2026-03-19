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
