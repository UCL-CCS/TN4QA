# TN4QA

TN4QA (Tensor Networks for Quantum Algorithms) is a package designed to build workflows that use tensor network methods to assist quantum algorithms.

## Installation

Install the dependencies using
```
pip install poetry
poetry install
```

## Getting Started

From the top level of the repository you should be able to build the docs using the following

```
cd ./docs
make html
python3 -m http.server -d build/html 8080
```

so that the docs are accessible through http://localhost:8080. The documentation contains class information as well as tutorials on how to use TN4QA.
