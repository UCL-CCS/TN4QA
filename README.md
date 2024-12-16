# Development

Install the dev dependencies using

```
pip install poetry
poetry install --with=dev
```
and activate the pre-commit hooks
```
pre-commit install
```

**Note:** It might be necessary to set the environment variable `LLVM_CONFIG` to point to the correct binary.
You can find this using `which llvm-config-<<version>`.
