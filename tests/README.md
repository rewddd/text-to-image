# Adding Tests for a Model

1. Create a file named `{model_name}.yaml` in the `artifacts` directory.
This file will store all your test cases for evaluating your model.

2. Each test case should consist of three key elements:
   - `name`: A descriptive name for the test case.
   - `input`: The input data to be provided to your model.
   - `output`: The expected output for the given input. If your model returns
        a `File` or `Image` output, it will automatically be converted to a
        sha256 encoded value. Therefore, provide the sha256 value of the
        expected output if your output is a `File` or `Image`.

3. Create a Python file named `test_{model_name}.py` for your model's tests.

4. In the `test_{model_name}.py` file, create a test class with a name starting
with `Test{model_name}`. This class should inherit from `BaseModelTest`, which
is defined in `helpers.py`.

5. Because you inherit from the `BaseModelTest` class, you need to implement
the `generate` method. In this method, parse the inputs to create a model
input, retrieve the generation result, and then return the model output as it
is.

To test your model against your inputs you can use:
```bash
pytest -k {model_name}
```

To test all models:
```bash
pytest
```
