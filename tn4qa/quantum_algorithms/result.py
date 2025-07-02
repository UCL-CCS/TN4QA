import json


class Result:
    """A class to store results of quantum algorithms"""

    def __init__(
        self,
        result: float | None = None,
        measurements: dict[str, int] | None = None,
        parameters: list[float] | None = None,
        metadata: dict | None = None,
        **kwargs,
    ):
        """Constructor

        Args:
            measurements: The measurement results {bitstring : count}
            result: The result of the algorithm (final energy, phase estimate, etc.)
            parameters: For variational circuits, the optimised parameters
            metadata: Timings, backend properties, etc.
        """
        self.measurements = measurements
        self.result = result
        self.parameters = parameters
        self.metadata = metadata
        self.info = kwargs

    def __str__(self):
        """Print statement"""
        output = "Quantum Alrogithm Result\n"
        if self.metadata:
            if "algorithm_name" in self.metadata:
                output += f"Algorithm: {self.metadata['algorithm_name']}\n"
            if "total_runtime" in self.metadata:
                output += f"Total runtime: {self.metadata['total_runtime']}\n"
        if self.result:
            output += f"Final result = {self.result}"
        return output

    def to_dict(self) -> dict:
        """Convert Result object to a dictionary"""
        return {
            "measurements": self.measurements,
            "result": self.result,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "info": self.info,
        }

    def save(self, filename: str) -> None:
        """Save Result object as a json file

        Args:
            filename: The filename for the result file
        """
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
