from ..mps import MatrixProductState
from ..tn import TensorNetwork


class MPSOptimiser(TensorNetwork):
    """
    A class for locally optimising tensors in a TN with respect to a reference MPS and the HS distance
    """

    def __init__(self, tn: TensorNetwork, reference: MatrixProductState) -> None:
        """
        Constructor

        Args:
            tn: The tensor network that will be optimised. Should be contractable to an MPS
            reference: The reference MPS
        """
        if isinstance(reference, TensorNetwork):
            tensors = tn.tensors + reference.tensors
        else:
            tensors = tn.tensors
        super().__init__(tensors, name="TNOptimiser")
        self.tn = tn
        self.reference = reference
        label_to_tensor_dict = tn.get_label_to_tensor_dict()
        self.variational_tensors = label_to_tensor_dict.get("variational", [])
