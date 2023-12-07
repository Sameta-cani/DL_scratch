class MulLayer:
    """Multiplication Layer for Neural Networks.

    This class represents a multiplication layer used in neural networks.
    It has forward and backward methods to perform forward and backward passes.

    Attributes:
        input_x (float): The input value x during the forward pass.
        input_y (float): The input value y during the forward pass.
    """
    def __init__(self):
        self.input_x = None
        self.input_y = None

    def forward(self, x, y):
        self.input_x = x
        self.input_y = y
        out = x * y
        return out
    
    def backward(self, dout):
        dx = dout * self.input_y
        dy = dout * self.input_x

        return dx, dy
    
class AddLayer:
    """Addition Layer for Neural Networks.

    This class represents an addition layer used in neural networks.
    It has forward and backward methods to perform forward and backward passes.

    Note:
        The initialization method does not have any specific functionality.
    """
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy