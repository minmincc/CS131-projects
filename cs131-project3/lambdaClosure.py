import copy
class LambdaClosure:
    def __init__(self, lambda_ast, captured_env):
        self.lambda_ast = lambda_ast
        self.captured_env = captured_env

def __create_lambda(self, lambda_ast):
    # Capture the current environment's state
    captured_env = copy.deepcopy(self.env)
    # Create a new LambdaClosure object with the lambda_ast and the captured environment
    return LambdaClosure(lambda_ast, captured_env)