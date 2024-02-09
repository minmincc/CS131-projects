
from brewparse import parse_program
from intbase import InterpreterBase
from intbase import ErrorType
class Interpreter(InterpreterBase):
    def __init__(self, console_output=True, inp=None, trace_output=False):
        super().__init__(console_output, inp)
        self.trace_output = trace_output
        self.variables = {}

    def run(self, program):
        ast = parse_program(program)
        if ast.elem_type == 'program':
            # Check if there's a main function and run it
            main_function = None
            for func in ast.dict['functions']:
                if func.dict['name'] == "main":
                    main_function = func
                    break
            
            if main_function:
                self.run_function_node(main_function)
            else:
                super().error(ErrorType.NAME_ERROR, "No main() function was found")
                
    def run_function_node(self, func_node):
        if func_node.elem_type == 'func':
            for stmt in func_node.dict['statements']:
                self.run_statement(stmt)

    def run_statement(self, stmt_node):
        if stmt_node.elem_type == '=':
            self.handle_assignment(stmt_node.dict['name'], stmt_node.dict['expression'])
        elif stmt_node.elem_type == 'fcall':
            self.handle_function_call(stmt_node)

    def handle_assignment(self, var_name, expression_node):
        value = self.evaluate_expression(expression_node)
        self.variables[var_name] = value
        if self.trace_output:
            print(f"{var_name} = {value}")

    def evaluate_expression(self, expr_node):
        if expr_node.elem_type in ['+', '-']:
            return self.expression_with_arithmetic(expr_node)
        elif expr_node.elem_type == 'fcall':
            return self.handle_function_call(expr_node)
        elif expr_node.elem_type == 'var':
            return self.get_variable_value(expr_node.dict['name'])
        elif expr_node.elem_type == 'int':
            return expr_node.dict['val']
        elif expr_node.elem_type == 'string':
            return expr_node.dict['val']

    def expression_with_arithmetic(self, expression_node):
        # Evaluating op1
        def evaluate_operand(operand_node):
            return self.evaluate_expression(operand_node)  # This change allows for handling of nested expressions

        # Evaluate op1 and op2
        op1_val = evaluate_operand(expression_node.dict['op1'])
        op2_val = evaluate_operand(expression_node.dict['op2'])

        if type(op1_val) != type(op2_val):
            super().error(ErrorType.TYPE_ERROR, f"Incompatible types: {type(op1_val).__name__} and {type(op2_val).__name__}")
            return None 
        if (expression_node.dict['op1'].elem_type == 'fcall' and expression_node.dict['op1'].dict['name'] == 'inputi') and \
        (expression_node.dict['op2'].elem_type == 'fcall' and expression_node.dict['op2'].dict['name'] == 'inputi'):
            super().error(ErrorType.TYPE_ERROR, "Cannot have two inputi function calls in a single arithmetic expression.")
            return None

        # Perform arithmetic operation
        if expression_node.elem_type == '+':
            return op1_val + op2_val
        elif expression_node.elem_type == '-':
            if isinstance(op1_val, str) or isinstance(op2_val, str):
                super().error(ErrorType.TYPE_ERROR, "Cannot subtract strings.")
                return None
            return op1_val - op2_val
        
    def handle_function_call(self, func_node):
        func_name = func_node.dict['name']
        args = [self.evaluate_expression(arg) for arg in func_node.dict['args']]
        if func_name == 'print':
            # Joining the arguments without spaces and outputting them
            self.output(''.join(map(str, args)))
            
        elif func_name == 'inputi':
            if len(args) > 1:
                super().error(
                    ErrorType.NAME_ERROR,
                    f"No inputi() function found that takes > 1 parameter"
                )
                return None
            if len(args) == 1:
                prompt = args[0]
                # Display the prompt to the user using the base class's output method
                super().output(prompt)

            # Get the user's input
            user_input = super().get_input()
            return int(user_input)
        else:  # If the function name is neither 'print' nor 'inputi'
            super().error(ErrorType.NAME_ERROR, f"Unknown function {func_name}")
            return None
        
    def get_variable_value(self, var_name):
        if var_name not in self.variables:
            super().error(ErrorType.NAME_ERROR, f"Variable {var_name} has not been defined")
            return None  # This is just an example. Depending on the flow of your program, you might need to handle this scenario differently.
        return self.variables[var_name]