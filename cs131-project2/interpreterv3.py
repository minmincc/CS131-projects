from brewparse import parse_program
from intbase import InterpreterBase
from intbase import ErrorType
import copy
class ReturnException(Exception):
    def __init__(self, value=None):
        self.value = "nil" if value is None else value
class Interpreter(InterpreterBase):
    def __init__(self, console_output=True, inp=None, trace_output=False):
        super().__init__(console_output, inp)
        self.trace_output = trace_output
        self.variables = {}
        self.functions = {} 
        self.variables_stack = [{}]

    def run(self, program):
        ast = parse_program(program)
        if ast.elem_type == 'program':
            # Store all functions with their parameter count
            for func in ast.dict['functions']:
                func_name = func.dict['name']
                param_count = len(func.dict.get('args', []))
                self.functions[(func_name, param_count)] = func

            # Check for main function and run
            main_function = self.functions.get(("main", 0), None)
            if main_function:
                self.run_function_node(main_function)
            else:
                super().error(ErrorType.NAME_ERROR, "No main() function was found")
                
    def run_function_node(self, func_node):
        self.variables_stack.append({})
        try:
            if func_node.elem_type == 'func':
                for stmt in func_node.dict['statements']:
                    self.run_statement(stmt)
        except ReturnException as e:
            return e.value
        finally:
        # Remove the function's scope after execution
            self.variables_stack.pop()
        return "nil" 

    def run_statement(self, stmt_node):
        if stmt_node.elem_type == '=':
            self.handle_assignment(stmt_node.dict['name'], stmt_node.dict['expression'])
        elif stmt_node.elem_type == 'fcall':
            self.handle_function_call(stmt_node)
        elif stmt_node.elem_type == 'if':
            self.handle_if_statement(stmt_node.dict['condition'], stmt_node.dict['statements'], stmt_node.dict.get('else_statements'))
        elif stmt_node.elem_type == 'while':
            self.handle_while_statement(stmt_node.dict['condition'], stmt_node.dict['statements'])
        elif stmt_node.elem_type == 'return':
            self.handle_return_statement(stmt_node.dict.get('expression'))

    def handle_assignment(self, var_name, expression_node):
        value = self.evaluate_expression(expression_node)
        # Dynamic scoping: Assign to the most immediate scope that has the variable
        scope = self.find_variable_scope(var_name)
        if scope is not None:
            scope[var_name] = value
        else:
            # If not found in any scope, assign it to the current (top) scope
            self.current_scope()[var_name] = value

        if self.trace_output:
            print(f"{var_name} = {value}")

    def current_scope(self):
        # Returns the current scope (top of the stack)
        return self.variables_stack[-1]
    
    def find_variable_scope(self, var_name):
        # Search for the variable's scope in the stack from top (most recent scope) to bottom (global scope)
        for scope in reversed(self.variables_stack):
            if var_name in scope:
                return scope
        return None
    
    def handle_if_statement(self, condition_node, true_statements, else_statements=None):
        condition_value = self.evaluate_expression(condition_node)

        if type(condition_value) != bool:
            self.error(ErrorType.TYPE_ERROR, "Condition in 'if' statement must evaluate to a boolean value.")
            return
        self.variables_stack.append({})
        try:
            if condition_value:
                for stmt in true_statements:
                    self.run_statement(stmt)
            elif else_statements:  # This condition ensures that even if else_statements is an empty list, it won't be executed
                for stmt in else_statements:
                    self.run_statement(stmt)
        finally:
            self.variables_stack.pop()
        
    def handle_while_statement(self, condition_node, statements):
        condition_value = self.evaluate_expression(condition_node)
        
        if type(condition_value) != bool:
            self.error(ErrorType.TYPE_ERROR, "Condition in 'while' statement must evaluate to a boolean value.")
            return
        
        while self.evaluate_expression(condition_node):
            self.variables_stack.append({})  # Start a new scope
            try:
                for statement in statements:
                    self.run_statement(statement)
            finally:
                self.variables_stack.pop()
                
    def handle_return_statement(self, expression_node):
        if expression_node is None:
            raise ReturnException(None)  # Return nil if there is no expression
        else:
            value = self.evaluate_expression(expression_node)
            value_copy = copy.deepcopy(value)
            raise ReturnException(value_copy)

    def evaluate_expression(self, expr_node):
        if expr_node.elem_type in ['+', '-', '*', '/']:
            return self.expression_with_arithmetic(expr_node)
        elif expr_node.elem_type == 'fcall':
            return self.handle_function_call(expr_node)
        elif expr_node.elem_type == 'var':
            return self.get_variable_value(expr_node.dict['name'])
        elif expr_node.elem_type in ['int']:
            return expr_node.dict['val']
        elif expr_node.elem_type == 'string':
            return expr_node.dict['val']
        elif expr_node.elem_type in ['&&', '||']:
            return self.expression_with_logical(expr_node)
        elif expr_node.elem_type in ['!','neg']:
            return self.expression_with_negation(expr_node)
        elif expr_node.elem_type in ['==', '!=', '<', '<=', '>', '>=']:
            return self.expression_with_comparison(expr_node)
        elif expr_node.elem_type == 'bool':
            return expr_node.dict['val']
        elif expr_node.elem_type == 'nil':
            return None
    def expression_with_logical(self, expression_node):
        # Evaluate the first operand
        op1_val = self.evaluate_expression(expression_node.dict['op1'])
        op2_val = self.evaluate_expression(expression_node.dict['op2'])
        
        # Check if both operands are booleans
        if not isinstance(op1_val, bool) or not isinstance(op2_val, bool):
            super().error(ErrorType.TYPE_ERROR, "Both operands must be of type bool for logical operations.")
            return None

        # Perform the logical operations with strict evaluation
        if expression_node.elem_type == '&&':
            return op1_val and op2_val
        elif expression_node.elem_type == '||':
            return op1_val or op2_val
        else:
            super().error(ErrorType.SYNTAX_ERROR, f"Unknown logical operator {expression_node.elem_type}.")
            return None
        
    def expression_with_negation(self, expression_node):
        op1_val = self.evaluate_expression(expression_node.dict['op1'])
        
        # Check for integer negation
        if expression_node.elem_type == 'neg':
            if not isinstance(op1_val, int):
                super().error(ErrorType.TYPE_ERROR, "Operand must be an integer.")
                return None
            return -op1_val

        # Check for boolean negation
        elif expression_node.elem_type == '!':
            if not isinstance(op1_val, bool):
                super().error(ErrorType.TYPE_ERROR, "Operand must be of type bool.")
                return None
            return not op1_val
        else:
            super().error(ErrorType.TYPE_ERROR, f"Unsupported negation operation: {expression_node.elem_type}.")
            return None
        
    def expression_with_comparison(self, expression_node):
        op1_val = self.evaluate_expression(expression_node.dict['op1'])
        op2_val = self.evaluate_expression(expression_node.dict['op2'])
        if(op1_val == None or op1_val =='nil'):
            op1_val = None
        if(op2_val == None or op1_val == 'nil'):
            op2_val = None
        # == and != checks can compare different types
        # 'None' is treated as 'nil' and can be compared with any type
        if op1_val is None and op2_val is None:
            return True if expression_node.elem_type == '==' else False
        if (op1_val is None) != (op2_val is None):
            return False if expression_node.elem_type == '==' else True
        
        if expression_node.elem_type == '==':
            return op1_val == op2_val if type(op1_val) == type(op2_val) else False
        elif expression_node.elem_type == '!=':
            return op1_val != op2_val if type(op1_val) == type(op2_val) else True

        # For other comparison operators, ensure neither value is None
        if op1_val is None or op2_val is None:
            super().error(ErrorType.TYPE_ERROR, "Cannot compare 'nil' with other types using this operator.")
            return None

        # Ensure the types of both operands are the same before comparing
        if type(op1_val) != type(op2_val):
            super().error(ErrorType.TYPE_ERROR, "Cannot compare values of different types with this operator.")
            return None

        # Standard comparisons
        if not isinstance(op1_val, int) or not isinstance(op2_val, int):
            super().error(ErrorType.TYPE_ERROR, "Both operands must be integers for this type of comparison.")
            return None
        
        if expression_node.elem_type == '<':
            return op1_val < op2_val
        elif expression_node.elem_type == '<=':
            return op1_val <= op2_val
        elif expression_node.elem_type == '>':
            return op1_val > op2_val
        elif expression_node.elem_type == '>=':
            return op1_val >= op2_val

        # If the comparison operator is not recognized, you can either return None or raise an error
        super().error(ErrorType.SYNTAX_ERROR, "Unknown comparison operator.")
        return None
        
    def expression_with_arithmetic(self, expression_node):
        # Evaluating op1
        def evaluate_operand(operand_node):
            return self.evaluate_expression(operand_node)  # This change allows for handling of nested expressions

        # Evaluate op1 and op2
        op1_val = evaluate_operand(expression_node.dict['op1'])
        op2_val = evaluate_operand(expression_node.dict['op2'])

        if isinstance(op1_val, bool) or isinstance(op2_val, bool) or op1_val is None or op2_val is None:
            super().error(ErrorType.TYPE_ERROR, "Arithmetic operations are not allowed with bool or nil")
            return None
        if type(op1_val) != type(op2_val):
            super().error(ErrorType.TYPE_ERROR, f"Incompatible types: {type(op1_val).__name__} and {type(op2_val).__name__}")
            return None 
        if (expression_node.dict['op1'].elem_type == 'fcall' and expression_node.dict['op1'].dict['name'] == 'inputi') and \
        (expression_node.dict['op2'].elem_type == 'fcall' and expression_node.dict['op2'].dict['name'] == 'inputi'):
            super().error(ErrorType.TYPE_ERROR, "Cannot have two inputi function calls in a single arithmetic expression.")
            return None

        # Perform arithmetic operation
        if expression_node.elem_type == '+':
            if isinstance(op1_val, str) or isinstance(op2_val, str):  # Allow string concatenation
                return str(op1_val) + str(op2_val)
            return op1_val + op2_val
        elif expression_node.elem_type == '-':
            if isinstance(op1_val, str) or isinstance(op2_val, str):
                super().error(ErrorType.TYPE_ERROR, "Cannot subtract strings.")
                return None
            return op1_val - op2_val
        elif expression_node.elem_type == '*':
            if isinstance(op1_val, str) or isinstance(op2_val, str):
                super().error(ErrorType.TYPE_ERROR, "Cannot multiply strings.")
                return None
            return op1_val * op2_val
        elif expression_node.elem_type == '/':
            if isinstance(op1_val, str) or isinstance(op2_val, str):
                super().error(ErrorType.TYPE_ERROR, "Cannot divide strings.")
                return None
            if op2_val == 0:
                super().error(ErrorType.VALUE_ERROR, "Division by zero.")
                return None
            return op1_val // op2_val
        
    def handle_function_call(self, func_node):
        func_name = func_node.dict['name']
        args = [self.evaluate_expression(arg) for arg in func_node.dict['args']]
        args_count = len(args)

        # Check for function overload resolution
        function_key = (func_name, args_count)
        
        # Handling built-in functions
        if function_key == ('print', len(args)):
            formatted_args = [(str(arg).lower() if isinstance(arg, bool) else str(arg)) for arg in args]
            self.output(''.join(formatted_args))
            return None
            
        elif function_key == ('inputi', len(args)):
            if len(args) > 1:
                super().error(ErrorType.NAME_ERROR, f"No inputi() function found that takes > 1 parameter")
                return None
            if len(args) == 1:
                prompt = args[0]
                super().output(prompt)

            user_input = super().get_input()
            return int(user_input)
        
        elif function_key == ('inputs', len(args)):
            if len(args) > 1:
                super().error(ErrorType.NAME_ERROR, "inputs() takes at most 1 parameter")
                return None
            if len(args) == 1:
                # Output the prompt before getting input.
                prompt = args[0]
                super().output(prompt)
                
            # Now, call get_input without any arguments as per its definition.
            user_input = super().get_input()
            return str(user_input)
            
        # Handling user-defined functions
        elif function_key in self.functions:
            # Retrieve the function node from our functions dictionary
            target_function = self.functions[function_key]
            
            
            # Check if number of arguments matches
            if len(args) != len(target_function.dict.get('args', [])):
                super().error(ErrorType.NAME_ERROR, f"Function {func_name} called with incorrect number of arguments")
                return None
            
            # Create a new scope for the function call
            new_scope = {}
            
            # Push the new scope to the stack
            self.variables_stack.append(new_scope)

            # Now, set the arguments in the new scope after it's been added to the stack
            for param_node, value in zip(target_function.dict['args'], args):
                param_name = param_node.dict['name']
                new_scope[param_name] = value
            
            

            # Execute the function
            try:
                return_val = self.run_function_node(target_function)
            finally:
                # Always remove the function's scope from the stack, even if an exception occurred
                self.variables_stack.pop()
                  # Assume you'd also want to pop the args after the call
            
            # Return the result
            return return_val if return_val is not None else "nil"
            
        else:
            super().error(ErrorType.NAME_ERROR, f"Unknown function {func_name}")
            return None
        
    def get_variable_value(self, var_name):
        # Dynamic scoping: Find the most immediate scope where the variable is defined
        scope = self.find_variable_scope(var_name)
        if scope is not None:
            return scope[var_name]
        else:
            super().error(ErrorType.NAME_ERROR, f"Variable {var_name} has not been defined")
            return None