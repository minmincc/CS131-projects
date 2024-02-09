
import copy
from enum import Enum

from brewparse import parse_program
from env_v2 import EnvironmentManager
from intbase import InterpreterBase, ErrorType
from type_valuev2 import Type, Value, create_value, get_printable

class LambdaClosure:
    def __init__(self, lambda_ast, captured_env):
        self.lambda_ast = lambda_ast
        self.captured_env = captured_env


class ExecStatus(Enum):
    CONTINUE = 1
    RETURN = 2


# Main interpreter class
class Interpreter(InterpreterBase):
    # constants
    NIL_VALUE = create_value(InterpreterBase.NIL_DEF)
    TRUE_VALUE = create_value(InterpreterBase.TRUE_DEF)
    BIN_OPS = {"+", "-", "*", "/", "==", "!=", ">", ">=", "<", "<=", "||", "&&"}

    # methods
    def __init__(self, console_output=True, inp=None, trace_output=False):
        super().__init__(console_output, inp)
        self.trace_output = trace_output
        self.__setup_ops()

    # run a program that's provided in a string
    # usese the provided Parser found in brewparse.py to parse the program
    # into an abstract syntax tree (ast)
    def run(self, program):
        ast = parse_program(program)
        self.__set_up_function_table(ast)
        self.env = EnvironmentManager()
        main_func = self.__get_func_by_name("main", 0)
        self.__run_statements(main_func.get("statements"))
    
    def __set_up_function_table(self, ast):
        self.func_name_to_ast = {}
        for func_def in ast.get("functions"):
            func_name = func_def.get("name")
            num_params = len(func_def.get("args"))
            if func_name not in self.func_name_to_ast:
                self.func_name_to_ast[func_name] = {}
            self.func_name_to_ast[func_name][num_params] = func_def

    def __get_func_by_name(self, name, num_params):
        if name not in self.func_name_to_ast:
            super().error(ErrorType.NAME_ERROR, f"Function {name} not found")
        candidate_funcs = self.func_name_to_ast[name]
        if num_params not in candidate_funcs:
            super().error(
                ErrorType.NAME_ERROR,
                f"Function {name} taking {num_params} params not found",
            )
        return candidate_funcs[num_params]

    def __run_statements(self, statements):
        self.env.push()
        for statement in statements:
            if self.trace_output:
                print(statement)
            status = ExecStatus.CONTINUE
            if statement.elem_type == InterpreterBase.FCALL_DEF:
                self.__call_func(statement)
            elif statement.elem_type == "=":
                self.__assign(statement)
            elif statement.elem_type == InterpreterBase.RETURN_DEF:
                status, return_val = self.__do_return(statement)
            elif statement.elem_type == Interpreter.IF_DEF:
                status, return_val = self.__do_if(statement)
            elif statement.elem_type == Interpreter.WHILE_DEF:
                status, return_val = self.__do_while(statement)

            if status == ExecStatus.RETURN:
                self.env.pop()
                return (status, return_val)

        self.env.pop()
        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)
    
    def __call_lambda(self, lambda_closure, actual_args):
        saved_env = self.env

        # Set the environment to the lambda's captured environment
        self.env = lambda_closure.captured_env
        self.env.push()

        # Bind actual arguments to lambda's parameters
        formal_args = lambda_closure.lambda_ast.get("args")
        if len(actual_args) != len(formal_args):
            super().error(ErrorType.TYPE_ERROR, "Incorrect number of arguments for lambda function")

        # Store references for reference arguments
        ref_args = {}
        for formal_ast, actual_ast in zip(formal_args, actual_args):
            if formal_ast.elem_type == "refarg":
                # Handle reference arguments
                if actual_ast.elem_type != "var":
                    super().error(ErrorType.TYPE_ERROR, "Reference argument must be a variable for pass by reference")
                arg_name = formal_ast.get("name")
                actual_var_name = actual_ast.get("name")
                # Store the reference
                var_ref = saved_env.get_reference(actual_var_name)
                if var_ref:
                    ref_args[arg_name] = var_ref
                    self.env.create(arg_name, saved_env.get(actual_var_name))
                else:
                    super().error(ErrorType.NAME_ERROR, f"Variable {actual_var_name} not found for pass by reference")
            else:
                # Evaluate and bind non-reference arguments
                result = self.__eval_expr(actual_ast)
                arg_name = formal_ast.get("name")
                self.env.create(arg_name, result)

        # Execute the lambda's body
        status, return_val = self.__run_statements(lambda_closure.lambda_ast.get("statements"))

        # Update the original variables for reference arguments
        for ref_name, ref_actual in ref_args.items():
            if ref_actual["type"] == "ref":
                ref_actual["env"][ref_actual["key"]] = self.env.get(ref_name)

        # Restore the original environment
        self.env.pop()
        self.env = saved_env

        return return_val
    
    def __call_func(self, call_node):

        func_identifier = call_node.get("name")
        actual_args = call_node.get("args")

        # Check if the identifier is a function reference from a variable or a direct function name
        func_or_closure = self.env.get(func_identifier)
        if isinstance(func_or_closure, dict) and func_or_closure.get("type") == "function_ref":
            # Retrieve the actual function name
            func_name = func_or_closure.get("name")
        elif func_identifier in self.func_name_to_ast and len(self.func_name_to_ast[func_identifier]) > 1:
            # Error if the function name is overloaded
            super().error(ErrorType.NAME_ERROR, f"Ambiguous function reference to overloaded function '{func_identifier}'")
            return
        else:
            func_name = func_identifier

        # Handle built-in functions like 'print' and 'input'
        if func_name == "print":
            return self.__call_print(call_node)
        if func_name == "inputi":
            return self.__call_input(call_node)
        if func_name == "inputs":
            return self.__call_input(call_node)

        # Call the function or lambda
        if isinstance(func_or_closure, LambdaClosure):
            if len(actual_args) != len(func_or_closure.lambda_ast.get("args")):
                super().error(ErrorType.TYPE_ERROR, "Incorrect number of arguments for lambda function")
            return self.__call_lambda(func_or_closure, actual_args)

        # Retrieve the function definition and call it
        func_def = self.func_name_to_ast.get(func_name, {}).get(len(actual_args))
        if not func_def:
            super().error(ErrorType.TYPE_ERROR, f"Incorrect number of arguments for function {func_name}")
        ref_lambdas = {}  # Store original lambda closures for reference arguments


        self.env.push()
        ref_args = {}  # Store references for reference arguments

        for formal_ast, actual_ast in zip(func_def.get("args"), actual_args):
            if formal_ast.elem_type == "refarg":
                # Handle reference arguments
                if actual_ast.elem_type == "var":
                    # For variable references, store the variable reference
                    arg_name = actual_ast.get("name")
                    ref_args[formal_ast.get("name")] = self.env.get_reference(arg_name)
                    self.env.create(formal_ast.get("name"), self.env.get(arg_name))
                elif actual_ast.elem_type == "lambda":
                    # For lambda references, directly use the lambda closure
                    lambda_closure = self.__eval_expr(actual_ast)
                    ref_args[formal_ast.get("name")] = lambda_closure
                    self.env.create(formal_ast.get("name"), lambda_closure)
            else:
                # Handle non-reference arguments with a deep copy for lambdas
                evaluated_arg = self.__eval_expr(actual_ast)
                if isinstance(evaluated_arg, LambdaClosure):
                    evaluated_arg = LambdaClosure(evaluated_arg.lambda_ast, copy.deepcopy(evaluated_arg.captured_env))
                self.env.create(formal_ast.get("name"), evaluated_arg)

        # Execute the function body
        status, return_val = self.__run_statements(func_def.get("statements"))

        # Update the original variables or lambda closures for reference arguments
        for ref_name, ref_actual in ref_args.items():
            if isinstance(ref_actual, dict) and ref_actual.get("type") == "ref":
                # Update variable references
                ref_actual["env"][ref_actual["key"]] = self.env.get(ref_name)
            elif isinstance(ref_actual, LambdaClosure):
                # Update lambda closures
                modified_lambda = self.env.get(ref_name)
                if modified_lambda != ref_actual:
                    ref_actual.captured_env = modified_lambda.captured_env

        self.env.pop()
        return return_val
    def __call_print(self, call_ast):
        output = ""
        for arg in call_ast.get("args"):
            result = self.__eval_expr(arg)  # result is a Value object
            output = output + get_printable(result)
        super().output(output)
        return Interpreter.NIL_VALUE

    def __call_input(self, call_ast):
        args = call_ast.get("args")
        if args is not None and len(args) == 1:
            result = self.__eval_expr(args[0])
            super().output(get_printable(result))
        elif args is not None and len(args) > 1:
            super().error(
                ErrorType.NAME_ERROR, "No inputi() function that takes > 1 parameter"
            )
        inp = super().get_input()
        if call_ast.get("name") == "inputi":
            return Value(Type.INT, int(inp))
        if call_ast.get("name") == "inputs":
            return Value(Type.STRING, inp)

    def __assign(self, assign_ast):
        var_name = assign_ast.get("name")
        value = self.__eval_expr(assign_ast.get("expression"))

        if isinstance(value, dict) and value.get("type") == "function_ref":
            # Check if the function has overloaded versions
            func_name = value.get("name")
            if len(self.func_name_to_ast.get(func_name, {})) > 1:
                super().error(ErrorType.NAME_ERROR, f"Ambiguous function reference: {func_name} has overloaded versions")
            else:
                # Store the function reference as a special value
                self.env.set(var_name, value)
        elif isinstance(value, LambdaClosure):
            self.env.set(var_name, value)
        else:
            # Check if var_name is a reference, if so, update the referenced variable
            var_ref = self.env.get_reference(var_name)
            if var_ref and var_ref.get("type") == "ref":
                var_ref["env"][var_ref["key"]] = value
            else:
                self.env.set(var_name, value)

    def __create_lambda(self, lambda_ast):
        # Capture the current environment's state
        captured_env = copy.deepcopy(self.env)
        # Create a new LambdaClosure object with the lambda_ast and the captured environment
        return LambdaClosure(lambda_ast, captured_env)
    def __eval_expr(self, expr_ast):
        # print("here expr")
        # print("type: " + str(expr_ast.elem_type))
        
        if expr_ast.elem_type == InterpreterBase.NIL_DEF:
            # print("getting as nil")
            return Interpreter.NIL_VALUE
        if expr_ast.elem_type == InterpreterBase.INT_DEF:
            return Value(Type.INT, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.STRING_DEF:
            # print("getting as str")
            return Value(Type.STRING, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.BOOL_DEF:
            return Value(Type.BOOL, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.VAR_DEF:
            var_name = expr_ast.get("name")
            var_info = self.env.get(var_name)
            if var_name in self.func_name_to_ast:
                # Treat function name as a reference to the function
                return {"type": "function_ref", "name": var_name}
            else:
                val = self.env.get(var_name)
                if val is None:
                    super().error(ErrorType.NAME_ERROR, f"Variable {var_name} not found")
                return val
        if expr_ast.elem_type == InterpreterBase.FCALL_DEF:
            return self.__call_func(expr_ast)
        if expr_ast.elem_type == InterpreterBase.LAMBDA_DEF:
            return self.__create_lambda(expr_ast)
        if expr_ast.elem_type in Interpreter.BIN_OPS:
            return self.__eval_op(expr_ast)
        if expr_ast.elem_type == Interpreter.NEG_DEF:
            return self.__eval_unary(expr_ast, Type.INT, lambda x: -1 * x)
        if expr_ast.elem_type == Interpreter.NOT_DEF:
            return self.__eval_unary(expr_ast, Type.BOOL, lambda x: not x)

    def __eval_op(self, arith_ast):
        left_value_obj = self.__eval_expr(arith_ast.get("op1"))
        right_value_obj = self.__eval_expr(arith_ast.get("op2"))

        # Handling comparisons involving functions or lambda closures
        if isinstance(left_value_obj, (LambdaClosure, dict)) or isinstance(right_value_obj, (LambdaClosure, dict)):
            if arith_ast.elem_type not in ["==", "!="]:
                super().error(ErrorType.TYPE_ERROR, "Invalid operation on functions or lambdas")
            # Compare function or lambda references
            comparison_result = (left_value_obj == right_value_obj)
            if arith_ast.elem_type == "!=":
                comparison_result = not comparison_result
            return Value(Type.BOOL, comparison_result)

        # Check if one operand is int and the other is bool for conversion
        if (left_value_obj.type() == Type.INT and right_value_obj.type() == Type.BOOL) or (left_value_obj.type() == Type.BOOL and right_value_obj.type() == Type.INT ):
            # Convert int to bool for logical and comparison operations
            if arith_ast.elem_type in ["&&", "||", "==", "!="]:
                left_value_obj = self.__convert_to_bool_if_int(left_value_obj)
                right_value_obj = self.__convert_to_bool_if_int(right_value_obj)
        if(left_value_obj.type() == Type.INT and right_value_obj.type() == Type.INT and  arith_ast.elem_type in ["&&", "||",]):
                left_value_obj = self.__convert_to_bool_if_int(left_value_obj)
                right_value_obj = self.__convert_to_bool_if_int(right_value_obj)
        if (left_value_obj.type() == Type.INT and right_value_obj.type() == Type.BOOL) or (left_value_obj.type() == Type.BOOL and right_value_obj.type() == Type.INT or (left_value_obj.type() == Type.BOOL and right_value_obj.type() == Type.BOOL)):
            # Convert bool to int for arithmetic operations
            if arith_ast.elem_type in ["+", "-", "*", "/"]:
                left_value_obj = self.__convert_to_int_if_bool(left_value_obj)
                right_value_obj = self.__convert_to_int_if_bool(right_value_obj)

        # Perform operation
        if not self.__compatible_types(arith_ast.elem_type, left_value_obj, right_value_obj):
            super().error(ErrorType.TYPE_ERROR, f"Incompatible types for {arith_ast.elem_type} operation")

        if arith_ast.elem_type not in self.op_to_lambda[left_value_obj.type()]:
            super().error(ErrorType.TYPE_ERROR, f"Incompatible operator {arith_ast.elem_type} for type {left_value_obj.type()}")

        f = self.op_to_lambda[left_value_obj.type()][arith_ast.elem_type]
        return f(left_value_obj, right_value_obj)

    def __compatible_types(self, oper, obj1, obj2):
        # DOCUMENT: allow comparisons ==/!= of anything against anything
        if oper in ["==", "!="]:
            return True
        return obj1.type() == obj2.type()

    def __eval_unary(self, arith_ast, t, f):
        value_obj = self.__eval_expr(arith_ast.get("op1"))

        # Check if the operand is a function or lambda
        if isinstance(value_obj, LambdaClosure) or (isinstance(value_obj, dict) and value_obj.get("type") == "function_ref"):
            super().error(ErrorType.TYPE_ERROR, "Invalid operation on functions or lambdas")

        # Convert int to bool if necessary
        if arith_ast.elem_type == Interpreter.NOT_DEF and value_obj.type() == Type.INT:
            value_obj = self.__convert_to_bool_if_int(value_obj)

        # Perform the unary operation
        if value_obj.type() != t:
            super().error(ErrorType.TYPE_ERROR, f"Incompatible type for {arith_ast.elem_type} operation")
        return Value(t, f(value_obj.value()))

    def __setup_ops(self):
        self.op_to_lambda = {}
        # set up operations on integers
        self.op_to_lambda[Type.INT] = {}
        self.op_to_lambda[Type.INT]["+"] = lambda x, y: Value(
            x.type(), x.value() + y.value()
        )
        self.op_to_lambda[Type.INT]["-"] = lambda x, y: Value(
            x.type(), x.value() - y.value()
        )
        self.op_to_lambda[Type.INT]["*"] = lambda x, y: Value(
            x.type(), x.value() * y.value()
        )
        self.op_to_lambda[Type.INT]["/"] = lambda x, y: Value(
            x.type(), x.value() // y.value()
        )
        self.op_to_lambda[Type.INT]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.INT]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )
        self.op_to_lambda[Type.INT]["<"] = lambda x, y: Value(
            Type.BOOL, x.value() < y.value()
        )
        self.op_to_lambda[Type.INT]["<="] = lambda x, y: Value(
            Type.BOOL, x.value() <= y.value()
        )
        self.op_to_lambda[Type.INT][">"] = lambda x, y: Value(
            Type.BOOL, x.value() > y.value()
        )
        self.op_to_lambda[Type.INT][">="] = lambda x, y: Value(
            Type.BOOL, x.value() >= y.value()
        )
        #  set up operations on strings
        self.op_to_lambda[Type.STRING] = {}
        self.op_to_lambda[Type.STRING]["+"] = lambda x, y: Value(
            x.type(), x.value() + y.value()
        )
        self.op_to_lambda[Type.STRING]["=="] = lambda x, y: Value(
            Type.BOOL, x.value() == y.value()
        )
        self.op_to_lambda[Type.STRING]["!="] = lambda x, y: Value(
            Type.BOOL, x.value() != y.value()
        )
        #  set up operations on bools
        self.op_to_lambda[Type.BOOL] = {}
        self.op_to_lambda[Type.BOOL]["&&"] = lambda x, y: Value(
            x.type(), x.value() and y.value()
        )
        self.op_to_lambda[Type.BOOL]["||"] = lambda x, y: Value(
            x.type(), x.value() or y.value()
        )
        self.op_to_lambda[Type.BOOL]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.BOOL]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )

        #  set up operations on nil
        self.op_to_lambda[Type.NIL] = {}
        self.op_to_lambda[Type.NIL]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.NIL]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )

    def __do_if(self, if_ast):
        cond_ast = if_ast.get("condition")
        result = self.__eval_expr(cond_ast)
        if result.type() == Type.INT:
            coerced_bool = result.value() != 0
            result = Value(Type.BOOL, coerced_bool)
        if result.type() != Type.BOOL:
            super().error(
                ErrorType.TYPE_ERROR,
                "Incompatible type for if condition",
            )
        if result.value():
            statements = if_ast.get("statements")
            status, return_val = self.__run_statements(statements)
            return (status, return_val)
        else:
            else_statements = if_ast.get("else_statements")
            if else_statements is not None:
                status, return_val = self.__run_statements(else_statements)
                return (status, return_val)

        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)


    def __do_while(self, while_ast):
        cond_ast = while_ast.get("condition")
        run_while = Interpreter.TRUE_VALUE
        while run_while.value():
            run_while = self.__eval_expr(cond_ast)
            if run_while.type() == Type.INT:
                coerced_bool = run_while.value() != 0
                run_while = Value(Type.BOOL, coerced_bool)
            if run_while.type() != Type.BOOL:
                super().error(
                    ErrorType.TYPE_ERROR,
                    "Incompatible type for while condition",
                )
            if run_while.value():
                statements = while_ast.get("statements")
                status, return_val = self.__run_statements(statements)
                if status == ExecStatus.RETURN:
                    return status, return_val

        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __do_return(self, return_ast):
        expr_ast = return_ast.get("expression")
        if expr_ast is None:
            return (ExecStatus.RETURN, Interpreter.NIL_VALUE)
        
        value_obj = self.__eval_expr(expr_ast)

        # Check if returning a function or a lambda closure
        if isinstance(value_obj, LambdaClosure) or (isinstance(value_obj, dict) and 'function' in value_obj):
            # Return a deep copy of the function or lambda closure
            return (ExecStatus.RETURN, copy.deepcopy(value_obj))

        return (ExecStatus.RETURN, value_obj)
    def __convert_to_bool_if_int(self, value):
        """Convert an integer value to boolean if necessary."""
        if isinstance(value, Value) and value.type() == Type.INT:
            return Value(Type.BOOL, value.value() != 0)
        return value
    def __convert_to_int_if_bool(self, value):
        """Convert a boolean value to an integer."""
        if isinstance(value, Value) and value.type() == Type.BOOL:
            return Value(Type.INT, int(value.value()))
        return value