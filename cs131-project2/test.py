from brewparse import parse_program
from interpreterv3 import Interpreter

def run_interpreter(source_code):
    try:
        # Parse the source code into an AST
        parsed_program = parse_program(source_code)
        
        # Initialize the interpreter with the AST
        interpreter = Interpreter()
        
        # Run the interpreter on the AST
        interpreter.run(source_code)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    program_source = """
        func foo(x) {
        if (x < 0) {
            print(x);
            return -x;
            print("this will not print");
        }
        print("this will not print either");
        return 5*x;
        }

        func main() {
        b = "4abc";
        c = "nil";
        a = c&&c+b;
        print(a);
        }
    """

    run_interpreter(program_source)