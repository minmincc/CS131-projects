class EnvironmentManager:
    def __init__(self):
        self.environment = [{}]

    # returns a VariableDef object
    def get(self, symbol):
        for env in reversed(self.environment):
            if symbol in env:
                return env[symbol]

        return None

    def set(self, symbol, value):
        for env in reversed(self.environment):
            if symbol in env:
                env[symbol] = value
                return

        # symbol not found anywhere in the environment
        self.environment[-1][symbol] = value

    # create a new symbol in the top-most environment, regardless of whether that symbol exists
    # in a lower environment
    def create(self, symbol, value):
        self.environment[-1][symbol] = value

    # used when we enter a nested block to create a new environment for that block
    def push(self):
        self.environment.append({})  # [{}] -> [{}, {}]

    # used when we exit a nested block to discard the environment for that block
    def pop(self):
        self.environment.pop()

    def set_reference(self, name, reference):
        """Set a reference to a variable."""
        if reference and reference.get("type") == "ref":
            self.environment[-1][name] = reference

    def get_reference(self, symbol):
        """Retrieve a reference to a variable."""
        for env in reversed(self.environment):
            if symbol in env:
                return {"type": "ref", "env": env, "key": symbol}
        return None
