"""
Dummy module for testing purposes.
This is a placeholder module that can be used for basic functionality testing.
"""


def dummy_function():
    """
    A simple dummy function that returns a greeting message.
    
    Returns:
        str: A dummy greeting message
    """
    return "Hello from dummy module!"


def dummy_echo(message):
    """
    A dummy echo function that returns the input message.
    
    Args:
        message (str): The message to echo
        
    Returns:
        str: The echoed message
    """
    return f"Echo: {message}"


if __name__ == "__main__":
    # Test the dummy functions
    print(dummy_function())
    print(dummy_echo("This is a test"))
