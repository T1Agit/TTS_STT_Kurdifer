"""
Test module for dummy.py
"""

import unittest
from dummy import dummy_function, dummy_echo


class TestDummy(unittest.TestCase):
    """Test cases for dummy module."""
    
    def test_dummy_function(self):
        """Test that dummy_function returns expected message."""
        result = dummy_function()
        self.assertEqual(result, "Hello from dummy module!")
        self.assertIsInstance(result, str)
    
    def test_dummy_echo(self):
        """Test that dummy_echo correctly echoes messages."""
        test_message = "Test message"
        result = dummy_echo(test_message)
        self.assertEqual(result, f"Echo: {test_message}")
        self.assertIn(test_message, result)
    
    def test_dummy_echo_empty(self):
        """Test that dummy_echo handles empty strings."""
        result = dummy_echo("")
        self.assertEqual(result, "Echo: ")


if __name__ == "__main__":
    unittest.main()
