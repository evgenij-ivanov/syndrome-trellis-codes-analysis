import unittest
import stc
import common
import numpy as np

class TestStcMethod(unittest.TestCase):
    
    def test_get_h(self):
        h = stc.get_h([[1, 0], [1, 1]], 3, 4)
        self.assertEqual(len(h), 3)
        self.assertEqual(len(h[0]), 4)
        self.assertTrue(h.tolist(), np.array([[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]]).tolist())
        
    def test_binary_to_message(self):
        message = common.binary_to_message('01010000011000010010010000100100')
        self.assertEqual(message, 'Pa$$')
        
    def test_message_to_binary(self):
        binary_str = common.message_to_binary('Pa$$')
        self.assertEqual(binary_str, '01010000011000010010010000100100')
        
if __name__ == '__main__':
    unittest.main()
        