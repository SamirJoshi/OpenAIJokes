import unittest
import pep8


class TestCodeFormat(unittest.TestCase):
    def test_pep8_conformance(self):
        """ Test that code follows PEP8 style """
        pep8_style = pep8.StyleGuide(quiet=True)
        files = ["src/dataset.py", "src/model.py"]
        result = pep8_style.check_files(files)
        self.assertEqual(result.total_errors, 0, "Found code style errors and warnings")
