import unittest
import pep8
from os import listdir
from os.path import isfile, join

class TestCodeFormat(unittest.TestCase):
    def test_pep8_conformance(self):
        """ Test that code follows PEP8 style """
        pep8_style = pep8.StyleGuide(quiet=True)
        dir_prefix = "src"
        files = [join(dir_prefix, f) for f in listdir(dir_prefix) if isfile(join(dir_prefix, f)) and f.endswith('.py')]
        result = pep8_style.check_files(files)
        stats = "\n".join(result.get_statistics())
        
        self.assertEqual(
            result.total_errors,
            0,
            'Found code style errors and warnings:\n' + stats
        )