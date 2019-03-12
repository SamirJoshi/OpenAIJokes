import unittest


def load_tests(loader, test_modules, pattern):
    suite = unittest.TestSuite()
    for t in test_modules:
        tests = loader.loadTestsFromTestCase(t)
        suite.addTests(tests)

    return suite


if __name__ == "__main__":
    import test.test_format as test_format
    import test.test_model as test_model

    loader = unittest.TestLoader()
    test_modules = [test_format.TestCodeFormat, test_model.TestModel]
    runner = unittest.TextTestRunner(verbosity=2)
    suite = load_tests(loader, test_modules, pattern=None)
    runner.run(suite)
    