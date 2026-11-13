import cProfile, importlib, sys, unittest
name, out = sys.argv[1], sys.argv[2]
sys.path[:0] = ['tests', 'src']; sys.argv = ['tests/' + name + '.py']
pr = cProfile.Profile(); pr.enable()
mod = importlib.import_module(name)
unittest.main(module=mod, exit=False, verbosity=0)
pr.disable(); pr.dump_stats(out)
