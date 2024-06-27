import os
import shutil as sf

path=os.getcwd()
sf.ratio(path='/data', output=path, seed=1337, ratio=(.8, .1, .1))