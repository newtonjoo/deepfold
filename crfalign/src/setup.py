from distutils.core import setup, Extension

setup(name = "pyCRF",
        version = "1.0",
        description = "print log",
        author = "InSilico",
        author_email = "newton@kias.re.kr",
        url = "http://lee.kias.re.kr",
        ext_modules = [Extension("pyCRF", ["pycrf.cpp"])]
        )
