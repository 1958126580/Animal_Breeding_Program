"""Build hook for the optional C++20 kernel ``abp._native`` (see ADR 0002).

All metadata lives in pyproject.toml.  The extension is marked optional: if
no C++20 compiler is available, installation still succeeds and ABP uses its
pure-Python reference kernels (the run manifest records which one was used).
"""

import sys

from setuptools import Extension, setup

if sys.platform == "win32":
    compile_args = ["/std:c++20", "/O2", "/EHsc"]
else:
    compile_args = ["-std=c++20", "-O3"]

setup(
    ext_modules=[
        Extension(
            "abp._native",
            sources=["src/abp/_native.cpp"],
            extra_compile_args=compile_args,
            language="c++",
            optional=True,
        )
    ]
)
