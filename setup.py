import os
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "chaos_proxy",
        ["symbolic_chaos_impl.cpp", "chaos_proxy_bindings.cpp"],
        include_dirs=[pybind11.get_include(), "."],
        language="c++",
        extra_compile_args=["-O3", "-Wall", "-shared", "-std=c++17", "-fPIC"],
        extra_link_args=["-ltbb"],
    ),
]

setup(
    name="chaos_proxy",
    ext_modules=ext_modules,
)
