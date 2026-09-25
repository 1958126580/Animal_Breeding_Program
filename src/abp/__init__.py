"""ABP - auditable genetic evaluation for animal breeding.

The package is organised in layers (see docs/adr/0001-primary-language.md):

``abp.core``       model definitions and stable scientific kernels
``abp.io``         data contracts and file readers
``abp.solvers``    mixed-model equations, REML
``abp.decision``   selection indices (breeding decisions)
``abp.workflows``  pipelines, run manifests, reports, atomic outputs
``abp.cli``        command-line interface (``abp``)
"""

__version__ = "0.1.0"
