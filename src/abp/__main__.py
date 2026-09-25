"""Allow ``python -m abp``."""
import sys

from .cli import main

sys.exit(main())
