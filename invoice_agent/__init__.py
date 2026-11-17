"""Acme invoice desk: extract, validate, approve, pay."""

import warnings

# Match on the message. Python applies the module filter to the full filename,
# so a package name never matches.
warnings.filterwarnings("ignore", message=r"urllib3 v2")
warnings.filterwarnings("ignore", message=r".*allowed_objects")

__version__ = "0.1.0"
