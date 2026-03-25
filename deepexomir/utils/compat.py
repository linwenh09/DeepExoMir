"""Compatibility shim for multimolecule + transformers 5.x.

The ``multimolecule`` package (v0.0.9) imports ``check_model_inputs`` from
``transformers.utils.generic``, but this decorator was removed in
transformers 5.x.  This module patches the missing symbol so that
``multimolecule`` can be imported without errors.

Usage
-----
Import this module **before** anything that triggers a ``multimolecule``
import (e.g. ``AutoModel.from_pretrained('multimolecule/...')``):

    from deepexomir.utils.compat import patch_multimolecule_compat
    patch_multimolecule_compat()
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def _check_model_inputs_noop(func):
    """No-op replacement for the removed ``check_model_inputs`` decorator.

    In older transformers builds, ``check_model_inputs`` validated model
    forward-method arguments.  In 5.x the decorator was removed, so we
    provide a transparent pass-through.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def patch_multimolecule_compat() -> None:
    """Monkey-patch ``transformers.utils.generic`` if needed.

    Safe to call multiple times – subsequent calls are no-ops.
    """
    global _PATCHED
    if _PATCHED:
        return

    import transformers.utils.generic as _generic

    if not hasattr(_generic, "check_model_inputs"):
        _generic.check_model_inputs = _check_model_inputs_noop
        logger.debug(
            "Patched transformers.utils.generic.check_model_inputs "
            "(no-op shim for multimolecule compatibility)."
        )

    _PATCHED = True
