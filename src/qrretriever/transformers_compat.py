"""Compatibility shims for optional Transformers dependencies."""


def disable_optional_torchvision() -> None:
    """
    Avoid importing a broken optional torchvision install through Transformers.

    QRRetriever only uses text causal language models, but recent Transformers
    versions may import image utilities while loading modeling utilities. If a
    local torchvision build is incompatible with torch, that optional import can
    fail before QRRetriever code runs.
    """
    try:
        from transformers.utils import import_utils

        import_utils._torchvision_available = False
        import_utils._torchvision_version = "N/A"
    except Exception:
        pass

