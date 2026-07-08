def __getattr__(name):
    if name in {"HubertMambaModel", "HubertMambaOutput"}:
        from .hubert_mamba import HubertMambaModel, HubertMambaOutput

        return {"HubertMambaModel": HubertMambaModel, "HubertMambaOutput": HubertMambaOutput}[name]
    raise AttributeError(name)


__all__ = ["HubertMambaModel", "HubertMambaOutput"]
