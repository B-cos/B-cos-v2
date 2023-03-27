from torchvision import models

__all__ = ["get_model"]


def get_model(model_config):
    assert not model_config.get("is_bcos", False), "Should be false!"

    # extract args
    arch_name = model_config["name"]
    args = model_config["args"]

    # create model
    model = getattr(models, arch_name)(**args)

    return model
