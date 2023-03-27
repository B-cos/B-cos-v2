import bcos

dependencies = ["torch", "torchvision"]
globals().update(bcos.pretrained._entrypoint_registry)  # noqa
