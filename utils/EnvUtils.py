import os
def use_modelscope() -> bool:
    return bool(int(os.environ.get(""
                                   "USE_MODELSCOPE_HUB", "0")))