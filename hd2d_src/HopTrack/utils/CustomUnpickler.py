import pickle

class CustomUnpickler(pickle.Unpickler):
    def __init__(self, file, *args, **kwargs):
        super().__init__(file, *args, **kwargs)
        self.module_mapping = {"HopTrack": "hd2d_src.HopTrack.HopTrack", "src.HopTrack.HopTrack": "hd2d_src.HopTrack.HopTrack"}

    def find_class(self, module, name):
        try:
            if module in self.module_mapping:
                adjusted_module = self.module_mapping[module]
                return super().find_class(adjusted_module, name)
            return super().find_class(module, name)
        except ImportError as e:
            print(f"Failed to load class {module}.{name}: {e}")
            raise