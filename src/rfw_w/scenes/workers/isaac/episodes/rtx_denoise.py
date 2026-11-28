# SPDX-License-Identifier: Apache-2.0
"""Optional CPU OIDN postprocessing of Isaac RTX pixels, with no geometry changes."""
import ctypes as C
import numpy as np


class Denoiser:
    def __init__(self, library, width, height):
        self.lib = C.CDLL(library)
        signatures = {
            'oidnNewDevice': ([C.c_int], C.c_void_p),
            'oidnCommitDevice': ([C.c_void_p], None),
            'oidnNewFilter': ([C.c_void_p, C.c_char_p], C.c_void_p),
            'oidnSetSharedFilterImage': ([C.c_void_p, C.c_char_p, C.c_void_p, C.c_int, C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t], None),
            'oidnSetFilterBool': ([C.c_void_p, C.c_char_p, C.c_bool], None),
            'oidnCommitFilter': ([C.c_void_p], None),
            'oidnExecuteFilter': ([C.c_void_p], None),
            'oidnGetDeviceError': ([C.c_void_p, C.POINTER(C.c_char_p)], C.c_int),
        }
        for name, (args, result) in signatures.items():
            f = getattr(self.lib, name)
            f.argtypes, f.restype = args, result
        self.device = self.lib.oidnNewDevice(1)
        self.lib.oidnCommitDevice(self.device)
        self.filter = self.lib.oidnNewFilter(self.device, b'RT')
        self.source = np.zeros((height, width, 3), dtype=np.float32)
        self.target = np.zeros_like(self.source)
        for name, data in [(b'color', self.source), (b'output', self.target)]:
            self.lib.oidnSetSharedFilterImage(self.filter, name, data.ctypes.data, 3, width, height, 0, 12, width * 12)
        self.lib.oidnSetFilterBool(self.filter, b'hdr', False)
        self.lib.oidnSetFilterBool(self.filter, b'srgb', True)
        self.lib.oidnCommitFilter(self.filter)

    def __call__(self, image):
        self.source[:] = image / 255.0
        self.lib.oidnExecuteFilter(self.filter)
        message = C.c_char_p()
        if self.lib.oidnGetDeviceError(self.device, C.byref(message)):
            raise RuntimeError(message.value)
        return np.clip(self.target * 255, 0, 255).astype(np.uint8)
