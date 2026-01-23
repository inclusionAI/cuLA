import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

import cutlass.torch as cutlass_torch

# D = 128

class DemoKernel:

    def __init__(
        self,
        D,
    ):
        self.D = D
    
    @cute.jit
    def __call__(
        self,
        F,
        stream,
    ):
        if cutlass.const_expr(self.D != 128):
            raise ValueError("D must be 128")
        if cutlass.const_expr(F != 128):
            raise ValueError("F must be 128")
        g_layout = cute.make_layout((self.D,1), stride=(1,self.D))
        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            sG_last: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(g_layout)], # type: ignore
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        print(f"SharedStorage size: {SharedStorage.__sizeof__()} bytes")
        self.kernel(g_layout).launch(
            grid=(1,1,1),
            block=(32,1,1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        g_layout,
    ):
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sG_last = storage.sG_last.get_tensor(
            g_layout,
            swizzle=None,
        )

        print(f"sG_last: {cute.pretty_str(sG_last)}")

    
if __name__ == "__main__":
    stream = cutlass_torch.default_stream()

    demo = DemoKernel(128)
    compiled = cute.compile(demo, 128, stream)
    compiled(stream)
    