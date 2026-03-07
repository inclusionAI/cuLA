"""Minimal script for ncu profiling of flashla KDA kernel."""
import sys, pathlib, torch, torch.nn.functional as F
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from flashla.kda_wrapper import flash_kda_prefill
from benchmark.utils import set_seed, exclusive_cumsum

D = 128; DTYPE = torch.bfloat16; DEV = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True

def run(total_T, N, H):
    set_seed(42)
    base = total_T // N; rem = total_T % N
    seq_lens = [base]*(N-1) + [base+rem]
    cu = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.long, device=DEV)
    q = F.normalize(torch.randn(1,total_T,H,D,dtype=DTYPE,device=DEV),p=2,dim=-1)
    k = F.normalize(torch.randn(1,total_T,H,D,dtype=DTYPE,device=DEV),p=2,dim=-1)
    v = torch.randn(1,total_T,H,D,dtype=DTYPE,device=DEV)
    g = F.logsigmoid(torch.randn(1,total_T,H,D,dtype=torch.float,device=DEV)).clamp(-5,0)
    beta = torch.randn(1,total_T,H,dtype=torch.float32,device=DEV).sigmoid()
    h0 = torch.randn(N,H,D,D,dtype=torch.float32,device=DEV)
    scale = D**-0.5

    # warmup (outside profiled region)
    for _ in range(3):
        flash_kda_prefill(q=q,k=k,v=v,g=g,beta=beta,scale=scale,
            initial_state=h0,output_final_state=True,safe_gate=True,cu_seqlens=cu)
    torch.cuda.synchronize()

    # profiled run
    flash_kda_prefill(q=q,k=k,v=v,g=g,beta=beta,scale=scale,
        initial_state=h0,output_final_state=True,safe_gate=True,cu_seqlens=cu)
    torch.cuda.synchronize()

if __name__ == "__main__":
    # Single representative config: H=64, T=32768, N=8
    print("Running flashla KDA kernel for ncu profiling...")
    run(32768, 8, 64)
    print("Done.")
