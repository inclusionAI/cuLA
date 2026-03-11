#!/usr/bin/env python3
"""Extract key NCU metrics from .ncu-rep files for comparison."""
import subprocess, csv, io, sys

def query_metrics(rep_file, metrics, kernel_filter=None):
    cmd = ["ncu", "--import", rep_file, "--csv", "--page", "raw",
           "--metrics", ",".join(metrics)]
    if kernel_filter:
        cmd += ["--kernel-name", f"regex:{kernel_filter}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    reader = csv.reader(io.StringIO(result.stdout))
    header = next(reader)  # metric names as column headers
    units = next(reader)   # units row
    
    fixed_cols = ["ID", "Process ID", "Process Name", "Host Name", "Kernel Name",
                  "Context", "Stream", "Block Size", "Grid Size", "Device", "CC"]
    n_fixed = len(fixed_cols)
    
    metric_names = header[n_fixed:]
    metric_units = units[n_fixed:]
    
    for row in reader:
        if row[0] == "0":
            values = row[n_fixed:]
            result_dict = {}
            for i, mname in enumerate(metric_names):
                result_dict[mname] = (values[i], metric_units[i])
            return result_dict
    return {}

all_metrics = [
    "gpu__time_duration.sum",
    "launch__grid_size", "launch__block_size",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_dynamic",
    "launch__shared_mem_per_block_static",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.per_cycle_active",
    "sm__maximum_warps_avg_per_active_cycle",
    "smsp__average_warp_latency_per_inst_issued.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_selected_per_issue_active.ratio",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_issued.avg.per_cycle_active",
    "sm__inst_executed.avg.per_cycle_elapsed",
    "dram__bytes.sum",
    "lts__t_sector_hit_rate.pct",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
]

def print_metrics(label, rep_file, kernel_filter=None):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    d = query_metrics(rep_file, all_metrics, kernel_filter)
    if not d:
        print("  No data found!")
        return d
    
    def v(metric):
        val, unit = d.get(metric, ("?", ""))
        return f"{val} {unit}".strip()
    
    print(f"\n  Duration: {v('gpu__time_duration.sum')}")
    
    print(f"\n  --- Launch Config ---")
    print(f"    Grid:        {v('launch__grid_size')}")
    print(f"    Block:       {v('launch__block_size')}")
    print(f"    Regs/thread: {v('launch__registers_per_thread')}")
    print(f"    SMEM dyn:    {v('launch__shared_mem_per_block_dynamic')}")
    print(f"    SMEM static: {v('launch__shared_mem_per_block_static')}")
    print(f"    Occ limit (regs):  {v('launch__occupancy_limit_registers')}")
    print(f"    Occ limit (smem):  {v('launch__occupancy_limit_shared_mem')}")
    
    print(f"\n  --- Throughput (% of peak) ---")
    print(f"    SM:    {v('sm__throughput.avg.pct_of_peak_sustained_elapsed')}")
    print(f"    DRAM:  {v('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed')}")
    print(f"    L1TEX: {v('l1tex__throughput.avg.pct_of_peak_sustained_elapsed')}")
    print(f"    L2:    {v('lts__throughput.avg.pct_of_peak_sustained_elapsed')}")
    
    print(f"\n  --- Occupancy ---")
    print(f"    Active warps/cycle: {v('sm__warps_active.avg.per_cycle_active')}")
    print(f"    Max warps/cycle:    {v('sm__maximum_warps_avg_per_active_cycle')}")
    
    print(f"\n  --- Warp Stall Reasons ---")
    print(f"    Avg latency/inst: {v('smsp__average_warp_latency_per_inst_issued.ratio')} cycles")
    stall_names = [
        ("long_scoreboard", "GMEM/L2 dep"),
        ("short_scoreboard", "SMEM/L1 dep"),
        ("wait", "fixed latency dep"),
        ("math_pipe_throttle", "math pipe busy"),
        ("barrier", "barrier sync"),
        ("not_selected", "ready but not picked"),
        ("mio_throttle", "MIO pipe busy"),
        ("lg_throttle", "LG pipe busy"),
        ("tex_throttle", "TEX pipe busy"),
        ("no_instruction", "no instr in I-buf"),
        ("sleeping", "sleeping/yield"),
        ("drain", "drain after exit"),
        ("dispatch_stall", "dispatch stall"),
        ("membar", "memory barrier"),
        ("selected", "ISSUED (selected)"),
    ]
    for sname, desc in stall_names:
        metric = f"smsp__average_warps_issue_stalled_{sname}_per_issue_active.ratio"
        print(f"    {desc:30s}: {v(metric)}")
    
    print(f"\n  --- Pipe Utilization (% of peak elapsed) ---")
    for p in ["tensor", "fma", "alu", "shared", "tma"]:
        m = f"sm__pipe_{p}_cycles_active.avg.pct_of_peak_sustained_elapsed"
        print(f"    {p:12s}: {v(m)}")
    
    print(f"\n  --- IPC ---")
    print(f"    Issued/cycle (active):    {v('smsp__inst_issued.avg.per_cycle_active')}")
    print(f"    Executed/cycle (elapsed): {v('sm__inst_executed.avg.per_cycle_elapsed')}")
    
    print(f"\n  --- Memory ---")
    print(f"    DRAM bytes total:      {v('dram__bytes.sum')}")
    print(f"    L2 hit rate:           {v('lts__t_sector_hit_rate.pct')}")
    print(f"    SMEM load wavefronts:  {v('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum')}")
    print(f"    SMEM store wavefronts: {v('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum')}")
    
    return d

if __name__ == "__main__":
    d1 = print_metrics("OUR KERNEL (CuTe DSL)", "ncu_ours.ncu-rep", "kernel_cutlass")
    d2 = print_metrics("FLA KERNEL (Triton)", "ncu_fla.ncu-rep", "chunk_gated")
    
    if d1 and d2:
        print(f"\n{'='*60}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*60}")
        t1 = float(d1.get("gpu__time_duration.sum", ("0",""))[0])
        t2 = float(d2.get("gpu__time_duration.sum", ("0",""))[0])
        print(f"  Our kernel: {t1:.2f} us")
        print(f"  FLA kernel: {t2:.2f} us")
        if t1 > 0:
            print(f"  Speedup:    {t2/t1:.2f}x")
