"""Parse and compare 4 NCU profiles: ours_g, ours_no_g, fla_g, fla_no_g."""
import subprocess
import re
import sys

REPORT = "/tmp/ncu_gating.ncu-rep"

# Key metrics to extract
METRICS = [
    "gpu__time_duration.avg",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_dynamic",
    "launch__grid_size",
    "launch__block_size",
    "sm__warps_active.avg.per_cycle_active",
    "smsp__average_warp_latency_per_inst_issued.ratio",
    "smsp__inst_executed.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    # PM sampling stalls
    "smsp__pcsamp_sample_count",
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_barrier",
    "smsp__pcsamp_warps_issue_stalled_wait",
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_not_selected",
    "smsp__pcsamp_warps_issue_stalled_sleeping",
    "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
    "smsp__pcsamp_warps_issue_stalled_mio_throttle",
    "smsp__pcsamp_warps_issue_stalled_no_instructions",
    "smsp__pcsamp_warps_issue_stalled_misc",
    "smsp__pcsamp_warps_issue_stalled_selected",
]


def get_all_kernels_raw(report_path):
    """Extract raw metrics for all kernels in a report."""
    metric_str = ",".join(METRICS)
    cmd = f'ncu --import {report_path} --page raw --metrics "{metric_str}" 2>&1'
    output = subprocess.check_output(cmd, shell=True, text=True)

    kernels = []
    current = {}
    for line in output.strip().split('\n'):
        line = line.rstrip()
        if not line or line.startswith('=='):
            continue
        # Detect kernel header (starts with '[')
        if line.startswith('['):
            if current:
                kernels.append(current)
            current = {'name': '', 'metrics': {}}
            continue
        # Kernel name line (indented, contains the kernel name)
        if line.strip().startswith('kernel_cutlass') or line.strip().startswith('chunk_gated'):
            current['name'] = line.strip().split()[0][:50]
            continue
        # Header line
        if 'Metric Name' in line and 'Metric Value' in line:
            continue
        if '--------' in line:
            continue
        # Metric line: "  metric_name   unit   value"
        parts = line.split()
        if len(parts) >= 2:
            name = parts[0]
            val_str = parts[-1]
            if name in METRICS:
                try:
                    val = float(val_str.replace(',', ''))
                except ValueError:
                    val = val_str
                current['metrics'][name] = val

    if current:
        kernels.append(current)
    return kernels


def fmt_val(v):
    if isinstance(v, float):
        if abs(v) >= 1e9:
            return f"{v/1e9:.2f}G"
        elif abs(v) >= 1e6:
            return f"{v/1e6:.1f}M"
        elif abs(v) >= 1e3:
            return f"{v/1e3:.1f}K"
        elif v == int(v):
            return f"{int(v)}"
        else:
            return f"{v:.2f}"
    return str(v)


def main():
    kernels = get_all_kernels_raw(REPORT)
    print(f"Found {len(kernels)} kernels\n")

    # Label them
    labels = ["Ours+G", "Ours-noG", "FLA+G", "FLA-noG"]
    if len(kernels) < 4:
        print("ERROR: Expected 4 kernels")
        for i, k in enumerate(kernels):
            print(f"  [{i}] {k.get('name', '?')}: {len(k['metrics'])} metrics")
        return

    # Print comparison table
    print(f"{'Metric':<55} {'Ours+G':>12} {'Ours-noG':>12} {'FLA+G':>12} {'FLA-noG':>12}")
    print("=" * 107)

    # Overview
    for m in ["gpu__time_duration.avg", "launch__registers_per_thread",
              "launch__shared_mem_per_block_dynamic", "launch__grid_size",
              "launch__block_size", "sm__warps_active.avg.per_cycle_active",
              "smsp__average_warp_latency_per_inst_issued.ratio",
              "smsp__inst_executed.sum"]:
        short = m.split("__")[-1].replace(".avg.per_cycle_active", "").replace(".avg", "").replace(".sum", "").replace(".ratio", "")
        vals = [fmt_val(kernels[i]['metrics'].get(m, 'N/A')) for i in range(4)]
        print(f"  {short:<53} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12}")

    print()
    # Memory
    print("--- Memory ---")
    for m in ["dram__bytes_read.sum", "dram__bytes_write.sum",
              "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
              "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
              "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"]:
        short = m.split("__", 1)[-1].replace(".sum", "").replace("pipe_lsu_mem_", "")
        vals = [fmt_val(kernels[i]['metrics'].get(m, 'N/A')) for i in range(4)]
        print(f"  {short:<53} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12}")

    print()
    # Stall breakdown (PM sampling)
    print("--- Stall Breakdown (PM sampling, warps) ---")
    total_samples = [kernels[i]['metrics'].get("smsp__pcsamp_sample_count", 1) for i in range(4)]

    stall_metrics = [
        "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
        "smsp__pcsamp_warps_issue_stalled_barrier",
        "smsp__pcsamp_warps_issue_stalled_wait",
        "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
        "smsp__pcsamp_warps_issue_stalled_not_selected",
        "smsp__pcsamp_warps_issue_stalled_sleeping",
        "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
        "smsp__pcsamp_warps_issue_stalled_mio_throttle",
        "smsp__pcsamp_warps_issue_stalled_no_instructions",
        "smsp__pcsamp_warps_issue_stalled_misc",
        "smsp__pcsamp_warps_issue_stalled_selected",
    ]

    for m in stall_metrics:
        short = m.replace("smsp__pcsamp_warps_issue_stalled_", "")
        vals_raw = [kernels[i]['metrics'].get(m, 0) for i in range(4)]
        # Show both count and percentage
        vals_str = []
        for i in range(4):
            v = vals_raw[i]
            if isinstance(v, (int, float)) and isinstance(total_samples[i], (int, float)) and total_samples[i] > 0:
                pct = v / total_samples[i] * 100
                vals_str.append(f"{fmt_val(v)} ({pct:.0f}%)")
            else:
                vals_str.append(fmt_val(v))
        print(f"  {short:<25} {vals_str[0]:>19} {vals_str[1]:>19} {vals_str[2]:>19} {vals_str[3]:>19}")

    # Summary
    print()
    print("--- Summary ---")
    t_ours_g = kernels[0]['metrics'].get('gpu__time_duration.avg', 0)
    t_ours_ng = kernels[1]['metrics'].get('gpu__time_duration.avg', 0)
    t_fla_g = kernels[2]['metrics'].get('gpu__time_duration.avg', 0)
    t_fla_ng = kernels[3]['metrics'].get('gpu__time_duration.avg', 0)
    print(f"  G-gating overhead (ours):  {t_ours_g:.1f} vs {t_ours_ng:.1f} us = +{(t_ours_g/t_ours_ng - 1)*100:.0f}%")
    print(f"  G-gating overhead (FLA):   {t_fla_g:.1f} vs {t_fla_ng:.1f} us = +{(t_fla_g/t_fla_ng - 1)*100:.0f}%")
    print(f"  Ours vs FLA (with G):      {t_ours_g:.1f} vs {t_fla_g:.1f} us = {t_fla_g/t_ours_g:.2f}x")
    print(f"  Ours vs FLA (no G):        {t_ours_ng:.1f} vs {t_fla_ng:.1f} us = {t_fla_ng/t_ours_ng:.2f}x")


if __name__ == "__main__":
    main()
