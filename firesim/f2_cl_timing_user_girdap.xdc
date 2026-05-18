# FireSim AWS F2 timing exceptions for Girdap experiments.
#
# Purpose:
#   This file records the timing exceptions validated against the routed
#   MatmulAccel4x4WSBF16Config F2 build reports from 2026-04-28.
#
# Validated effect:
#   The following clock-pair rows moved to "Ignored / False Path" in
#   report_clock_interaction:
#     - mmcm_clkout0 -> WRAPPER/CL/clk_main_a0
#     - WRAPPER/CL/clk_main_a0 -> mmcm_clkout0
#     - clk_main_a0 -> WRAPPER/CL/clk_main_a0
#     - WRAPPER/CL/clk_main_a0 -> clk_main_a0
#
# Apply in the generated F2 CL build as cl_timing_user.xdc, typically under:
#   platforms/f2/aws-fpga-firesim-f2/hdk/cl/developer_designs/<cl_f2...>/build/constraints/
#
# Do not use this to hide real same-clock timing failures. The later
# rst_main_n_sync_reg_replica high-fanout reset path is intentionally not cut
# here because it is a physical reset distribution/timing-closure issue.

# The shell exposes both clk_main_a0 and WRAPPER/CL/clk_main_a0 as distinct
# clock objects. In this FireSim F2 implementation, Vivado otherwise times
# crossings between them and reports unsafe partial false paths.
set_false_path -from [get_clocks -quiet {WRAPPER/CL/clk_main_a0}] -to [get_clocks -quiet {clk_main_a0}]
set_false_path -from [get_clocks -quiet {clk_main_a0}] -to [get_clocks -quiet {WRAPPER/CL/clk_main_a0}]

# The DDR MIG/MMCM clock is asynchronous to the CL main clock. Without these
# exceptions, Vivado reports false SH_DDR recovery/setup failures such as:
#   SH_DDR/.../div_clk_rst_r1_reg_replica/C, mmcm_clkout0 -> WRAPPER/CL/clk_main_a0
set_false_path -from [get_clocks -quiet {mmcm_clkout0}] -to [get_clocks -quiet {WRAPPER/CL/clk_main_a0}]
set_false_path -from [get_clocks -quiet {WRAPPER/CL/clk_main_a0}] -to [get_clocks -quiet {mmcm_clkout0}]

# Keep the direct clk_main_a0 form too, in case a generated build reports the
# main clock without the WRAPPER/CL hierarchy prefix.
set_false_path -from [get_clocks -quiet {mmcm_clkout0}] -to [get_clocks -quiet {clk_main_a0}]
set_false_path -from [get_clocks -quiet {clk_main_a0}] -to [get_clocks -quiet {mmcm_clkout0}]

# Existing FireSim/F2 reset synchronizer exceptions that removed the earlier
# ddr_ready CLR false failures from the top timing paths.
set_false_path -to [get_pins -quiet -hierarchical {*ddr_ready_pre_sync_meta_reg/CLR}]
set_false_path -to [get_pins -quiet -hierarchical {*ddr_ready_sync_reg/CLR}]

# Reset synchronizer CLR exceptions observed in the generated F2 report.
set_false_path -to [get_pins -quiet -hierarchical -filter {NAME =~ */rst_main_n_sync_reg/CLR}]
set_false_path -to [get_pins -quiet -hierarchical -filter {NAME =~ */rst_main_n_sync_reg_replica/CLR}]
