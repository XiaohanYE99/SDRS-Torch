"""Clean batch_simulator.py: remove ACM, newton_ls, friction_eps params, debug delegates"""

with open('batch_simulator.py', 'r', encoding='utf-8') as f:
    content = f.read()

original_len = content.count('\n') + 1
print(f"Original: {original_len} lines")

import re

# 1. Remove ACM imports: acm_nested_eval, _normalize_contact_barrier_mode
content = content.replace(
    "from simulator import (\n    barrier_eval,\n    acm_nested_eval,\n    _normalize_contact_barrier_mode,\n    ConvexHullPBADSimulator,\n)",
    "from simulator import (\n    barrier_eval,\n    ConvexHullPBADSimulator,\n)")

# 2. Fix __init__ - remove solver, friction_eps_s/n, contact_barrier, acm params, contact_distance_mode
# This needs careful line-by-line work
lines = content.split('\n')

# Find and rewrite __init__ signature and body
new_lines = []
i = 0
in_init = False
init_done = False
skip_until_next_def = False

# Track which debug methods to delete
delete_methods = ['debug_energy', 'debug_backward', 'debug_verify_theta_derivatives_fd',
                  '_debug_wv_row', '_debug_num_envs_to_run', '_reference_single_sim']

while i < len(lines):
    line = lines[i]
    
    # Delete debug delegate methods (they just call single-env)
    should_delete = False
    for dm in delete_methods:
        if f'    def {dm}(' in line:
            should_delete = True
            break
    
    if should_delete:
        # Skip until next method at same level
        i += 1
        while i < len(lines):
            if lines[i].startswith('    def ') or (lines[i].strip().startswith('# ==') and i+1 < len(lines) and lines[i+1].strip().startswith('#  IFT')):
                break
            i += 1
        continue
    
    new_lines.append(line)
    i += 1

content = '\n'.join(new_lines)
final_len = content.count('\n') + 1
print(f"After debug method removal: {final_len} lines (removed {original_len - final_len})")

with open('batch_simulator.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done!")
