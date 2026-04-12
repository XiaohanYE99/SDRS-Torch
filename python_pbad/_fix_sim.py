"""Delete _solve_newton_ls remnant, ACM top-level code, and debug_verify_manifold from simulator.py"""
import re

with open('simulator.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Original: {len(lines)} lines")

# 1. Delete lines from after "return theta, manifolds, H_bar_last" (line 2448, 0-idx 2447)
#    to before "def _snapshot_pair_manifolds" 
start_del = None
end_del = None
for i in range(2447, len(lines)):
    stripped = lines[i].strip()
    if stripped == 'return theta, manifolds, H_bar_last' and start_del is None:
        start_del = i + 1  # delete starting from next line
    if '    def _snapshot_pair_manifolds(self):' in lines[i] and start_del is not None:
        end_del = i
        break

if start_del and end_del:
    print(f"Deleting newton_ls remnant: lines {start_del+1}-{end_del} ({end_del - start_del} lines)")
    lines = lines[:start_del] + lines[end_del:]
else:
    print(f"WARNING: newton_ls remnant not found (start={start_del}, end={end_del})")

# 2. Delete ACM top-level: ACM_CONTACT_ENERGY_README and acm_nested_eval
acm_start = None
acm_end = None
for i, line in enumerate(lines):
    if '# ============================================================================' in line and acm_start is None:
        # Check if next line mentions ACM
        if i+1 < len(lines) and 'ACM' in lines[i+1]:
            acm_start = i
    if line.startswith('def acm_nested_eval(') and acm_start is not None:
        # Find end of this function
        for j in range(i+1, len(lines)):
            if lines[j].strip() and not lines[j].startswith(' ') and not lines[j].startswith('\t'):
                acm_end = j
                break
        break

if acm_start is not None and acm_end is not None:
    print(f"Deleting ACM block: lines {acm_start+1}-{acm_end} ({acm_end - acm_start} lines)")
    lines = lines[:acm_start] + lines[acm_end:]
else:
    print(f"WARNING: ACM block not found (start={acm_start}, end={acm_end})")

# 3. Delete _normalize_contact_barrier_mode
ncb_start = None
ncb_end = None
for i, line in enumerate(lines):
    if line.startswith('def _normalize_contact_barrier_mode('):
        ncb_start = i
    if ncb_start is not None and i > ncb_start and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
        ncb_end = i
        break

if ncb_start is not None and ncb_end is not None:
    print(f"Deleting _normalize_contact_barrier_mode: lines {ncb_start+1}-{ncb_end} ({ncb_end - ncb_start} lines)")
    lines = lines[:ncb_start] + lines[ncb_end:]
else:
    print(f"WARNING: _normalize_contact_barrier_mode not found")

# 4. Delete debug_verify_manifold_p_u_derivatives
dvm_start = None
dvm_end = None
for i, line in enumerate(lines):
    if '    def debug_verify_manifold_p_u_derivatives(' in line:
        dvm_start = i
    if dvm_start is not None and i > dvm_start:
        # Find next method at same indent level
        if line.startswith('    def ') and i > dvm_start + 1:
            dvm_end = i
            break

if dvm_start is not None and dvm_end is not None:
    print(f"Deleting debug_verify_manifold_p_u_derivatives: lines {dvm_start+1}-{dvm_end} ({dvm_end - dvm_start} lines)")
    lines = lines[:dvm_start] + lines[dvm_end:]
else:
    print(f"WARNING: debug_verify_manifold_p_u_derivatives not found (start={dvm_start}, end={dvm_end})")

print(f"Final: {len(lines)} lines")

with open('simulator.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Done!")
