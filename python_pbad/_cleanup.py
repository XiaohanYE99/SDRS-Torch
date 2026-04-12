import os

path = os.path.join(os.path.dirname(__file__), 'simulator.py')

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f'Original line count: {len(lines)}')

# Verify Task 1 boundaries (1-indexed lines, 0-indexed in list)
print('=== Task 1 boundaries ===')
print(f'Line 2439: {lines[2438].rstrip()!r}')
print(f'Line 2440: {lines[2439].rstrip()!r}')
print(f'Line 2441 (first 50): {lines[2440][:50]!r}')
print(f'Line 2508: {lines[2507].rstrip()!r}')
print(f'Line 2509: {lines[2508].rstrip()!r}')
print(f'Line 2510: {lines[2509].rstrip()!r}')

# Verify Task 2 boundaries
print('=== Task 2 boundaries ===')
print(f'Line 42: {lines[41].rstrip()!r}')
print(f'Line 43 (first 50): {lines[42][:50]!r}')
print(f'Line 133: {lines[132].rstrip()!r}')
print(f'Line 134: {lines[133].rstrip()!r}')
print(f'Line 135: {lines[134].rstrip()!r}')

# Task 1: Remove lines 2441-2508 (indices 2440-2507)
new_lines = lines[:2440] + lines[2508:]

# Task 2: Remove lines 43-133 (indices 42-132) from the ALREADY modified list
# After Task 1, the indices for lines 1-2440 are unchanged
new_lines = new_lines[:42] + new_lines[133:]

print(f'\nNew line count: {len(new_lines)}')

# Verify the join looks correct around both edit sites
# After removal, old line 134 is now at index 42
print('\n=== After edit: around ACM removal site ===')
for i in range(40, 46):
    print(f'  new[{i}]: {new_lines[i].rstrip()!r}')

# After both removals, the old line 2509 is now shifted
# Original idx 2508 -> after Task1 removal it becomes idx 2508-68=2440
# Then after Task2 removal it becomes 2440-91=2349
shifted_idx = 2508 - 68 - 91
print(f'\n=== After edit: around Newton-LS removal site (new idx ~{shifted_idx}) ===')
for i in range(shifted_idx - 2, shifted_idx + 5):
    print(f'  new[{i}]: {new_lines[i].rstrip()!r}')

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('\nFile written successfully.')
print(f'Final line count: {len(new_lines)}')
