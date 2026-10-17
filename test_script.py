from collections import deque

history = deque([1, 2, 3000, 3500, 3550, 3600])

hour_ago = 100
minute_ago = 3540

# Original
in_hour = sum(1 for stamp in history if stamp >= hour_ago)
in_minute = sum(1 for stamp in history if stamp >= minute_ago)
print(f"Original: in_hour={in_hour}, in_minute={in_minute}")

# Optimized
opt_in_hour = 0
opt_in_minute = 0
for stamp in reversed(history):
    if stamp < hour_ago:
        break
    opt_in_hour += 1
    if stamp >= minute_ago:
        opt_in_minute += 1

print(f"Optimized: in_hour={opt_in_hour}, in_minute={opt_in_minute}")
