from pathlib import Path

p = Path("step2_produce_tweet_prev_none.md")  # adjust path if needed
data = p.read_bytes()

# Find the exact position of the bad byte 0x9C
pos = data.find(bytes([0x9C]))
print("0x9C position:", pos)

# Print surrounding bytes for diagnosis
if pos != -1:
    start = max(0, pos - 20)
    end = min(len(data), pos + 20)
    print("Surrounding bytes:", data[start:end])

# Also list any non-ASCII characters (helpful even if 0x9C not found)
text = data.decode("utf-8", errors="replace")
non_ascii = [(i, ch, hex(ord(ch))) for i, ch in enumerate(text) if ord(ch) > 127]
print("Non-ASCII count:", len(non_ascii))
print("First 20 non-ASCII:", non_ascii[:20])
