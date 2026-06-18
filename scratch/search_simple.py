html_path = "/Users/admin/Documents/dev_src/stress_index/scratch/post_12854.html"

with open(html_path, "r", encoding="utf-8") as f:
    content = f.read()

print(f"Total HTML file length: {len(content)}")

# Let's search for keywords case-insensitively
keywords = ["svr", "lunch", "standardscaler", "robustscaler", "minmaxscaler", "kernel", "epsilon", "seed"]
for kw in keywords:
    pos = 0
    found_count = 0
    while True:
        pos = content.lower().find(kw, pos)
        if pos == -1:
            break
        found_count += 1
        # Print a snippet of 200 chars around the match
        snippet = content[max(0, pos-100):min(len(content), pos+100)]
        # Clean snippet for printable characters
        snippet_clean = "".join(c if c.isprintable() else " " for c in snippet)
        print(f"Keyword '{kw}' match {found_count} at index {pos}:")
        print(f"  ... {snippet_clean} ...")
        pos += len(kw)
    print(f"Keyword '{kw}' total matches: {found_count}\n")
