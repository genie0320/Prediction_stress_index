import re
from urllib.parse import urlparse

html_path = "/Users/admin/Documents/dev_src/stress_index/scratch/post_12854.html"

with open(html_path, "r", encoding="utf-8") as f:
    content = f.read()

# Extract all absolute URLs
urls = re.findall(r'https?://[^\s"\'>]+', content)
domains = set()
for url in urls:
    # clean URL
    url_clean = url.split('"')[0].split("'")[0].split(')')[0].split(',')[0]
    parsed = urlparse(url_clean)
    domains.add(parsed.netloc)

print("=== Found Domains ===")
for d in sorted(domains):
    print(d)
    
# Let's search for "codeshare" inside URLs or paths in the HTML
matches = re.findall(r'[^\s"\'>]*codeshare[^\s"\'>]*', content)
print("\n=== Matches containing 'codeshare' ===")
for m in set(matches[:20]):
    print(m)
