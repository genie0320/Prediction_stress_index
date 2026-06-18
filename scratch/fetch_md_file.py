import urllib.request
import re

html_path = "/Users/admin/Documents/dev_src/stress_index/scratch/post_12854.html"

with open(html_path, "r", encoding="utf-8") as f:
    content = f.read()

# Search for md_file value
# Example: md_file:"codeshare\u002F236526\u002F12854\u002Fmd_file.html"
match = re.search(r'md_file:\s*"([^"]+)"', content)

if match:
    raw_path = match.group(1)
    # Clean the backslashes and unicode slash escapes
    clean_path = raw_path.replace(r'\u002F', '/')
    print("Extracted path:", clean_path)
    
    # Let's try downloading from different domains
    domains = [
        "https://dacon.s3.ap-northeast-2.amazonaws.com/",
        "https://dacon.io/",
        "https://r2-library.dacon.co.kr/"
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    success = False
    for domain in domains:
        url = domain + clean_path
        print(f"Trying URL: {url} ...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                md_content = response.read().decode('utf-8')
            print(f"  Success from {domain}! Content size: {len(md_content)}")
            
            # Save it
            out_path = "/Users/admin/Documents/dev_src/stress_index/scratch/svr_solution.html"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            print(f"Saved to {out_path}")
            
            # Print a snippet of content
            print("\n=== Solution Snippet ===")
            print(md_content[:3000])
            print("========================")
            
            success = True
            break
        except Exception as e:
            print(f"  Failed: {e}")
            
    if not success:
        print("Could not download the md_file from any of the domains.")
else:
    print("Could not find md_file field in HTML.")
