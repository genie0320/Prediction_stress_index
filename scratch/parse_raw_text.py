import re
import html
import json

html_path = "/Users/admin/Documents/dev_src/stress_index/scratch/post_12854.html"

with open(html_path, "r", encoding="utf-8") as f:
    content = f.read()

# Let's search for window.__NUXT__ string
nuxt_match = re.search(r'window\.__NUXT__\s*=(.*?);\s*</script>', content, re.DOTALL)

if nuxt_match:
    print("Found window.__NUXT__!")
    nuxt_text = nuxt_match.group(1)
    
    # In window.__NUXT__, text blocks are usually string literals enclosed in quotes.
    # Let's extract all double-quoted and single-quoted strings that contain "SVR"
    # To be safe and simple, let's find strings of length > 50 containing "SVR"
    pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
    strings = re.findall(pattern, nuxt_text)
    
    # Also try single quotes
    pattern_single = r"'([^'\\]*(?:\\.[^'\\]*)*)'"
    strings_single = re.findall(pattern_single, nuxt_text)
    
    all_strings = strings + strings_single
    
    print(f"Extracted {len(all_strings)} string literals.")
    
    matches = []
    for s in all_strings:
        # Unescape unicode characters if any (like \u003C or similar)
        # Sometime Dacon encodes it
        try:
            s_decoded = s.encode().decode('unicode-escape')
        except:
            s_decoded = s
            
        if "SVR" in s_decoded or "StandardScaler" in s_decoded or "fit_transform" in s_decoded:
            matches.append(html.unescape(s_decoded))
            
    print(f"Found {len(matches)} string literals containing keywords.")
    
    for idx, match in enumerate(matches):
        print(f"\n=== Match {idx+1} (first 1000 chars) ===")
        print(match[:2000])
        print("=======================================")
        
        # Write to file
        with open(f"/Users/admin/Documents/dev_src/stress_index/scratch/svr_code_extracted_{idx+1}.py", "w", encoding="utf-8") as f:
            f.write(match)
            
else:
    print("Could not find window.__NUXT__")
