import json
import re
import html

html_path = "/Users/admin/Documents/dev_src/stress_index/scratch/post_12854.html"

with open(html_path, "r", encoding="utf-8") as f:
    content = f.read()

# Let's search for NEXT_DATA script tag
script_match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', content)

if script_match:
    print("Found __NEXT_DATA__ script tag!")
    data = json.loads(script_match.group(1))
    
    # Save parsed JSON to inspect
    json_out_path = "/Users/admin/Documents/dev_src/stress_index/scratch/post_12854_next_data.json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved next data JSON to {json_out_path}")
    
    # Let's search for fields like 'code' or 'description' or 'content' in the JSON
    # We will recursively walk the JSON and look for any long text that has markdown or python code
    def find_content(obj):
        results = []
        if isinstance(obj, str):
            if "import " in obj or "SVR" in obj or "SVR(" in obj or "StandardScaler" in obj:
                results.append(obj)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                results.extend(find_content(v))
        elif isinstance(obj, list):
            for item in obj:
                results.extend(find_content(item))
        return results
        
    res = find_content(data)
    print(f"Found {len(res)} text blocks containing keywords.")
    
    # Save the text blocks to a markdown file
    out_md = "/Users/admin/Documents/dev_src/stress_index/scratch/post_12854_content.md"
    with open(out_md, "w", encoding="utf-8") as f:
        for idx, block in enumerate(res):
            f.write(f"\n# Match {idx+1}\n")
            f.write(block)
            f.write("\n" + "="*80 + "\n")
    print(f"Saved extracted contents to {out_md}")
    
    # Print the first few hundred characters of each match to console
    for idx, block in enumerate(res):
        print(f"\n--- Match {idx+1} (first 500 chars) ---")
        print(block[:1000])
        print("---------------------------------------")
else:
    print("Could not find __NEXT_DATA__ in HTML file. Scanning for other patterns...")
    # fallback to searching code in text
    code_blocks = re.findall(r'<pre>(.*?)</pre>', content, re.DOTALL)
    print(f"Found {len(code_blocks)} pre blocks.")
    for idx, block in enumerate(code_blocks):
        print(f"Block {idx+1}: {block[:200]}")
