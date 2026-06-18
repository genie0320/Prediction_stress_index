import urllib.request
import re
import html

post_url = "https://dacon.io/competitions/official/236526/codeshare/12854"
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

print("Fetching fresh Dacon post page...")
req = urllib.request.Request(post_url, headers=headers)
with urllib.request.urlopen(req) as response:
    html_content = response.read().decode('utf-8')

# Search for the presigned URL pattern for md_file.html or md_file.ipynb
# Example: https://dacon.s3.ap-northeast-2.amazonaws.com/codeshare/236526/12854/md_file.html?...
# Note that characters like / might be escaped as \u002F or \/ in the Nuxt JSON state
# Let's find all occurrences of dacon.s3.ap-northeast-2.amazonaws.com/codeshare/236526/12854/md_file
pattern = r'https?:[^\s"\'>]+dacon\.s3\.ap-northeast-2\.amazonaws\.com[^\s"\'>]+codeshare[^\s"\'>]+12854[^\s"\'>]+md_file[^\s"\'>]+'
matches = re.findall(pattern, html_content)

print(f"\nFound {len(matches)} potential presigned URLs:")

# Clean and download
downloaded = False
for idx, match in enumerate(set(matches)):
    # Clean URL (unescape slash and decode HTML entities)
    clean_url = match.replace(r'\u002F', '/').replace(r'\/', '/').replace('&amp;', '&')
    clean_url = clean_url.split('"')[0].split("'")[0].split(')')[0]
    
    print(f"\n[{idx+1}] Cleaned URL: {clean_url}")
    
    # Determine output extension
    ext = "html"
    if "md_file.ipynb" in clean_url:
        ext = "ipynb"
        
    out_path = f"/Users/admin/Documents/dev_src/stress_index/scratch/extracted_md_file_{idx+1}.{ext}"
    print(f"Attempting download to {out_path}...")
    
    try:
        dl_req = urllib.request.Request(clean_url, headers=headers)
        with urllib.request.urlopen(dl_req) as dl_resp:
            file_data = dl_resp.read()
            
        with open(out_path, "wb") as f:
            f.write(file_data)
            
        print(f"  SUCCESS! Saved {len(file_data)} bytes to {out_path}")
        
        # If it is a notebook, let's print cell source codes
        if ext == "ipynb":
            import json
            nb = json.loads(file_data.decode('utf-8'))
            print("\n--- Jupyter Notebook Cells ---")
            cell_idx = 0
            for cell in nb.get('cells', []):
                if cell.get('cell_type') == 'code':
                    cell_idx += 1
                    print(f"\n[Code Cell {cell_idx}]")
                    print("".join(cell.get('source', [])))
            print("------------------------------")
            
        downloaded = True
    except Exception as e:
        print(f"  Failed to download: {e}")

if not downloaded:
    print("Could not download any of the presigned files.")
