import urllib.request
import re
import html

# The list of codeshare IDs we found
post_ids = ['12875', '12854', '12616', '12876', '12869', '12507', '12529', '12877']

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

for pid in post_ids:
    url = f"https://dacon.io/competitions/official/236526/codeshare/{pid}"
    print(f"Scanning post {pid} at {url}...")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            content = response.read().decode('utf-8')
            
        # Try to find the title of the post in HTML
        # Usually it is inside <title> or <h1/h2> tags
        title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
        if title_match:
            title = html.unescape(title_match.group(1))
            print(f"  Title: {title}")
            
            if "SVR" in title or "No free lunch" in title or "1위" in title:
                print(f"\n>>> FOUND TARGET POST {pid}! Title: {title} <<<")
                
                # Write HTML to file to inspect or parse code blocks
                out_path = f"/Users/admin/Documents/dev_src/stress_index/scratch/post_{pid}.html"
                with open(out_path, "w") as f:
                    f.write(content)
                print(f"Saved post HTML to {out_path}")
                
                # Let's extract code blocks
                # Codes in Dacon are usually inside <pre><code> or in a JS variable.
                # Let's search for python code blocks in HTML
                # Or just print some lines around "SVR" or "fit" or "predict"
                code_blocks = re.findall(r'<pre><code class="python">(.*?)</code></pre>', content, re.DOTALL)
                if not code_blocks:
                    code_blocks = re.findall(r'<code>(.*?)</code>', content, re.DOTALL)
                
                print(f"Found {len(code_blocks)} code blocks.")
                for idx, block in enumerate(code_blocks):
                    clean_block = html.unescape(block)
                    if "SVR" in clean_block or "import" in clean_block or "train" in clean_block:
                        print(f"\n--- Code Block {idx+1} ---")
                        # Print first 2000 chars of code
                        print(clean_block[:2000])
                        print("-----------------------")
    except Exception as e:
        print(f"  Error scanning post {pid}: {e}")
