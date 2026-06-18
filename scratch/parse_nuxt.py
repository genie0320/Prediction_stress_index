import re
import js2py
import json

html_path = "/Users/admin/Documents/dev_src/stress_index/scratch/post_12854.html"

with open(html_path, "r", encoding="utf-8") as f:
    content = f.read()

# Let's search for __NUXT__ in script tags
nuxt_match = re.search(r'<script>\s*window\.__NUXT__\s*=(.*?);\s*</script>', content, re.DOTALL)

if nuxt_match:
    print("Found window.__NUXT__ script tag!")
    nuxt_js = nuxt_match.group(1)
    
    # We can evaluate the JS to get the Python dictionary
    # Using js2py to safely execute the JS assignment
    try:
        context = js2py.EvalJs()
        context.execute(f"var nuxt_data = {nuxt_js};")
        data = context.nuxt_data.to_dict()
        
        # Save to file
        with open("/Users/admin/Documents/dev_src/stress_index/scratch/post_12854_nuxt.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("Successfully evaluated and saved window.__NUXT__ to post_12854_nuxt.json")
        
        # Let's search for python code or text in the nuxt data
        def find_code(obj):
            results = []
            if isinstance(obj, str):
                if "import " in obj or "SVR" in obj or "StandardScaler" in obj:
                    results.append(obj)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    results.extend(find_code(v))
            elif isinstance(obj, list):
                for item in obj:
                    results.extend(find_code(item))
            return results
            
            
        res = find_code(data)
        print(f"Found {len(res)} text blocks containing code/keywords.")
        for idx, block in enumerate(res):
            print(f"\n--- Match {idx+1} (first 1000 chars) ---")
            print(block[:1500])
            print("---------------------------------------")
            
            # Save block to separate file
            with open(f"/Users/admin/Documents/dev_src/stress_index/scratch/extracted_code_{idx+1}.py", "w", encoding="utf-8") as f:
                f.write(block)
                
    except Exception as e:
        print("Error evaluating JS:", e)
else:
    print("Could not find window.__NUXT__ in HTML. Searching for any script contents containing SVR...")
    # fallback: scan all script tags
    script_contents = re.findall(r'<script>(.*?)</script>', content, re.DOTALL)
    for idx, script in enumerate(script_contents):
        if "SVR" in script:
            print(f"Script {idx+1} contains SVR, size: {len(script)}")
            print(script[:1000])
