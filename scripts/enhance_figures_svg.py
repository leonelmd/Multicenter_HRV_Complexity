import os
import re

scripts_dir = "."
files = [f"generate_figure{i}.py" for i in range(2, 9)]

print("Enhancing figures 2-8 with SVG support...")

for filename in files:
    filepath = os.path.join(scripts_dir, filename)
    if not os.path.exists(filepath):
        print(f"Skipping {filename} (not found)")
        continue
        
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if SVG already added
    if "format='svg'" in content:
        print(f"{filename}: Already supports SVG.")
        continue
    
    # Find plt.savefig call
    # It might span multiple lines, but usually ends with )
    # We look for the line containing `plt.savefig(out_path` or similar
    # And insert the SVG save immediately after.
    
    # Regex to find the savefig call and capture indentation
    # Assuming standard indentation
    match = re.search(r'^(\s*)plt\.savefig\((.+?)\)', content, re.MULTILINE | re.DOTALL)
    if match:
        indent = match.group(1)
        full_call = match.group(0)
        
        # We need to construct the new call
        # We assume 'out_path' is the variable name for the filename
        # But it might be different. Let's inspect the arguments.
        # Actually, most scripts use `out_path`.
        # If not, we can use `filename.replace('.png', '.svg')` on the first argument?
        # But arguments are inside `(.+?)`.
        # Let's check if `out_path` is present in the arguments.
        args = match.group(2)
        
        new_lines = []
        if "out_path" in args:
            svg_save = f"{indent}plt.savefig(out_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')"
        elif "output_file" in args: # Example alternative
             svg_save = f"{indent}plt.savefig(output_file.replace('.png', '.svg'), format='svg', bbox_inches='tight')"
        else:
            # Fallback: Assume first arg is the filename variable/string
            # This is risky. Let's inspect specific files if needed.
            # Step 731 showed all have `savefig`.
            # Let's assume `out_path` based on `generate_figure2.py` (Step 486) and `generate_figure7.py` (Step 636).
            # Both used `out_path`.
            # I will trust `out_path`.
            svg_save = f"{indent}plt.savefig(out_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')"

        # Replace the original call with Original + New
        new_content = content.replace(full_call, f"{full_call}\n{svg_save}")
        
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"{filename}: Updated.")
    else:
        print(f"{filename}: Could not find plt.savefig pattern.")

print("Done.")
