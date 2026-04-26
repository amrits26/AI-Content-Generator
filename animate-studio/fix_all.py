import os

def sanitize_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # This forces every line to use 4 spaces and removes any hidden tabs
        sanitized_lines = [line.expandtabs(4).rstrip() + '\n' for line in lines]
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(sanitized_lines)
        print(f"✅ Fixed: {filepath}")
    except Exception as e:
        print(f"❌ Could not fix {filepath}: {e}")

# Target only your python files
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".py") and file != "fix_all.py":
            sanitize_file(os.path.join(root, file))

print("\n🚀 All files standardized to 4-space indents. Try 'python main.py' now.")
