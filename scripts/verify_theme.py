import ast, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
src = (BASE / 'app.py').read_text(encoding='utf-8')

# 1. Syntax
ast.parse(src)
print("Syntax: PASS")

# 2. Key functions / dicts exist
required = ['get_theme', 'PALETTE', 'style_fig', 'inject_css',
            'class_donut_fig', 'accuracy_bar_fig', 'hist_fig',
            'gauge_fig', 'contrib_fig', 'compare_fig']
for name in required:
    found = (f'def {name}' in src or f'{name} =' in src)
    print(f"  {'PASS' if found else 'FAIL'}  {name} present")

# 3. Old _fig_base and PLY_TMPL global constant gone
if '_fig_base' in src:
    print("  FAIL  _fig_base() still present")
else:
    print("  PASS  _fig_base() removed")

if 'PLY_TMPL  =' in src or "PLY_TMPL = " in src:
    print("  FAIL  PLY_TMPL still a standalone constant")
else:
    print("  PASS  PLY_TMPL removed as standalone constant")

# 4. style_fig is called on every figure-returning function
for fn in ['class_donut_fig', 'accuracy_bar_fig', 'hist_fig',
           'gauge_fig', 'contrib_fig', 'compare_fig']:
    # Check that 'style_fig' appears after each def
    start = src.find(f'def {fn}')
    end   = src.find('\ndef ', start + 1)
    block = src[start:end] if end != -1 else src[start:]
    ok    = 'style_fig' in block
    print(f"  {'PASS' if ok else 'FAIL'}  style_fig called in {fn}")

# 5. config.toml has light and dark theme settings
cfg = (BASE / '.streamlit' / 'config.toml').read_text()
for key in ['[theme.light]', '[theme.dark]', '#1E6F7C', '#4FB3BF']:
    found = key in cfg
    print(f"  {'PASS' if found else 'FAIL'}  config.toml: {key}")

old_single = '[theme]\nbase' in cfg or ('base                   = ' in cfg and '[theme.light]' not in cfg)
print(f"  {'FAIL' if old_single else 'PASS'}  old single-theme block removed")

print()
print("All checks done.")
