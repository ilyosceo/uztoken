import sys
sys.path.insert(0, '.')
from uztoken.affixes import get_suffix_variants

variants = get_suffix_variants()
# What suffixes start with 'y'?
y_variants = [v for v in variants if v['variant'].startswith('y')]
for v in y_variants:
    print(f"  '{v['variant']}' -> {v['affix'].id} ({v['affix'].meaning_uz})")

# Word: lashtirolmayotganliklaringizdandir
# After removing 'dir': lashtirolmayotganliklaringizdan
# After removing 'dan': lashtirolmayotganliklaringiz
# After removing 'ingiz': lashtirolmayotganliklar
# After removing 'lar': lashtirolmayotganlik
# After removing 'lik': lashtirolmayotgan
# After removing 'gan': lashtirolmayot   <-- problem! 'yotgan' needs to match but 'gan' matches first
# We need 'yotgan' variant that's longer than 'gan'

# Let's check what 'mayotgan' or 'yotgan' looks like
may_variants = [v for v in variants if 'yot' in v['variant'] or 'mayot' in v['variant']]
for v in may_variants:
    print(f"  '{v['variant']}' -> {v['affix'].id} ({v['affix'].meaning_uz})")

print("\n--- Check for 'may' and 'ma' ---")
ma_variants = [v for v in variants if v['variant'] in ('may', 'ma', 'mas')]
for v in ma_variants:
    print(f"  '{v['variant']}' -> {v['affix'].id} ({v['affix'].meaning_uz})")
