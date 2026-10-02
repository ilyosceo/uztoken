import re

p = 'uztoken/affixes.py'
s = open(p, encoding='utf-8').read()

# 1) Derivational suffixes must attach to bare stems too -> remove follows restriction
for did in ['deriv_chi', 'deriv_lik', 'deriv_li', 'deriv_siz', 'deriv_la', 'deriv_ish']:
    m = re.search(r'(Affix\("%s".*?)follows=\{[^}]*\}, priority=6(\),\n)' % did, s, re.S)
    if m:
        s = s[:m.start(1)] + m.group(1) + m.group(2) + s[m.end(2):]

# 2) Verb person suffix variants (shaxs qo'shimchalari)
repl = {
    'verb_pers_1sg': ('["man"]', '["man", "im", "ym"]'),
    'verb_pers_2sg': ('["san"]', '["san", "sing", "ng", "ing"]'),
    'verb_pers_1pl': ('["miz"]', '["miz", "imiz", "mizlar"]'),
    'verb_pers_2pl': ('["siz", "sizlar"]', '["siz", "sizlar", "ingiz", "qing", "larsiz"]'),
    'verb_pers_3pl': ('["lar", "dilar"]', '["lar", "dilar", "adi"]'),
}
for aid, (old, new) in repl.items():
    m = re.search(r'Affix\("%s".*?%s' % (aid, re.escape(old)), s, re.S)
    assert m, aid
    s = s[:m.end() - len(old)] + new + s[m.end():]

# 3) Negation variants
s = s.replace('["ma", "mas", "may"]',
              '["ma", "mas", "may", "magan", "maydi", "masin", "maysiz", "maymiz"]')

# 4) Question particle variants
s = s.replace('AffixCategory.QUESTION, ["mi"])',
              'AffixCategory.QUESTION, ["mi", "mimu", "maymi"], priority=9)')

# 5) Voice / multiword verb chunks before Particles section
anchor = '    # Particles\n'
voice = """    # Verb voice / aspectual compound chunks (yordamchi fe'llar qatlam sifatida)
    Affix("voice_dir", "dir", "turtirish nisbati (-dir/-tir/-giz)", "causative", AffixType.DERIVATIONAL, AffixCategory.VOICE, ["dir", "tir", "giz", "qiz", "adir", "ydir", "tdir"], priority=8),
    Affix("voice_il", "il", "o'zlik/o'tish nisbati (-il/-in)", "reflexive/passive", AffixType.DERIVATIONAL, AffixCategory.VOICE, ["il", "in"], priority=7),
    Affix("voice_ish", "ish", "o'zaro nisbati", "reciprocal", AffixType.DERIVATIONAL, AffixCategory.VOICE, ["ish"], priority=6),
    Affix("aux_kelib", "kelib", "yo'l fe'l shakli (-ib kelib)", "converb aux", AffixType.DERIVATIONAL, AffixCategory.VOICE, ["kelib", "ketib", "qo'yib", "qoyib"], priority=6),
    Affix("aux_ola", "ola", "mumkinlik yordamchisi (-ol-)", "potential aux", AffixType.DERIVATIONAL, AffixCategory.VOICE, ["oladigan", "olmaydigan", "olgan", "olmagan", "ola", "oli", "oloq"], priority=7),
    Affix("aux_kora", "ko'ra", "takrorlanish yordamchisi (-ib ko'ra)", "repetitive aux", AffixType.DERIVATIONAL, AffixCategory.VOICE, ["ko'ra", "koray", "bera", "beri", "qila", "qilay", "tura", "yura"], priority=6),

"""
assert anchor in s
s = s.replace(anchor, voice + anchor)

open(p, 'w', encoding='utf-8').write(s)
print('ok')
