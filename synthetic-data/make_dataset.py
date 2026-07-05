# -*- coding: utf-8 -*-
"""
Synthetic Bengali Hallucination-Detection dataset generator.
Schema matches competition train.csv / dataset samples.json:
    context, prompt_bn, response_bn, label   (1 = faithful, 0 = hallucinated)

Design principle: label correctness is *guaranteed* wherever possible.
  A. Context-grounded (intrinsic): the passage we write IS the ground truth,
     so faithful/hallucinated labels are correct by construction.
  B. Computable math/logic: the correct answer is computed in code.
  C. Curated closed-book banks (idioms, word meanings, spelling, general
     knowledge): high-confidence facts; hallucinated = deliberate corruption
     of a known-correct fact, so both labels are correct as long as the seed
     fact is correct.

Reproducible: fixed SEED. No external data / network required.
"""
import csv, json, random, os

SEED = 20260705
random.seed(SEED)

BN = str.maketrans('0123456789', '০১২৩৪৫৬৭৮৯')
def bn(x):
    return str(x).translate(BN)

NULL = '[NULL]'
rows = []  # list of dict(context, prompt_bn, response_bn, label)

def add(ctx, q, resp, label):
    rows.append({'context': ctx if ctx else NULL,
                 'prompt_bn': q.strip(),
                 'response_bn': str(resp).strip(),
                 'label': int(label)})

# ------------------------------------------------------------------ #
#  Name / entity pools (invented-but-plausible; only used in CONTEXT   #
#  rows where the passage itself defines truth, so realism is fine).    #
# ------------------------------------------------------------------ #
FIRST = ['আব্দুল', 'মেহেদী', 'রফিকুল', 'শামসুর', 'জাহানারা', 'সেলিনা', 'নুরুল', 'তারেক',
         'কামরুল', 'ফরিদা', 'আনিসুল', 'মাহবুব', 'সুফিয়া', 'বেগম', 'হুমায়ুন', 'জসীম',
         'দিলীপ', 'অমিয়', 'সুকান্ত', 'বিভূতি', 'প্রমথ', 'অতুল', 'নীহার', 'শরৎ']
LAST = ['হাসান', 'রহমান', 'ইসলাম', 'আহমেদ', 'চৌধুরী', 'হক', 'কবির', 'মজুমদার', 'সরকার',
        'বন্দ্যোপাধ্যায়', 'চট্টোপাধ্যায়', 'মুখোপাধ্যায়', 'দাশগুপ্ত', 'সেন', 'বসু', 'ঘোষ',
        'ভট্টাচার্য', 'রায়', 'তালুকদার', 'খান']
PLACES = ['ঢাকা', 'চট্টগ্রাম', 'ময়মনসিংহ', 'সিলেট', 'রাজশাহী', 'খুলনা', 'বরিশাল', 'রংপুর',
          'কুমিল্লা', 'যশোর', 'ফরিদপুর', 'পাবনা', 'নোয়াখালী', 'দিনাজপুর', 'বগুড়া', 'টাঙ্গাইল',
          'নদীয়া', 'হুগলি', 'বর্ধমান', 'মুর্শিদাবাদ', 'বাঁকুড়া', 'পুরুলিয়া', 'মেদিনীপুর']
FIELDS = ['কবি', 'ঔপন্যাসিক', 'বিজ্ঞানী', 'চিত্রশিল্পী', 'রাজনীতিবিদ', 'শিক্ষাবিদ',
          'সংগীতশিল্পী', 'চলচ্চিত্র পরিচালক', 'সাংবাদিক', 'ভাষাবিদ', 'অর্থনীতিবিদ', 'সমাজসেবক']
WORKS = ['সন্ধ্যাতারা', 'রক্তকরবীর মাঠ', 'নদীর ওপারে', 'অগ্নিপথ', 'মেঘবালিকা', 'শেষ প্রহর',
         'কুয়াশার রেখা', 'বালুচরের গান', 'নিঃশব্দ যাত্রা', 'পাথরের ফুল', 'দিগন্তের ডাক',
         'সোনালি ধান', 'বৃষ্টির শহর', 'অন্ধকারের উৎস', 'জলছবি', 'মৃত্তিকার ঋণ']
AWARDS = ['একুশে পদক', 'বাংলা একাডেমি পুরস্কার', 'স্বাধীনতা পুরস্কার', 'রবীন্দ্র পুরস্কার',
          'সাহিত্য অকাদেমি পুরস্কার', 'জাতীয় চলচ্চিত্র পুরস্কার']

def pick(pool, avoid=None):
    while True:
        x = random.choice(pool)
        if x != avoid:
            return x

def perturb_year(y, span=40):
    while True:
        d = random.randint(-span, span)
        if d != 0 and 1000 < y + d < 2026:
            return y + d

# ------------------------------------------------------------------ #
#  A. CONTEXT-GROUNDED (intrinsic) — person passages                   #
# ------------------------------------------------------------------ #
def gen_person():
    name = f'{pick(FIRST)} {pick(LAST)}'
    by = random.randint(1850, 1990)
    bp = pick(PLACES)
    field = pick(FIELDS)
    work = pick(WORKS)
    ay = random.randint(by + 30, min(by + 70, 2024))
    award = pick(AWARDS)
    dy = random.randint(ay, min(ay + 20, 2025))
    ctx = (f'{name} ({bn(by)} সালে {bp} অঞ্চলে জন্মগ্রহণ করেন) একজন প্রখ্যাত {field} ছিলেন। '
           f'তাঁর অন্যতম সৃষ্টি "{work}"। {bn(ay)} সালে তাঁকে {award}ে ভূষিত করা হয়। '
           f'{bn(dy)} সালে তিনি মৃত্যুবরণ করেন।')
    # (question, correct answer, wrong-answer generator)
    slots = [
        (f'{name} কত সালে জন্মগ্রহণ করেন?', bn(by), lambda: bn(perturb_year(by))),
        (f'{name} কোথায় জন্মগ্রহণ করেন?', bp, lambda: pick(PLACES, bp)),
        (f'{name} পেশায় কী ছিলেন?', field, lambda: pick(FIELDS, field)),
        (f'{name}-এর অন্যতম সৃষ্টি কোনটি?', work, lambda: pick(WORKS, work)),
        (f'{name} কত সালে {award} লাভ করেন?', bn(ay), lambda: bn(perturb_year(ay, 20))),
        (f'{name} কত সালে মৃত্যুবরণ করেন?', bn(dy), lambda: bn(perturb_year(dy, 20))),
    ]
    return ctx, slots

# districts / geography passages
REGIONS = ['উত্তরবঙ্গ', 'দক্ষিণবঙ্গ', 'পূর্বাঞ্চল', 'পশ্চিমাঞ্চল', 'মধ্যাঞ্চল']
def gen_place():
    name = pick(PLACES) + ' জেলা'
    area = random.randint(1200, 8000)
    pop = random.randint(8, 60) * 100000
    yr = random.randint(1790, 1984)
    height = random.randint(120, 900)
    region = pick(REGIONS)
    ctx = (f'{name} বাংলাদেশের {region}ে অবস্থিত। এর মোট আয়তন {bn(area)} বর্গকিলোমিটার এবং '
           f'জনসংখ্যা প্রায় {bn(pop)}। জেলাটি {bn(yr)} সালে প্রশাসনিকভাবে প্রতিষ্ঠিত হয়। '
           f'এই জেলার সর্বোচ্চ পাহাড়ের উচ্চতা {bn(height)} মিটার।')
    slots = [
        (f'{name}-এর আয়তন কত?', f'{bn(area)} বর্গকিলোমিটার',
         lambda: f'{bn(area + random.choice([-1,1]) * random.randint(200, 2000))} বর্গকিলোমিটার'),
        (f'{name} কত সালে প্রতিষ্ঠিত হয়?', bn(yr), lambda: bn(perturb_year(yr, 30))),
        (f'{name}-এর সর্বোচ্চ পাহাড়ের উচ্চতা কত?', f'{bn(height)} মিটার',
         lambda: f'{bn(height + random.choice([-1,1]) * random.randint(30, 300))} মিটার'),
        (f'{name} কোন অঞ্চলে অবস্থিত?', region, lambda: pick(REGIONS, region)),
    ]
    return ctx, slots

# rivers
RIVERS = ['পদ্মা', 'মেঘনা', 'যমুনা', 'তিস্তা', 'করতোয়া', 'সুরমা', 'কুশিয়ারা', 'ধলেশ্বরী',
          'গড়াই', 'মধুমতি', 'বুড়িগঙ্গা', 'শীতলক্ষ্যা', 'কর্ণফুলী', 'সাঙ্গু', 'মাতামুহুরী']
def gen_river():
    name = pick(RIVERS) + ' নদী'
    length = random.randint(80, 700)
    src = pick(PLACES); mouth = pick(PLACES, src)
    ctx = (f'{name} বাংলাদেশের একটি গুরুত্বপূর্ণ নদী। এর দৈর্ঘ্য প্রায় {bn(length)} কিলোমিটার। '
           f'নদীটি {src} অঞ্চল থেকে উৎপন্ন হয়ে {mouth} অঞ্চলে প্রবাহিত হয়েছে।')
    slots = [
        (f'{name}-এর দৈর্ঘ্য কত?', f'{bn(length)} কিলোমিটার',
         lambda: f'{bn(length + random.choice([-1,1]) * random.randint(20, 250))} কিলোমিটার'),
        (f'{name} কোন অঞ্চল থেকে উৎপন্ন হয়েছে?', src, lambda: pick(PLACES, src)),
    ]
    return ctx, slots

# software / organisation
SOFT = ['অভ্র', 'বিজয়', 'নিকশ', 'ল্যাবেঙ্গল', 'পুঠি', 'বর্ণ', 'শব্দকল্প']
LANGS = ['ভিজুয়াল বেসিক', 'সি++', 'জাভা', 'পাইথন', 'ডেলফি', 'সি শার্প']
def gen_soft():
    name = pick(SOFT) + ' সফটওয়্যার'
    yr = random.randint(1998, 2018)
    founder = f'{pick(FIRST)} {pick(LAST)}'
    lang = pick(LANGS)
    ctx = (f'{name} {bn(yr)} সালে বাজারে আসে। এটি প্রথম তৈরি করেন {founder}। '
           f'সফটওয়্যারটি {lang} প্রোগ্রামিং ভাষায় লেখা হয়েছিল।')
    slots = [
        (f'{name} কত সালে বাজারে আসে?', bn(yr), lambda: bn(perturb_year(yr, 15))),
        (f'{name} কে তৈরি করেন?', founder, lambda: f'{pick(FIRST)} {pick(LAST)}'),
        (f'{name} কোন প্রোগ্রামিং ভাষায় লেখা হয়েছিল?', lang, lambda: pick(LANGS, lang)),
    ]
    return ctx, slots

CONTEXT_GENS = [gen_person, gen_place, gen_river, gen_soft]

def make_context_row(label):
    """Return exactly one context-grounded row with the requested label."""
    for _ in range(30):
        ctx, slots = pick(CONTEXT_GENS)()
        q, correct, wrong_fn = random.choice(slots)
        if label == 1:
            return (ctx, q, correct, 1)
        htype = random.random()
        if htype < 0.7:
            cand = wrong_fn()                            # value/entity swap
        elif htype < 0.85:
            cand = f'{correct}, যা {bn(random.randint(2,9))}টি আন্তর্জাতিক পুরস্কার অর্জন করে'
        else:                                            # unsupported: answer from another slot
            other = random.choice([s for s in slots if s[0] != q])
            cand = other[1]
        if str(cand) != str(correct):
            return (ctx, q, cand, 0)
    return (ctx, q, wrong_fn(), 0)

def emit_context_rows(n_target):
    for _ in range(n_target):
        ctx, q, r, l = make_context_row(1 if random.random() < 0.5 else 0)
        add(ctx, q, r, l)

# ------------------------------------------------------------------ #
#  B. COMPUTABLE MATH / LOGIC (closed-book, label verified in code)    #
# ------------------------------------------------------------------ #
def wrong_num(correct, span=9):
    while True:
        d = random.randint(-span, span)
        if d != 0 and correct + d >= 0:
            return correct + d

ITEMS = ['কলম', 'বই', 'আম', 'চেয়ার', 'পেন্সিল', 'খাতা', 'ডিম', 'মার্বেল', 'টিকিট', 'গাছ']
def gen_math_shop():
    a = random.randint(30, 950); b = random.randint(1, a)
    ans = a - b
    it = pick(ITEMS)
    q = f'একটি দোকানে {bn(a)}টি {it} ছিল। {bn(b)}টি বিক্রি হয়ে গেলে কতটি {it} অবশিষ্ট থাকে?'
    return q, bn(ans), bn(wrong_num(ans))

def gen_math_sum():
    a = random.randint(15, 4000); b = random.randint(15, 4000)
    ans = a + b
    q = f'{bn(a)} ও {bn(b)} এর যোগফল কত?'
    return q, bn(ans), bn(wrong_num(ans, 30))

def gen_math_mul():
    a = random.randint(3, 99); b = random.randint(3, 99)
    ans = a * b
    q = f'{bn(a)} ও {bn(b)} এর গুণফল কত?'
    return q, bn(ans), bn(wrong_num(ans, max(3, ans // 10)))

def gen_math_avg():
    xs = [random.randint(10, 300) for _ in range(3)]
    while sum(xs) % 3 != 0:
        xs = [random.randint(10, 300) for _ in range(3)]
    ans = sum(xs) // 3
    q = f'{bn(xs[0])}, {bn(xs[1])} ও {bn(xs[2])} সংখ্যা তিনটির গড় কত?'
    return q, bn(ans), bn(wrong_num(ans))

def gen_math_pct():
    base = random.choice([200, 400, 500, 800, 1000, 1200])
    p = random.choice([5, 10, 12, 15, 20, 25, 40, 50])
    ans = base * p // 100
    q = f'{bn(base)} টাকার {bn(p)}% কত টাকা?'
    return q, f'{bn(ans)} টাকা', f'{bn(wrong_num(ans, max(2, ans//5)))} টাকা'

def gen_math_twodigit():
    # units digit is k more than tens; number = m*(digit sum)+c  -> pick solvable
    for _ in range(50):
        t = random.randint(1, 6); k = random.randint(1, 3)
        u = t + k
        if u > 9: continue
        num = 10 * t + u
        s = t + u
        # find a linear relation num = m*s + c that holds, present it
        m = random.choice([2, 3]); c = num - m * s
        if c <= 0: continue
        q = (f'দুই অঙ্কবিশিষ্ট একটি সংখ্যার এককের অঙ্ক দশকের অঙ্ক অপেক্ষা {bn(k)} বেশি। '
             f'সংখ্যাটি এর অঙ্কদ্বয়ের সমষ্টির {bn(m)} গুণ অপেক্ষা {bn(c)} বেশি। সংখ্যাটি কত?')
        return q, bn(num), bn(wrong_num(num))
    return None

def gen_math_prime_prob():
    lo = random.choice([10, 20, 30, 40]); hi = lo + 10
    nums = list(range(lo, hi + 1))
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True
    fav = sum(1 for n in nums if is_prime(n) or n % 5 == 0)
    tot = len(nums)
    from math import gcd
    g = gcd(fav, tot)
    ans = f'{bn(fav // g)}/{bn(tot // g)}'
    wf = fav + random.choice([-1, 1])
    wrong = f'{bn(max(1, wf) // max(1, gcd(max(1, wf), tot)))}/{bn(tot // max(1, gcd(max(1, wf), tot)))}'
    q = (f'{bn(lo)} থেকে {bn(hi)} পর্যন্ত সংখ্যা থেকে যেকোনো একটিকে ইচ্ছেমতো নিলে সেটি '
         f'মৌলিক অথবা ৫-এর গুণিতক হওয়ার সম্ভাবনা কত?')
    return q, ans, (wrong if wrong != ans else bn(fav) + '/' + bn(tot + 1))

MATH_GENS = [gen_math_shop, gen_math_sum, gen_math_mul, gen_math_avg, gen_math_pct,
             gen_math_twodigit, gen_math_prime_prob]

def emit_math_rows(n_target):
    made = 0; guard = 0
    while made < n_target and guard < n_target * 20:
        guard += 1
        out = pick(MATH_GENS)()
        if out is None:
            continue
        q, correct, wrong = out
        if str(correct) == str(wrong):
            continue
        if random.random() < 0.5:
            add(NULL, q, correct, 1)
        else:
            add(NULL, q, wrong, 0)
        made += 1

# ------------------------------------------------------------------ #
#  C. CURATED CLOSED-BOOK BANKS                                        #
# ------------------------------------------------------------------ #
IDIOMS = [
    ('অহি-নকুল সম্পর্ক', 'চিরশত্রুতা'), ('গোঁফ খেজুরে', 'নিতান্ত অলস'),
    ('ঢাকের কাঠি', 'মোসাহেব বা চাটুকার'), ('উড়নচণ্ডী', 'অমিতব্যয়ী'),
    ('কূপমণ্ডূক', 'সংকীর্ণমনা বা সীমিত জ্ঞানের মানুষ'), ('ধরি মাছ না ছুঁই পানি', 'কৌশলে কার্যোদ্ধার'),
    ('আদায় কাঁচকলায়', 'তিক্ত সম্পর্ক'), ('চোখের বালি', 'চক্ষুশূল বা অপ্রিয় ব্যক্তি'),
    ('গোবর গণেশ', 'মূর্খ'), ('ভিজে বেড়াল', 'কপটচারী'), ('তামার বিষ', 'অর্থের কুপ্রভাব'),
    ('বকধার্মিক', 'ভণ্ড ধার্মিক'), ('রাবণের চিতা', 'চিরস্থায়ী অশান্তি'), ('শাঁখের করাত', 'উভয় সংকট'),
    ('আটকপালে', 'হতভাগ্য'), ('ইঁদুর কপালে', 'মন্দভাগ্য'), ('কান পাতলা', 'সহজে বিশ্বাসপ্রবণ'),
    ('গদাই লস্করি চাল', 'অত্যন্ত ধীর গতি'), ('ননীর পুতুল', 'শ্রমকাতর ব্যক্তি'),
    ('হাতের পাঁচ', 'শেষ সম্বল'), ('কেঁচে গণ্ডূষ', 'পুনরায় আরম্ভ'), ('ব্যাঙের সর্দি', 'অসম্ভব ঘটনা'),
    ('বিড়াল তপস্বী', 'ভণ্ড সাধু'), ('সোনার পাথরবাটি', 'অসম্ভব বস্তু'), ('অগ্নিশর্মা', 'অত্যন্ত ক্রুদ্ধ'),
    ('খয়ের খাঁ', 'চাটুকার'), ('দুধের মাছি', 'সুসময়ের বন্ধু'), ('লেজে গোবরে', 'বিশৃঙ্খলা'),
    ('ঘোড়ার ডিম', 'অবাস্তব বা কিছুই না'), ('চাঁদের হাট', 'আনন্দের প্রাচুর্য'),
    ('আক্কেল সেলামি', 'নির্বুদ্ধিতার দণ্ড'), ('গাছে কাঁঠাল গোঁফে তেল', 'আগেভাগে প্রস্তুতি'),
    ('তুলসী বনের বাঘ', 'ভণ্ড'), ('এলোপাতাড়ি', 'বিশৃঙ্খলভাবে'), ('কাঠের পুতুল', 'নির্জীব বা অসাড়'),
    ('ভরাডুবি', 'সর্বনাশ'),
]
def emit_idioms():
    for idiom, mean in IDIOMS:
        add(NULL, f'"{idiom}" বাগধারাটির অর্থ কী?', mean, 1)
        add(NULL, f'"{idiom}" এর ভাবার্থ কী?', mean, 1)
        add(NULL, f'"{idiom}" বাগধারাটির অর্থ কী?', pick([m for _, m in IDIOMS], mean), 0)
        add(NULL, f'"{idiom}" এর ভাবার্থ কী?', pick([m for _, m in IDIOMS], mean), 0)

WORDS = [
    ('competent', 'যোগ্য'), ('flat', 'সমতল'), ('rigid', 'অনমনীয়'), ('vivid', 'প্রাণবন্ত'),
    ('brief', 'সংক্ষিপ্ত'), ('ancient', 'প্রাচীন'), ('humble', 'বিনয়ী'), ('generous', 'উদার'),
    ('fragile', 'ভঙ্গুর'), ('rapid', 'দ্রুত'), ('vacant', 'খালি'), ('genuine', 'খাঁটি'),
    ('hostile', 'বৈরী'), ('obscure', 'অস্পষ্ট'), ('novel', 'অভিনব'), ('mandatory', 'বাধ্যতামূলক'),
    ('feasible', 'সম্ভবপর'), ('diligent', 'পরিশ্রমী'), ('scarce', 'দুর্লভ'), ('vague', 'অস্পষ্ট'),
    ('reluctant', 'অনিচ্ছুক'), ('abundant', 'প্রচুর'), ('candid', 'অকপট'), ('feeble', 'দুর্বল'),
    ('prudent', 'বিচক্ষণ'), ('sturdy', 'মজবুত'), ('serene', 'প্রশান্ত'), ('turbulent', 'উত্তাল'),
]
def emit_words():
    for w, m in WORDS:
        add(NULL, f'ইংরেজি "{w}" শব্দের বাংলা অর্থ কী?', m, 1)
        add(NULL, f'"{w}" শব্দের অর্থ কী?', m, 1)
        add(NULL, f'ইংরেজি "{w}" শব্দের বাংলা অর্থ কী?', pick([x for _, x in WORDS], m), 0)
        add(NULL, f'"{w}" শব্দের অর্থ কী?', pick([x for _, x in WORDS], m), 0)

SPELL = [
    ('মনীষী', ['মনিষী', 'মণিষী', 'মনীষি']), ('শুশ্রূষা', ['শুশ্রুষা', 'সুশ্রূষা', 'শূশ্রূষা']),
    ('মুমূর্ষু', ['মুমুর্ষু', 'মূমূর্ষু', 'মুমূর্ষূ']), ('দ্বন্দ্ব', ['দ্বন্ধ', 'দন্দ্ব', 'দ্বন্দ']),
    ('পিপীলিকা', ['পিপিলিকা', 'পীপিলিকা', 'পিপীলীকা']), ('সমীচীন', ['সমিচীন', 'সমীচিন', 'সমিচিন']),
    ('মূর্ধন্য', ['মুর্ধন্য', 'মূর্ধণ্য', 'মুর্ধণ্য']), ('আকাঙ্ক্ষা', ['আকাংখা', 'আকাঙ্খা', 'আকাঙ্ক্খা']),
    ('বিভীষিকা', ['বিভিষীকা', 'বিভীষিকা'.replace('ভী','ভি'), 'বীভিষিকা']),
    ('স্বায়ত্তশাসন', ['স্বায়ত্বশাসন', 'স্বায়ত্ত্বশাসন', 'সায়ত্তশাসন']),
    ('ভৌগোলিক', ['ভৌগলিক', 'ভৌগোলীক', 'ভৌগলীক']), ('শ্রদ্ধাঞ্জলি', ['শ্রদ্ধাঞ্জলী', 'শ্রধাঞ্জলি', 'স্রদ্ধাঞ্জলি']),
    ('দুর্বিষহ', ['দুর্বিসহ', 'দূর্বিষহ', 'দুর্বীষহ']), ('উচ্ছ্বাস', ['উচ্ছাস', 'উচ্ছ্বাশ', 'উছ্বাস']),
    ('নিরীক্ষণ', ['নীরিক্ষণ', 'নিরীক্ষন', 'নীরীক্ষণ']), ('ব্যতিক্রম', ['ব্যাতিক্রম', 'ব্যতিক্রম'.replace('তি','তী'), 'বেতিক্রম']),
]
def emit_spelling():
    for correct, wrongs in SPELL:
        add(NULL, f'নিচের কোন বানানটি শুদ্ধ — "{correct}" নাকি "{wrongs[0]}"?', correct, 1)
        add(NULL, f'"{correct}" শব্দের শুদ্ধ বানান কি "{wrongs[0]}"?', wrongs[0], 0)
        add(NULL, f'"{correct}" শব্দের একটি ভুল/অশুদ্ধ বানান লিখুন।', pick(wrongs), 1)

# General knowledge — (question, correct, [wrong distractors])
GK = [
    ('আয়তনে পৃথিবীর ক্ষুদ্রতম দেশ কোনটি?', 'ভ্যাটিকান সিটি', ['মালদ্বীপ', 'মোনাকো', 'নাউরু']),
    ('আয়তনে পৃথিবীর বৃহত্তম দেশ কোনটি?', 'রাশিয়া', ['কানাডা', 'চীন', 'যুক্তরাষ্ট্র']),
    ('বাংলাদেশের রাজধানীর নাম কী?', 'ঢাকা', ['চট্টগ্রাম', 'কলকাতা', 'সিলেট']),
    ('সৌরজগতের বৃহত্তম গ্রহ কোনটি?', 'বৃহস্পতি', ['শনি', 'পৃথিবী', 'নেপচুন']),
    ('সূর্যের নিকটতম গ্রহ কোনটি?', 'বুধ', ['শুক্র', 'পৃথিবী', 'মঙ্গল']),
    ('মানবদেহের বৃহত্তম অঙ্গ কোনটি?', 'ত্বক', ['যকৃত', 'হৃৎপিণ্ড', 'ফুসফুস']),
    ('রক্তকে লাল রঙ প্রদানকারী উপাদানে কোন ধাতু থাকে?', 'লৌহ', ['তামা', 'দস্তা', 'ক্যালসিয়াম']),
    ('সালোকসংশ্লেষণ উদ্ভিদকোষের কোন অঙ্গাণুতে ঘটে?', 'ক্লোরোপ্লাস্ট', ['মাইটোকন্ড্রিয়া', 'নিউক্লিয়াস', 'রাইবোজোম']),
    ('পানির রাসায়নিক সংকেত কী?', 'H2O', ['CO2', 'O2', 'H2O2']),
    ('বাংলা ভাষার প্রাচীনতম নিদর্শন কোনটি?', 'চর্যাপদ', ['শ্রীকৃষ্ণকীর্তন', 'মঙ্গলকাব্য', 'পদাবলি']),
    ('"বিদ্রোহী" কবিতাটির রচয়িতা কে?', 'কাজী নজরুল ইসলাম', ['রবীন্দ্রনাথ ঠাকুর', 'জীবনানন্দ দাশ', 'সুকান্ত ভট্টাচার্য']),
    ('"গীতাঞ্জলি" কাব্যগ্রন্থের রচয়িতা কে?', 'রবীন্দ্রনাথ ঠাকুর', ['কাজী নজরুল ইসলাম', 'মাইকেল মধুসূদন দত্ত', 'জসীমউদ্দীন']),
    ('"পথের পাঁচালী" উপন্যাসের লেখক কে?', 'বিভূতিভূষণ বন্দ্যোপাধ্যায়', ['শরৎচন্দ্র চট্টোপাধ্যায়', 'তারাশঙ্কর বন্দ্যোপাধ্যায়', 'মানিক বন্দ্যোপাধ্যায়']),
    ('"অগ্নিবীণা" কাব্যগ্রন্থের রচয়িতা কে?', 'কাজী নজরুল ইসলাম', ['রবীন্দ্রনাথ ঠাকুর', 'জীবনানন্দ দাশ', 'বুদ্ধদেব বসু']),
    ('বাংলাদেশের জাতীয় ফুল কী?', 'শাপলা', ['গোলাপ', 'পদ্ম', 'বেলি']),
    ('বাংলাদেশের জাতীয় পাখি কী?', 'দোয়েল', ['ময়না', 'কোকিল', 'শালিক']),
    ('বাংলাদেশের জাতীয় ফল কী?', 'কাঁঠাল', ['আম', 'জাম', 'লিচু']),
    ('বাংলাদেশের জাতীয় পশু কী?', 'রয়েল বেঙ্গল টাইগার', ['হাতি', 'চিতা', 'সিংহ']),
    ('বাংলাদেশের মহান স্বাধীনতা যুদ্ধ কোন সালে শুরু হয়?', '১৯৭১', ['১৯৫২', '১৯৪৭', '১৯৬৯']),
    ('ভাষা আন্দোলন কোন সালে সংঘটিত হয়?', '১৯৫২', ['১৯৭১', '১৯৪৭', '১৯৬৬']),
    ('বাংলাদেশের স্বাধীনতা দিবস কোন তারিখে পালিত হয়?', '২৬ মার্চ', ['১৬ ডিসেম্বর', '২১ ফেব্রুয়ারি', '১৪ ডিসেম্বর']),
    ('বাংলাদেশের বিজয় দিবস কোন তারিখে পালিত হয়?', '১৬ ডিসেম্বর', ['২৬ মার্চ', '২১ ফেব্রুয়ারি', '১৭ এপ্রিল']),
    ('পৃথিবীর একমাত্র প্রাকৃতিক উপগ্রহের নাম কী?', 'চাঁদ', ['সূর্য', 'শুক্র', 'মঙ্গল']),
    ('মানবদেহে মোট কতটি হাড় থাকে (প্রাপ্তবয়স্ক)?', '২০৬টি', ['২১২টি', '১৯৮টি', '৩০৬টি']),
    ('আলোর গতিবেগ প্রায় কত?', 'সেকেন্ডে ৩ লক্ষ কিলোমিটার', ['সেকেন্ডে ৩ হাজার কিলোমিটার', 'সেকেন্ডে ৩ কোটি কিলোমিটার', 'সেকেন্ডে ৩০ লক্ষ কিলোমিটার']),
    ('DNA-এর পূর্ণরূপ কী?', 'ডিঅক্সিরাইবোনিউক্লিক অ্যাসিড', ['ডাইনাইট্রোঅ্যামাইনো অ্যাসিড', 'রাইবোনিউক্লিক অ্যাসিড', 'ডিঅক্সিরাইবোজ অ্যাসিড']),
    ('বাংলাদেশের দীর্ঘতম নদী কোনটি?', 'মেঘনা', ['পদ্মা', 'যমুনা', 'কর্ণফুলী']),
    ('বাংলা বর্ণমালায় স্বরবর্ণ কয়টি?', '১১টি', ['৭টি', '৯টি', '৩৯টি']),
    ('বাংলা বর্ণমালায় ব্যঞ্জনবর্ণ কয়টি?', '৩৯টি', ['৩৫টি', '১১টি', '৪১টি']),
    ('"সমাস" ভাষাকে কী করে?', 'সংক্ষেপ করে', ['বিস্তৃত করে', 'অলংকৃত করে', 'জটিল করে']),
    ('মুঘল সম্রাট হুমায়ুন বাংলার কোন স্থানের নাম দেন "জান্নাতাবাদ"?', 'গৌড়', ['সোনারগাঁও', 'ঢাকা', 'মুর্শিদাবাদ']),
    ('জাতিসংঘ কোন সালে প্রতিষ্ঠিত হয়?', '১৯৪৫', ['১৯১৯', '১৯৪৮', '১৯৫০']),
    ('মানবদেহের সবচেয়ে শক্তিশালী পেশি কোথায় থাকে?', 'চোয়ালে', ['হাতে', 'পায়ে', 'পিঠে']),
    ('সবচেয়ে হালকা মৌল কোনটি?', 'হাইড্রোজেন', ['হিলিয়াম', 'অক্সিজেন', 'কার্বন']),
    ('বাংলাদেশ কোন সালে জাতিসংঘের সদস্যপদ লাভ করে?', '১৯৭৪', ['১৯৭১', '১৯৭২', '১৯৭৫']),
    ('রংধনুতে মোট কয়টি রঙ থাকে?', '৭টি', ['৫টি', '৬টি', '৯টি']),
    ('একটি বৃত্তের পরিধি ও ব্যাসের অনুপাত কোন ধ্রুবক?', 'পাই (π)', ['e', 'ফাই (φ)', 'ল্যাম্বডা']),
    ('বাংলা সাহিত্যের প্রথম মহিলা কবি কে?', 'চন্দ্রাবতী', ['কামিনী রায়', 'বেগম রোকেয়া', 'স্বর্ণকুমারী দেবী']),
    ('"অর্থনীতি" বিষয়ে প্রথম নোবেল বিজয়ী বাঙালি কে?', 'অমর্ত্য সেন', ['মুহাম্মদ ইউনূস', 'রবীন্দ্রনাথ ঠাকুর', 'জগদীশ চন্দ্র বসু']),
    ('মুহাম্মদ ইউনূস কোন সালে নোবেল শান্তি পুরস্কার লাভ করেন?', '২০০৬', ['১৯৯৮', '২০১০', '২০০০']),
]
def emit_gk():
    for q, correct, wrongs in GK:
        add(NULL, q, correct, 1)
        for w in wrongs:               # all distractors -> hallucinated
            add(NULL, q, w, 0)

# ------------------------------------------------------------------ #
#  BUILD                                                               #
# ------------------------------------------------------------------ #
TARGET = 5000
emit_idioms(); emit_words(); emit_spelling(); emit_gk()   # curated closed-book
emit_math_rows(1900)                                       # computable closed-book

# dedupe closed-book pool
seen = set(); dedup = []
for r in rows:
    key = (r['context'], r['prompt_bn'], r['response_bn'])
    if key not in seen:
        seen.add(key); dedup.append(r)
rows = dedup
null_rows = rows[:]                                         # everything so far is [NULL]
n_null = len(null_rows)
null_ones = sum(1 for r in null_rows if r['label'] == 1)
null_zeros = n_null - null_ones

# fill to TARGET with context rows, forcing per-label counts so the whole
# dataset ends up exactly 50/50 and ~ the target context share.
n_ctx = TARGET - n_null
need_ones = TARGET // 2 - null_ones
need_zeros = TARGET // 2 - null_zeros
# clamp (context filler can only add non-negative counts)
need_ones = max(need_ones, 0); need_zeros = max(need_zeros, 0)
scale = n_ctx / max(need_ones + need_zeros, 1)
need_ones = int(round(need_ones * scale)); need_zeros = n_ctx - need_ones

ctx_seen = set()
def emit_forced_context(n, label):
    made = 0; guard = 0
    while made < n and guard < n * 40:
        guard += 1
        c, q, r, l = make_context_row(label)
        key = (c, q, r)
        if key in ctx_seen:
            continue
        ctx_seen.add(key); add(c, q, r, l); made += 1

emit_forced_context(need_ones, 1)
emit_forced_context(need_zeros, 0)

final = rows[:]
random.shuffle(final)
final = final[:TARGET]

OUT = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(OUT, 'synthetic_train_5000.csv')
json_path = os.path.join(OUT, 'synthetic_train_5000.json')
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=['context', 'prompt_bn', 'response_bn', 'label'])
    w.writeheader(); w.writerows(final)
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(final, f, ensure_ascii=False, indent=1)

# ------------------------------------------------------------------ #
#  report                                                             #
# ------------------------------------------------------------------ #
from collections import Counter
lab = Counter(r['label'] for r in final)
ctxc = Counter('CTX' if r['context'] != NULL else 'NULL' for r in final)
print(f'total rows        : {len(final)}')
print(f'label dist        : {dict(lab)}  (1=faithful, 0=hallucinated)')
print(f'context presence  : {dict(ctxc)}')

print(f'wrote: {csv_path}')
print(f'wrote: {json_path}')
