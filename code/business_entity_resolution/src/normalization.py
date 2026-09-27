import re
import unicodedata

try:
    from unidecode import unidecode
    _HAS_UNIDECODE = True
except ImportError:
    _HAS_UNIDECODE = False

# Legal Suffix Canonicalization Dictionary
LEGAL_SUFFIX_MAP = {
    r'\bpvt\.?\b': 'private',
    r'\bltd\.?\b': 'limited',
    r'\binc\.?\b': 'incorporated',
    r'\bcorp\.?\b': 'corporation',
    r'\bco\.?\b': 'company',
    r'\bllp\.?\b': 'limited liability partnership',
    r'\bllc\.?\b': 'limited liability company',
}

# Address Abbreviation Canonicalization Dictionary
ADDRESS_ABBR_MAP = {
    r'\brd\.?\b': 'road',
    r'\bst\.?\b': 'street',
    r'\bave\.?\b': 'avenue',
    r'\bblvd\.?\b': 'boulevard',
    r'\bopp\.?\b': 'opposite',
    r'\bnr\.?\b': 'near',
    r'\bbldg\.?\b': 'building',
    r'\bflr\.?\b': 'floor',
    r'\bapt\.?\b': 'apartment',
    r'\bno\.?\b': 'number',
}

def normalize_text(text: str) -> str:
    """
    Standardizes business names and addresses:
    - Normalizes Unicode (NFKD) to strip diacritics / non-ASCII noise
    - Lowercases text and expands legal suffixes & address abbreviations
    - Strips punctuation while keeping spaces and numbers
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Unicode -> Latin. Use unidecode to TRANSLITERATE non-Latin scripts
    # (Devanagari/Tamil business names romanize to approximate Latin, e.g.
    # 'ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்' -> 'raaj innnvesttmenntts') so they can match the
    # English reference via char n-grams. The old NFKD+ASCII-ignore path DELETED
    # all non-Latin characters, turning ~half of the Indian names into empty
    # strings and making them unmatchable.
    if _HAS_UNIDECODE:
        text = unidecode(text)
    else:
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower()
    
    # 2. Canonicalize Legal Suffixes
    for pattern, repl in LEGAL_SUFFIX_MAP.items():
        text = re.sub(pattern, repl, text)
        
    # 3. Canonicalize Address Terminology
    for pattern, repl in ADDRESS_ABBR_MAP.items():
        text = re.sub(pattern, repl, text)
        
    # 4. Remove special characters except alphanumeric and whitespace
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_pin_code(text: str) -> str:
    """Extracts 5 or 6 digit PIN/ZIP codes if present."""
    if not isinstance(text, str):
        return ""
    match = re.search(r'\b\d{5,6}\b', text)
    return match.group(0) if match else ""

def extract_door_numbers(text: str) -> set:
    """Extracts numerical building/door numbers from an address string."""
    if not isinstance(text, str):
        return set()
    numbers = re.findall(r'\b\d+\b', text)
    return set(numbers)
