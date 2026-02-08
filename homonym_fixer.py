"""
Context-aware homonym correction using local LLM.
Fixes common speech-to-text errors like their/there/they're.
"""
import re
import subprocess
import json
from typing import Optional, Tuple
from dataclasses import dataclass


# Common homonym groups - words that sound alike
HOMONYM_GROUPS = [
    # Possessives vs contractions
    ("their", "there", "they're"),
    ("your", "you're"),
    ("its", "it's"),
    ("whose", "who's"),

    # To/too/two
    ("to", "too", "two"),

    # Common coding/tech terms
    ("write", "right", "rite"),
    ("read", "red"),
    ("no", "know"),
    ("new", "knew", "gnu"),
    ("for", "four", "fore"),
    ("or", "ore"),
    ("be", "bee"),
    ("by", "bye", "buy"),
    ("in", "inn"),
    ("ad", "add"),
    ("allowed", "aloud"),
    ("brake", "break"),
    ("capital", "capitol"),
    ("cite", "site", "sight"),
    ("complement", "compliment"),
    ("council", "counsel"),
    ("discrete", "discreet"),
    ("effect", "affect"),
    ("ensure", "insure"),
    ("accept", "except"),
    ("than", "then"),
    ("weather", "whether"),
    ("were", "where", "we're"),
    ("which", "witch"),
    ("would", "wood"),
    ("passed", "past"),
    ("peace", "piece"),
    ("plain", "plane"),
    ("principal", "principle"),
    ("stationary", "stationery"),
    ("wait", "weight"),
    ("weak", "week"),
    ("wear", "where", "ware"),
]

# Build lookup: word -> group of alternatives
HOMONYM_LOOKUP = {}
for group in HOMONYM_GROUPS:
    for word in group:
        HOMONYM_LOOKUP[word.lower()] = group


@dataclass
class CorrectionResult:
    original: str
    corrected: str
    changed: bool
    corrections: list[Tuple[str, str]]  # List of (original, corrected) pairs


class HomonymFixer:
    """
    Fixes homonym errors using context-aware LLM post-processing.
    """

    def __init__(
        self,
        model: str = "llama3.2:1b",
        enabled: bool = True,
        timeout: float = 2.0,  # Max seconds to wait for LLM
    ):
        self.model = model
        self.enabled = enabled
        self.timeout = timeout
        self._ollama_available: Optional[bool] = None

    def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        if self._ollama_available is not None:
            return self._ollama_available

        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                timeout=2,
            )
            self._ollama_available = result.returncode == 0
        except Exception:
            self._ollama_available = False

        return self._ollama_available

    def _contains_homonym(self, text: str) -> bool:
        """Check if text contains any potential homonyms."""
        words = re.findall(r'\b\w+\b', text.lower())
        return any(word in HOMONYM_LOOKUP for word in words)

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama API for correction."""
        try:
            result = subprocess.run(
                [
                    "ollama", "run", self.model,
                    "--nowordwrap",
                    prompt,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
        return None

    def fix(self, text: str) -> CorrectionResult:
        """
        Fix homonym errors in text using LLM context analysis.

        Returns CorrectionResult with original, corrected text, and list of changes.
        """
        if not self.enabled or not text:
            return CorrectionResult(text, text, False, [])

        # Quick check - if no homonyms present, skip LLM call
        if not self._contains_homonym(text):
            return CorrectionResult(text, text, False, [])

        # Check Ollama availability
        if not self._check_ollama():
            return CorrectionResult(text, text, False, [])

        # Build prompt for LLM
        prompt = f"""Fix any homonym errors in this dictated text. Only fix clear errors based on context.
Common errors: their/there/they're, your/you're, its/it's, to/too/two, write/right, etc.

IMPORTANT: Return ONLY the corrected text, nothing else. If no corrections needed, return the original text exactly.

Text: {text}

Corrected:"""

        # Call LLM
        corrected = self._call_ollama(prompt)

        if not corrected:
            return CorrectionResult(text, text, False, [])

        # Clean up LLM response
        corrected = corrected.strip()
        # Remove any markdown or quotes the LLM might add
        corrected = corrected.strip('"\'`')

        # Find what changed
        corrections = []
        if corrected.lower() != text.lower():
            # Simple word-by-word comparison
            orig_words = text.split()
            corr_words = corrected.split()

            for i, (o, c) in enumerate(zip(orig_words, corr_words)):
                if o.lower() != c.lower():
                    corrections.append((o, c))

        changed = len(corrections) > 0

        return CorrectionResult(
            original=text,
            corrected=corrected if changed else text,
            changed=changed,
            corrections=corrections,
        )


class RuleBasedFixer:
    """
    Simple rule-based homonym fixer (no LLM needed).
    Uses basic grammar patterns for common cases.
    """

    # Patterns: (regex, replacement, description)
    RULES = [
        # "they're" when followed by verb-ing or adjective
        (r"\btheir\s+(going|coming|doing|being|getting|making|trying|looking)\b",
         r"they're \1", "their -> they're before verb-ing"),

        # "there" for location after "over/out/up/down/in"
        (r"\b(over|out|up|down|in)\s+their\b",
         r"\1 there", "their -> there after direction"),

        # "you're" when followed by adjective or verb-ing
        (r"\byour\s+(going|coming|doing|being|right|wrong|welcome|sure|correct)\b",
         r"you're \1", "your -> you're before common words"),

        # "it's" for "it is" before adjective/adverb
        (r"\bits\s+(a|an|the|not|very|really|quite|so|too|just|also)\b",
         r"it's \1", "its -> it's before article/adverb"),

        # "to" vs "too" - "too" before adjective
        (r"\bto\s+(much|many|late|early|soon|bad|good|hard|easy|fast|slow)\b",
         r"too \1", "to -> too before adjective"),

        # "two" for numbers
        (r"\b(one|1)\s+(?:or|and)\s+to\b",
         r"\1 or two", "to -> two in counting"),
    ]

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._compiled_rules = [
            (re.compile(pattern, re.IGNORECASE), replacement, desc)
            for pattern, replacement, desc in self.RULES
        ]

    def fix(self, text: str) -> CorrectionResult:
        """Apply rule-based corrections."""
        if not self.enabled or not text:
            return CorrectionResult(text, text, False, [])

        corrected = text
        corrections = []

        for pattern, replacement, desc in self._compiled_rules:
            match = pattern.search(corrected)
            if match:
                original_word = match.group(0)
                corrected = pattern.sub(replacement, corrected)
                corrections.append((original_word, desc))

        return CorrectionResult(
            original=text,
            corrected=corrected,
            changed=len(corrections) > 0,
            corrections=corrections,
        )


def create_fixer(use_llm: bool = True, model: str = "llama3.2:1b") -> HomonymFixer | RuleBasedFixer:
    """
    Create appropriate fixer based on settings.

    Args:
        use_llm: If True, use LLM-based fixer (slower but smarter)
        model: Ollama model to use

    Returns:
        Fixer instance
    """
    if use_llm:
        return HomonymFixer(model=model)
    else:
        return RuleBasedFixer()


def test_fixer():
    """Test the homonym fixer."""
    print("=== Homonym Fixer Test ===\n")

    test_sentences = [
        "Their going to the store.",
        "I think your right about that.",
        "Its a beautiful day.",
        "I have to much work to do.",
        "I want to go over their.",
        "The dog wagged it's tail.",  # Actually wrong - should be "its"
        "I need to items from the store.",
        "Weather or not you agree.",
        "I don't no what to do.",
        "Please right that down.",
    ]

    # Test rule-based first
    print("Rule-based corrections:")
    print("-" * 40)
    rule_fixer = RuleBasedFixer()
    for sentence in test_sentences:
        result = rule_fixer.fix(sentence)
        if result.changed:
            print(f"  '{sentence}'")
            print(f"  -> '{result.corrected}'")
            print()

    # Test LLM-based
    print("\nLLM-based corrections (llama3.2:1b):")
    print("-" * 40)
    llm_fixer = HomonymFixer(model="llama3.2:1b", timeout=5.0)

    if not llm_fixer._check_ollama():
        print("  Ollama not available")
        return

    for sentence in test_sentences[:3]:  # Just test a few to save time
        print(f"  Testing: '{sentence}'")
        result = llm_fixer.fix(sentence)
        if result.changed:
            print(f"  -> '{result.corrected}'")
        else:
            print(f"  -> (no change)")
        print()


if __name__ == "__main__":
    test_fixer()
