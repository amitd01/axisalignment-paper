"""Prompt set for the scaled run.

36 prompts across six registers. The 135M pilot used five prompts, three of them
alternate-uses variants, which was both underpowered and the friendliest possible
case for a diversity intervention. This set is balanced by design so that
task-family effects can be separated from the operator effect.
"""

PROMPTS = {
    # --- divergent thinking (Guilford alternate-uses family) ---
    "divergent": [
        "List some unusual uses for a brick. One unusual use is to",
        "List some unusual uses for a paperclip. You could use it to",
        "Unusual uses for an old newspaper include using it to",
        "An unexpected thing you can do with a bicycle wheel is to",
        "One surprising use for a glass bottle is to",
        "A clever use for an empty tin can is to",
    ],
    # --- narrative openings ---
    "narrative": [
        "Write the opening line of a short story about a lighthouse keeper who",
        "Begin a story about a woman who finds a letter addressed to someone who",
        "The first sentence of a novel about a failing orchestra: The conductor",
        "Open a story set in a railway station where nobody ever",
        "Start a story about a cartographer who discovers that one island",
        "The opening of a story about two brothers who inherit a",
    ],
    # --- explanatory ---
    "explanatory": [
        "A good way to explain entropy to a child is to say that",
        "The reason old bridges were built with arches is because",
        "Explain why the sky looks red at sunset. The main reason is that",
        "The simplest way to understand compound interest is to imagine",
        "Why do metals conduct electricity? The short answer is that",
        "Explain how vaccines work in one sentence: they",
    ],
    # --- argumentative ---
    "argumentative": [
        "The main argument against the proposal is that it",
        "In my opinion, the best approach to this problem would be to",
        "One strong objection to banning cars from city centres is that",
        "The case for teaching statistics before calculus rests on",
        "A reasonable person might disagree with this because",
        "The weakest part of that reasoning is that it assumes",
    ],
    # --- problem solving ---
    "problem_solving": [
        "An unconventional solution to urban traffic congestion would be to",
        "To reduce food waste in a restaurant, one approach is to",
        "A cheap way to cool a building without air conditioning is to",
        "If a small library wanted more visitors, it could try",
        "One way to make a long commute less tiring is to",
        "To stop a leaking roof until a repair is possible, you could",
    ],
    # --- descriptive ---
    "descriptive": [
        "Describe a colour to someone who has never been able to see. You might say",
        "Describe the smell of rain on hot pavement. It is like",
        "Describe the sound of a crowded market without naming any instrument. It",
        "Describe what old paper feels like between your fingers. It",
        "Describe a city at four in the morning. The streets",
        "Describe the taste of something bitter to someone who has only tasted sweet things. It",
    ],
}

FLAT = [(fam, p) for fam, ps in PROMPTS.items() for p in ps]

if __name__ == "__main__":
    print(f"{len(FLAT)} prompts across {len(PROMPTS)} families")
    for fam, ps in PROMPTS.items():
        print(f"  {fam:<16} {len(ps)}")
