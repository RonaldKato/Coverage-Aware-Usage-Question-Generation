"""
data_generation.py
-------------------
Builds a synthetic, but structurally realistic, review-sentence corpus that
mirrors the twelve product categories used in Kostric et al. (2024)
("Generating Usage-related Questions for Preference Elicitation in
Conversational Recommender Systems").

Why synthetic data?
The original paper's corpus (Amazon Reviews, Ni et al. 2019) is not
redistributable and is not reachable from this offline research
environment. To let every result in the accompanying paper be re-run
end-to-end without external downloads, we generate a corpus from a
hand-authored "usage-facet taxonomy" (one JSON file per category, see
`data/taxonomy/*.json`). Each facet is expanded into many surface-form
review sentences using a template bank that deliberately covers *more*
syntactic patterns than the original paper's single "for + gerund" rule
(that heuristic is reproduced as pattern P1 below for a fair baseline).

The taxonomy plays the role of a silver-standard "population of possible
uses" for a category -- i.e. exactly the ground truth that RQ-Coverage
(see metrics.py) needs and that the original paper notes is unavailable.
Because we control the generator, we know the true facet of every
sentence, which lets us measure recall/coverage exactly instead of only
approximating it.
"""
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
TAXO_DIR = ROOT / "data" / "taxonomy"
TAXO_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Usage-facet taxonomy: category -> list of (facet_id, facet_phrase, aspect)
# ---------------------------------------------------------------------------
# Each facet is a distinct "way an item is used" (the unit of recall/coverage).
# facet_phrase is a short gerund/noun usage description; aspect is the product
# feature the review sentence will foreground alongside the usage.

TAXONOMY = {
    "Backpacking Packs": [
        ("multi-day hiking", "hip belt"), ("air travel / carry-on", "compressibility"),
        ("weekend camping trips", "frame support"), ("thru-hiking", "ventilation"),
        ("day hikes", "light weight"), ("winter mountaineering", "ice-axe loops"),
        ("commuting with a laptop", "padded sleeve"), ("bikepacking", "roll-top closure"),
        ("scout trips with kids", "durability"), ("ultralight backpacking", "pack volume"),
    ],
    "Tents": [
        ("backpacking trips", "packed weight"), ("car camping", "vestibule space"),
        ("festival camping", "quick setup"), ("winter camping", "pole strength"),
        ("family camping trips", "interior room"), ("solo trekking", "footprint size"),
        ("desert camping", "ventilation mesh"), ("rainy-season trips", "rainfly coverage"),
        ("backyard sleepovers", "ease of pitching"), ("canoe trips", "compact stuff sack"),
    ],
    "Bikes": [
        ("commuting to work", "fender mounts"), ("conquering tough terrain", "tire width"),
        ("climbing hills", "low gear ratio"), ("long distance touring", "saddle comfort"),
        ("casual weekend rides", "upright geometry"), ("racing", "frame weight"),
        ("riding with kids", "cargo rack"), ("gravel trails", "tire clearance"),
        ("commuting in the rain", "disc brakes"), ("beach cruising", "wide tires"),
    ],
    "Jackets": [
        ("winter hiking", "insulation"), ("running in the rain", "breathability"),
        ("everyday commuting", "packability"), ("ski trips", "waterproofing"),
        ("casual wear in the city", "styling"), ("mountaineering expeditions", "hood design"),
        ("layering in cold weather", "fit"), ("travel in variable climates", "packable hood"),
        ("dog walking in winter", "warmth"), ("fishing trips", "pocket layout"),
    ],
    "Vacuums": [
        ("cleaning pet hair", "suction power"), ("hardwood floors", "brush roll"),
        ("thick carpets", "motor strength"), ("small apartments", "compact size"),
        ("stairs", "cordless design"), ("car interiors", "attachment hose"),
        ("allergy-prone households", "HEPA filter"), ("large homes", "battery life"),
        ("quick daily touch-ups", "lightweight body"), ("under furniture", "low profile head"),
    ],
    "Blenders": [
        ("smoothies with frozen fruit", "blade sharpness"), ("baby food", "small jars"),
        ("protein shakes", "portability"), ("soups", "heating function"),
        ("crushing ice for cocktails", "motor power"), ("nut butters", "high-speed setting"),
        ("meal prepping", "large pitcher"), ("small single servings", "personal cup"),
        ("dorm room use", "compact footprint"), ("daily green juice", "easy cleanup"),
    ],
    "Espresso Machines": [
        ("making espresso drinks", "pressure consistency"), ("milk-based lattes", "steam wand"),
        ("beginners learning to pull shots", "guided controls"), ("home baristas", "PID control"),
        ("small kitchens", "compact footprint"), ("office break rooms", "programmable dosing"),
        ("daily morning routine", "quick warm-up"), ("entertaining guests", "double boiler"),
        ("travel and camping", "manual lever"), ("decaf and specialty beans", "temperature control"),
    ],
    "Grills": [
        ("weekend barbecues", "grate size"), ("camping trips", "portability"),
        ("searing steaks", "high heat output"), ("slow smoking ribs", "temperature control"),
        ("apartment balconies", "compact size"), ("tailgating", "lightweight build"),
        ("large family gatherings", "cooking surface area"), ("quick weeknight grilling", "fast ignition"),
        ("grilling vegetables", "even heat distribution"), ("searing satay skewers", "small charcoal use"),
    ],
    "Walk-Behind Lawn Mowers": [
        ("small suburban lawns", "compact deck"), ("thick overgrown grass", "engine power"),
        ("hilly yards", "self-propel drive"), ("weekly mowing routine", "easy start"),
        ("large open lawns", "wide cutting deck"), ("mulching leaves in fall", "mulching blade"),
        ("tight corners around flower beds", "maneuverability"), ("eco-conscious mowing", "battery power"),
        ("rocky or uneven terrain", "sturdy wheels"), ("bagging clippings", "large collection bag"),
    ],
    "Birdhouses": [
        ("attracting bluebirds", "entrance hole size"), ("backyard gardens", "mounting pole"),
        ("harsh winter climates", "insulated walls"), ("easy cleaning between seasons", "hinged roof"),
        ("attracting wrens", "compact cavity"), ("mounting on a tree", "hanging hook"),
        ("decorative porch display", "finish and paint"), ("predator protection", "reinforced entrance"),
        ("rainy climates", "sloped roof"), ("multiple nesting seasons", "weatherproof wood"),
    ],
    "Feeders": [
        ("attracting cardinals", "perch design"), ("squirrel-proofing the yard", "baffle"),
        ("winter feeding", "seed capacity"), ("hummingbirds", "nectar reservoir"),
        ("small balconies", "compact mounting"), ("large flocks", "multiple feeding ports"),
        ("easy refilling", "wide fill port"), ("rainy climates", "weatherproof roof"),
        ("window viewing", "suction mount"), ("attracting finches", "small seed ports"),
    ],
    "Snow Shovels": [
        ("clearing long driveways", "wide blade"), ("light powdery snow", "lightweight shaft"),
        ("heavy wet snow", "reinforced blade"), ("clearing steps and porches", "compact size"),
        ("elderly users with back issues", "ergonomic handle"), ("icy walkways", "ice-scraper edge"),
        ("apartment sidewalks", "foldable design"), ("quick daily clearing", "wide push design"),
        ("clearing car windshields", "multi-tool attachment"), ("rural properties", "durable build"),
    ],
}

for cat, facets in TAXONOMY.items():
    with open(TAXO_DIR / f"{cat.replace(' ', '_')}.json", "w") as f:
        json.dump([{"facet_id": i, "usage": u, "aspect": a} for i, (u, a) in enumerate(facets)], f, indent=2)

# ---------------------------------------------------------------------------
# 2. Surface-form templates (P1..P7) -- P1 reproduces the original paper's
#    single extraction pattern ("for + V-ing"); P2-P7 are additional
#    linguistic realizations of usage that the original heuristic misses,
#    which is precisely why the original pipeline under-covers usage space.
# ---------------------------------------------------------------------------
PATTERNS = {
    "P1_for_gerund": [
        "The {aspect} is great for {usage}.",
        "Perfect for {usage}, thanks to the {aspect}.",
        "I bought this mainly for {usage} and the {aspect} does not disappoint.",
    ],
    "P2_when_clause": [
        "When I go {usage_ing}, the {aspect} really shows its value.",
        "I noticed the {aspect} makes a big difference when {usage_ing}.",
    ],
    "P3_ideal_adj": [
        "This is an ideal choice if you're into {usage}.",
        "Ideal for anyone who does a lot of {usage}.",
    ],
    "P4_during_prep": [
        "During {usage}, the {aspect} held up really well.",
        "We relied on the {aspect} during several {usage} trips.",
    ],
    "P5_comparative": [
        "Compared to my old one, this is much better for {usage} because of the {aspect}.",
    ],
    "P6_bare_noun": [
        "Bought it for {usage}; the {aspect} was the deciding factor.",
        "My go-to gear for {usage} now, mostly because of the {aspect}.",
    ],
    "P7_first_person_narrative": [
        "Every time we head out for {usage}, I'm glad we have the {aspect}.",
        "After a season of {usage}, the {aspect} still performs like new.",
    ],
    # distractor sentences: mention the item but no usage information;
    # these should be correctly rejected as N/A by a good pipeline.
    "N0_generic_praise": [
        "Great product, exactly as described, fast shipping.",
        "Thank you so much for coming up with such a great product.",
        "Five stars, would buy again.",
        "The build quality feels premium for the price.",
    ],
}


def _gerundify(phrase: str) -> str:
    """Very small helper to turn a usage noun phrase into a gerund clause."""
    if phrase.endswith("ing"):
        return phrase
    # crude heuristic pluralization/verb conversion is unnecessary here since
    # all taxonomy entries are already written as gerund/noun usage phrases.
    return phrase


@dataclass
class ReviewSentence:
    category: str
    sentence: str
    pattern: str
    facet_id: int
    usage_gold: str
    aspect_gold: str
    is_applicable: bool


def generate_corpus(n_per_facet: int = 6, distractor_ratio: float = 0.18) -> List[ReviewSentence]:
    """Generate the full synthetic corpus.

    n_per_facet: number of applicable sentences generated per usage facet
                 (spread across the P1-P7 patterns), simulating the fact
                 that popular uses are mentioned in many reviews while some
                 niche uses appear rarely.
    distractor_ratio: fraction of *additional* N/A sentences injected per
                 category, mirroring the ~25% N/A rate reported in the
                 original paper (Section 4.3).
    """
    corpus: List[ReviewSentence] = []
    pattern_keys = [k for k in PATTERNS if not k.startswith("N0")]

    for cat, facets in TAXONOMY.items():
        # Zipfian popularity across facets: some uses are mentioned far more
        # often than others in real review corpora -> this is what makes
        # naive frequency-based selection under-cover the tail.
        weights = sorted([random.random() ** 2 for _ in facets], reverse=True)
        random.shuffle(weights)
        for (facet_id, (usage, aspect)), w in zip(enumerate(facets), weights):
            n = max(1, int(round(n_per_facet * (0.3 + 1.4 * w))))
            for _ in range(n):
                pat = random.choice(pattern_keys)
                tmpl = random.choice(PATTERNS[pat])
                sent = tmpl.format(usage=usage, usage_ing=_gerundify(usage), aspect=aspect)
                corpus.append(ReviewSentence(cat, sent, pat, facet_id, usage, aspect, True))

        n_distractors = max(1, int(round(len(facets) * n_per_facet * distractor_ratio)))
        for _ in range(n_distractors):
            sent = random.choice(PATTERNS["N0_generic_praise"])
            corpus.append(ReviewSentence(cat, sent, "N0_generic_praise", -1, "", "", False))

    random.shuffle(corpus)
    return corpus


if __name__ == "__main__":
    corpus = generate_corpus()
    out = ROOT / "data" / "synthetic_reviews.jsonl"
    with open(out, "w") as f:
        for r in corpus:
            f.write(json.dumps(r.__dict__) + "\n")
    print(f"Generated {len(corpus)} sentences across {len(TAXONOMY)} categories -> {out}")
