"""
generate_data.py
Generates synthetic but realistic Airbnb NYC 2019 datasets:
  - AB_NYC_2019.csv   (48 895 listings)
  - reviews.csv       (50 000 guest reviews with comments)

Run: python generate_data.py
"""
import random
import csv
import math
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

BASE = Path(__file__).parent
LISTINGS_OUT = BASE / "AB_NYC_2019.csv"
REVIEWS_OUT  = BASE / "reviews.csv"
CLEANED_DIR  = BASE / "Cleaned Data"
CLEANED_DIR.mkdir(exist_ok=True)

# ── NYC geography ────────────────────────────────────────────────────────────
BOROUGHS = {
    "Manhattan":   {"lat": (40.700, 40.880), "lon": (-74.020, -73.910),
                    "hoods": ["Midtown", "Harlem", "Upper East Side",
                              "Greenwich Village", "Chelsea", "SoHo",
                              "Financial District", "Upper West Side",
                              "Lower East Side", "Tribeca"]},
    "Brooklyn":    {"lat": (40.570, 40.740), "lon": (-74.050, -73.830),
                    "hoods": ["Williamsburg", "Park Slope", "DUMBO",
                              "Bushwick", "Bed-Stuy", "Crown Heights",
                              "Sunset Park", "Borough Park"]},
    "Queens":      {"lat": (40.540, 40.800), "lon": (-73.960, -73.700),
                    "hoods": ["Astoria", "Jackson Heights", "Flushing",
                              "Long Island City", "Jamaica", "Forest Hills"]},
    "Bronx":       {"lat": (40.780, 40.920), "lon": (-73.940, -73.750),
                    "hoods": ["Fordham", "Riverdale", "Mott Haven",
                              "Pelham Bay", "Tremont"]},
    "Staten Island":{"lat": (40.490, 40.650), "lon": (-74.260, -74.050),
                    "hoods": ["St. George", "Stapleton", "Tottenville",
                              "New Dorp", "Eltingville"]},
}

ROOM_TYPES    = ["Entire home/apt", "Private room", "Shared room"]
ROOM_WEIGHTS  = [0.52, 0.44, 0.04]
HOST_NAMES    = ["Sofia", "Marco", "Emma", "James", "Olivia", "Liam",
                 "Ava", "Noah", "Isabella", "Mason", "Lucas", "Mia",
                 "Charlotte", "Ethan", "Amelia", "Harper", "Aria",
                 "Elijah", "Evelyn", "Aiden", "Chloe", "Jackson",
                 "Ella", "Grayson", "Luna", "Zoe", "Nora"]
LISTING_ADJECTIVES = ["Cozy", "Spacious", "Charming", "Modern", "Sunny",
                       "Stylish", "Lovely", "Bright", "Beautiful", "Quiet",
                       "Elegant", "Trendy", "Peaceful", "Luxurious", "Homey"]
LISTING_NOUNS = ["Studio", "Apartment", "Room", "Loft", "Suite",
                 "Getaway", "Retreat", "Haven", "Hideaway", "Gem",
                 "Place", "Home", "Nest", "Pad", "Space"]

# ── Review text templates ─────────────────────────────────────────────────────
POSITIVE_PHRASES = [
    "Absolutely loved staying here! The place was spotless and the host was super responsive.",
    "Amazing experience from start to finish. The apartment exceeded all expectations.",
    "Perfect location in NYC! The host was incredibly welcoming and the space was beautiful.",
    "Wonderful stay. Everything was exactly as described and the neighbourhood was great.",
    "Highly recommend this place! Very clean, comfortable, and close to everything.",
    "The host went above and beyond to make us feel at home. Stunning views too!",
    "Best Airbnb I've ever stayed in. Modern, clean, well-equipped kitchen.",
    "Fantastic spot! Will definitely book again next time I'm in NYC.",
    "Super cozy and well-located apartment. The host was very communicative.",
    "Great value for the price. Clean, comfortable, and exactly as pictured.",
    "Such a lovely place! The neighbourhood was vibrant and walkable.",
    "Outstanding host, immaculate apartment, fantastic stay overall.",
    "Really enjoyed our stay. The place had a great vibe and excellent amenities.",
    "The apartment was even nicer than the photos. Very happy with our choice.",
    "Excellent communication from the host. Smooth check-in and check-out.",
    "This place has character and charm. Loved every minute of our stay.",
    "Perfectly located for exploring the city. Very clean and comfortable.",
    "Spectacular views and a very comfortable bed. Slept like a baby!",
    "The host was so friendly and helpful with local recommendations.",
    "Immaculate, stylish, and very well equipped. Highly recommend.",
]

NEUTRAL_PHRASES = [
    "Decent place, nothing special but served its purpose for the price.",
    "The apartment was okay. A bit smaller than expected but manageable.",
    "Average stay. The location was good but the place could use some updates.",
    "It was fine. Host was responsive but communication could be better.",
    "The place is clean and functional. Not luxurious but comfortable enough.",
    "Standard NYC apartment experience. Location was convenient.",
    "Nothing exceptional but a reasonable place to stay for the price.",
    "The place is as described. Some minor noise from the street at night.",
    "Acceptable accommodation. The wifi was slow but everything else was ok.",
    "The stay met basic expectations. A few things could be improved.",
    "Decent for a short trip. The neighbourhood was quieter than expected.",
    "It's a compact space as expected in NYC. Functional and clean.",
    "Average experience. The place was tidy and the host was polite.",
    "Okay stay. The apartment is dated but clean and well-located.",
    "Acceptable for the price. Would consider other options next time.",
    "The place served its purpose. Nothing stood out positively or negatively.",
    "Fair accommodation. The noise level was higher than we expected.",
    "It was a reasonable stay. The location was convenient for our needs.",
    "Satisfactory place. Would stay again if nothing better was available.",
    "The apartment is basic but has all the essentials you need.",
]

NEGATIVE_PHRASES = [
    "Very disappointed. The place was nothing like the photos.",
    "Terrible experience. There were cockroaches and the host didn't help.",
    "Would not recommend. The apartment was dirty and smelled musty.",
    "Awful stay. The heating was broken and it was freezing cold.",
    "The host was unresponsive throughout our stay. Very frustrating.",
    "Not as described at all. Felt misleading and uncomfortable.",
    "The neighbourhood felt unsafe and the lock on the door was broken.",
    "Poor value. The place was tiny, dirty, and overpriced for what was offered.",
    "Extremely noisy throughout the night. Couldn't sleep at all.",
    "The photos are highly misleading. The actual space was much smaller.",
    "Mouldy bathroom and a very uncomfortable bed. Would not return.",
    "The check-in process was a nightmare. Host went silent for hours.",
    "Disgusting kitchen. Clearly hadn't been cleaned before our arrival.",
    "The listing had bugs/mice. Had to cut our stay short.",
    "False advertising. The view shown doesn't exist and the space was cramped.",
    "Worst Airbnb experience ever. Save your money and book a hotel.",
    "Broken appliances and the host took days to respond. Very stressful.",
    "The place was cold, dark, and felt unsafe. Never again.",
    "Multiple issues including broken WiFi, cold water, and noisy neighbours.",
    "Hidden fees and unexpected rules that weren't mentioned in the listing.",
]

REVIEWER_NAMES = ["Alice", "Bob", "Carlos", "Diana", "Edward", "Fiona",
                  "George", "Hannah", "Ivan", "Julia", "Kevin", "Laura",
                  "Michael", "Nina", "Oliver", "Patricia", "Quinn", "Rachel",
                  "Samuel", "Tanya", "Ursula", "Victor", "Wendy", "Xavier",
                  "Yara", "Zack", "Amara", "Bruno", "Chiara", "Dmitri"]


def rand_float(lo, hi, decimals=6):
    return round(random.uniform(lo, hi), decimals)


def rand_date(start="2015-01-01", end="2019-12-31"):
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    return s + timedelta(days=random.randint(0, (e - s).days))


# ── Generate listings ─────────────────────────────────────────────────────────
print("Generating listings …")
listings = []
listing_id = 1000
for _ in range(48895):
    borough = random.choices(list(BOROUGHS.keys()),
                             weights=[0.47, 0.32, 0.12, 0.05, 0.04])[0]
    info  = BOROUGHS[borough]
    hood  = random.choice(info["hoods"])
    rtype = random.choices(ROOM_TYPES, weights=ROOM_WEIGHTS)[0]

    if rtype == "Entire home/apt":
        price = int(random.lognormvariate(math.log(180), 0.7))
    elif rtype == "Private room":
        price = int(random.lognormvariate(math.log(90), 0.6))
    else:
        price = int(random.lognormvariate(math.log(55), 0.5))
    price = min(price, 2000)

    n_reviews = random.randint(0, 600)
    last_review = str(rand_date()) if n_reviews > 0 else ""
    rpm = round(random.uniform(0.1, 5.0), 2) if n_reviews > 0 else ""

    listings.append({
        "id": listing_id,
        "name": f"{random.choice(LISTING_ADJECTIVES)} {random.choice(LISTING_NOUNS)} in {hood}",
        "host_id": random.randint(10000, 999999),
        "host_name": random.choice(HOST_NAMES),
        "neighbourhood_group": borough,
        "neighbourhood": hood,
        "latitude": rand_float(*info["lat"]),
        "longitude": rand_float(*info["lon"]),
        "room_type": rtype,
        "price": price,
        "minimum_nights": random.choice([1, 1, 1, 2, 2, 3, 5, 7, 14, 30]),
        "number_of_reviews": n_reviews,
        "last_review": last_review,
        "reviews_per_month": rpm,
        "calculated_host_listings_count": random.randint(1, 10),
        "availability_365": random.randint(0, 365),
    })
    listing_id += random.randint(1, 5)

with open(LISTINGS_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=listings[0].keys())
    writer.writeheader()
    writer.writerows(listings)
print(f"  ✓ {LISTINGS_OUT.name}: {len(listings):,} rows")

# ── Generate reviews ──────────────────────────────────────────────────────────
print("Generating reviews …")
listing_ids = [l["id"] for l in listings if l["number_of_reviews"] > 0]

SENTIMENT_WEIGHTS = [0.60, 0.25, 0.15]   # pos / neu / neg

reviews = []
review_id = 1
for _ in range(50000):
    lid = random.choice(listing_ids)
    sentiment_bucket = random.choices(["pos", "neu", "neg"],
                                      weights=SENTIMENT_WEIGHTS)[0]
    if sentiment_bucket == "pos":
        comment = random.choice(POSITIVE_PHRASES)
    elif sentiment_bucket == "neu":
        comment = random.choice(NEUTRAL_PHRASES)
    else:
        comment = random.choice(NEGATIVE_PHRASES)

    reviews.append({
        "listing_id": lid,
        "id": review_id,
        "date": str(rand_date()),
        "reviewer_id": random.randint(100000, 9999999),
        "reviewer_name": random.choice(REVIEWER_NAMES),
        "comments": comment,
    })
    review_id += random.randint(1, 3)

with open(REVIEWS_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=reviews[0].keys())
    writer.writeheader()
    writer.writerows(reviews)
print(f"  ✓ {REVIEWS_OUT.name}: {len(reviews):,} rows")
print("\nDone! You can now run the notebooks.")
