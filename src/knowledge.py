"""Static knowledge for the skin chatbot: disease facts, follow-up questions,
vocabulary helpers and safety patterns. Informational only, not medical advice."""

DISEASES = {
    "acne": {
        "name": "Acne",
        "aliases": ["acne", "pimple problem"],
        "description": "Acne happens when hair follicles clog with oil and dead skin cells, causing pimples, blackheads or whiteheads.",
        "common_symptoms": ["Pimples", "Blackheads", "Whiteheads", "Oily skin", "Deep painful lumps"],
        "typically_affects": "Face, forehead, chest, upper back, shoulders",
        "hallmark": "blackheads, whiteheads and pimples on oily areas",
        "general_advice": "Wash gently twice a day, use non-comedogenic products and avoid picking or squeezing spots.",
        "causes": "Clogged pores from excess oil and dead skin, bacteria, and hormone changes. Stress, some cosmetics and certain medicines can make it worse.",
        "contagious": "No. Acne is not contagious.",
        "treatment": "Gentle cleansing, non-comedogenic products and not picking at spots help. Over-the-counter products with benzoyl peroxide, salicylic acid or adapalene are common first steps. Deep or scarring acne usually needs prescription treatment from a dermatologist.",
        "see_doctor": "See a dermatologist for deep painful cysts, scarring, no improvement after 6 to 8 weeks of basic care, or if acne is affecting your mood.",
    },
    "eczema": {
        "name": "Eczema (atopic dermatitis)",
        "aliases": ["eczema", "atopic dermatitis"],
        "description": "Eczema causes dry, itchy, inflamed skin. It is common in children but can start at any age and tends to flare and settle.",
        "common_symptoms": ["Dry skin", "Itching", "Red patches", "Cracked skin", "Oozing or crusting"],
        "typically_affects": "Elbow creases, behind the knees, hands, face",
        "hallmark": "dry, intensely itchy skin, often in the creases of elbows and knees",
        "general_advice": "Moisturize regularly with a fragrance-free cream, avoid harsh soaps and known triggers, and try not to scratch.",
        "causes": "Usually a mix of a weaker skin barrier, an overactive immune response and genetics. It often runs with asthma or hay fever. Dry air, soaps, fragrance, sweat, stress and some fabrics can trigger flares.",
        "contagious": "No. Eczema is not contagious.",
        "treatment": "Moisturize at least twice a day with a fragrance-free cream or ointment, keep showers short and lukewarm, avoid triggers and avoid scratching. Flares are often treated with prescription anti-inflammatory creams.",
        "see_doctor": "See a doctor if the skin oozes, crusts or turns yellow (possible infection), if itching ruins your sleep, or if regular moisturizing is not enough.",
    },
    "psoriasis": {
        "name": "Psoriasis",
        "aliases": ["psoriasis"],
        "description": "Psoriasis causes thick red patches covered in silvery scales. It is an immune condition that speeds up skin cell growth.",
        "common_symptoms": ["Red patches", "Silvery scales", "Dry cracked skin", "Itching", "Pitted nails"],
        "typically_affects": "Elbows, knees, scalp, lower back",
        "hallmark": "well-defined red plaques with silvery-white scale",
        "general_advice": "Moisturize often, keep baths short and lukewarm, avoid skin injury and try to manage stress.",
        "causes": "An immune system condition that speeds up skin cell turnover. Genetics play a role, and flares can follow stress, infections, skin injury, cold weather or some medicines.",
        "contagious": "No. Psoriasis is not contagious.",
        "treatment": "Moisturizers, short lukewarm baths and stress management help. A dermatologist can prescribe topical treatments, light therapy or other medicines depending on severity.",
        "see_doctor": "See a dermatologist for joint pain or stiffness, widespread plaques, nail changes, or patches that keep returning.",
    },
    "ringworm": {
        "name": "Ringworm (tinea)",
        "aliases": ["ringworm", "ring worm", "tinea", "athlete's foot", "athletes foot", "jock itch"],
        "description": "Ringworm is a fungal skin infection that causes a ring-shaped rash. Despite the name, no worm is involved.",
        "common_symptoms": ["Ring-shaped rash", "Raised scaly edge", "Clearer center", "Itching", "Slowly expanding circle"],
        "typically_affects": "Arms, legs, trunk, scalp, groin, feet",
        "hallmark": "a ring with a raised scaly edge and a clearer center",
        "general_advice": "Keep the area clean and dry, and do not share towels or clothing.",
        "causes": "A fungal infection. It spreads by skin contact with an infected person or pet, or through shared towels, clothing and damp gym surfaces.",
        "contagious": "Yes. It spreads through direct contact and shared items, so avoid sharing towels and wash your hands after touching the area.",
        "treatment": "Keep the area clean and dry. Over-the-counter antifungal creams such as clotrimazole or terbinafine are commonly used for 2 to 4 weeks. Scalp or nail infections usually need prescription tablets.",
        "see_doctor": "See a doctor for scalp or nail involvement, a widespread rash, or no improvement after about 2 weeks of antifungal cream.",
    },
    "vitiligo": {
        "name": "Vitiligo",
        "aliases": ["vitiligo"],
        "description": "Vitiligo causes patches of skin to lose their pigment, so they look white or lighter than the skin around them.",
        "common_symptoms": ["White patches", "Loss of skin color", "Early graying of hair in the area", "Smooth, usually painless skin"],
        "typically_affects": "Hands, face, arms, feet, around the mouth and eyes",
        "hallmark": "smooth, completely pale patches with no scaling or itching",
        "general_advice": "Protect pale patches from the sun with sunscreen, since they burn easily.",
        "causes": "The immune system attacks the cells that make pigment. Genetics and other autoimmune conditions are linked, and skin injury or stress sometimes comes before it.",
        "contagious": "No. Vitiligo is not contagious.",
        "treatment": "Sunscreen protects the lighter skin. A dermatologist can discuss topical medicines, light therapy or cosmetic cover. Treatment works slowly and aims to restore or even out color.",
        "see_doctor": "See a dermatologist when patches first appear or spread, since early review gives more options, and if it is affecting your wellbeing.",
    },
    "rosacea": {
        "name": "Rosacea",
        "aliases": ["rosacea"],
        "description": "Rosacea causes lasting redness, visible tiny blood vessels and sometimes small bumps on the central face.",
        "common_symptoms": ["Facial redness", "Visible blood vessels", "Easy flushing", "Pimple-like bumps", "Stinging skin"],
        "typically_affects": "Nose, cheeks, forehead, chin",
        "hallmark": "persistent redness and flushing in the center of the face without blackheads",
        "general_advice": "Avoid triggers such as sun, heat, spicy food and alcohol, use daily sunscreen and gentle skincare.",
        "causes": "Not fully understood. Blood vessel changes, genetics and skin inflammation may play a part. Sun, heat, spicy food, alcohol and stress are common triggers.",
        "contagious": "No. Rosacea is not contagious.",
        "treatment": "Gentle skincare, daily sunscreen and avoiding your triggers help. A dermatologist can prescribe topical or oral medicines, and laser treatment for visible vessels.",
        "see_doctor": "See a dermatologist for redness that does not fade, bumps that keep returning, thickening skin on the nose, or irritated eyes.",
    },
    "contact_dermatitis": {
        "name": "Contact dermatitis",
        "aliases": ["contact dermatitis", "allergic rash", "allergic reaction rash"],
        "description": "Contact dermatitis is a rash that appears where the skin touched an irritant or allergen, such as metal, soap, cosmetics or plants.",
        "common_symptoms": ["Red itchy rash", "Burning or stinging", "Blisters", "Dry cracked skin", "Rash matching the shape of the contact"],
        "typically_affects": "Hands, face, neck, wrists, wherever the trigger touched",
        "hallmark": "a rash limited to the area that touched a new product, metal or plant",
        "general_advice": "Find and avoid the trigger, rinse the area with mild soap and water, and use a plain moisturizer.",
        "causes": "The skin reacts to something it touched. It can be irritant (soaps, cleaners, frequent water) or allergic (nickel, fragrance, latex, plants like poison ivy).",
        "contagious": "No. The rash is not contagious, although plant oils left on hands or clothes can spread to other people until washed off.",
        "treatment": "Identify and avoid the trigger, wash the area with mild soap and water, moisturize and wear protective gloves where needed. Cool compresses ease itching. A doctor may prescribe a steroid cream for stronger reactions.",
        "see_doctor": "See a doctor for a rash on the face or genitals, severe blistering, signs of infection, or a rash that does not clear in 2 to 3 weeks after avoiding the trigger.",
    },
    "hives": {
        "name": "Hives (urticaria)",
        "aliases": ["hives", "urticaria", "welts"],
        "description": "Hives are raised, very itchy welts that appear suddenly, move around and usually fade within hours.",
        "common_symptoms": ["Raised itchy welts", "Welts that turn white when pressed", "Sudden onset", "Comes and goes within hours", "Swelling"],
        "typically_affects": "Can appear anywhere on the body",
        "hallmark": "welts that appear suddenly, move around and fade within hours",
        "general_advice": "Avoid known triggers and use cool compresses. Get urgent help if you have swelling of the lips, tongue or throat, or trouble breathing.",
        "causes": "Histamine release, often from allergies (foods, medicines, insect stings), infections, heat, cold, pressure or stress. Often no cause is found.",
        "contagious": "No. Hives are not contagious.",
        "treatment": "Non-drowsy antihistamines are the usual first step, along with cool compresses and avoiding triggers. A doctor can advise if hives last more than a few weeks.",
        "see_doctor": "Seek emergency care right away if hives come with swelling of the lips, tongue or throat, trouble breathing or dizziness. See a doctor if they last more than about 6 weeks.",
    },
    "seborrheic_dermatitis": {
        "name": "Seborrheic dermatitis",
        "aliases": ["seborrheic dermatitis", "seborrhoeic dermatitis", "seb derm", "cradle cap", "dandruff"],
        "description": "Seborrheic dermatitis causes greasy, yellowish flakes and redness on oily areas such as the scalp, eyebrows and sides of the nose. Dandruff is a mild form.",
        "common_symptoms": ["Flaky scalp", "Greasy yellow scales", "Redness", "Mild itching", "Dandruff"],
        "typically_affects": "Scalp, eyebrows, sides of the nose, behind the ears, chest",
        "hallmark": "greasy, yellowish flakes on oily areas",
        "general_advice": "Use a medicated anti-dandruff shampoo, wash affected areas gently and manage stress.",
        "causes": "An inflammatory reaction linked to a yeast (Malassezia) that lives on oily skin. Stress, cold dry weather and some health conditions can trigger flares.",
        "contagious": "No. Seborrheic dermatitis is not contagious.",
        "treatment": "Anti-dandruff shampoos with ketoconazole, selenium sulfide or zinc pyrithione are common for the scalp. Mild antifungal or steroid creams are used on the face under medical advice.",
        "see_doctor": "See a doctor for thick crusting, hair loss, redness that spreads, or no improvement after about 4 weeks of medicated shampoo.",
    },
    "scabies": {
        "name": "Scabies",
        "aliases": ["scabies", "mites"],
        "description": "Scabies is caused by tiny mites that burrow into the skin, causing intense itching that is worst at night.",
        "common_symptoms": ["Intense itching at night", "Thin burrow lines", "Tiny red bumps", "Rash between fingers", "Others around you also itchy"],
        "typically_affects": "Between the fingers, wrists, elbows, waistline, groin",
        "hallmark": "intense night-time itch, thin burrow lines and others in the household itching too",
        "general_advice": "See a doctor for treatment, wash bedding and clothes in hot water, and have close contacts checked.",
        "causes": "Tiny mites (Sarcoptes scabiei) burrowing into the skin. They spread through prolonged skin-to-skin contact and sometimes shared bedding.",
        "contagious": "Yes, it is highly contagious. Household members and close contacts usually need treatment at the same time.",
        "treatment": "Scabies needs prescription medicine, usually a permethrin cream applied over the whole body, sometimes tablets. Wash bedding and clothes in hot water. Itching can last for weeks after the mites are gone.",
        "see_doctor": "See a doctor as soon as you suspect scabies, since treatment needs a prescription and close contacts should be treated too.",
    },
}

# Yes/No follow-ups. "supports" are the diseases a "yes" points toward.
QUESTIONS = {
    "itch_night": {
        "text": "Is the itching much worse at night?",
        "supports": ["scabies", "eczema"],
    },
    "household_itch": {
        "text": "Is anyone else in your home or close contact also itchy?",
        "supports": ["scabies"],
    },
    "burrows": {
        "text": "Do you see thin wavy lines or tiny burrows, especially between your fingers or on your wrists?",
        "supports": ["scabies"],
    },
    "ring_shape": {
        "text": "Is the rash ring-shaped, with a raised edge and a clearer center?",
        "supports": ["ringworm"],
    },
    "expanding": {
        "text": "Has the patch been slowly growing outward over days or weeks?",
        "supports": ["ringworm", "vitiligo"],
    },
    "silvery_scale": {
        "text": "Are the red patches covered with thick, silvery-white scales?",
        "supports": ["psoriasis"],
    },
    "nail_changes": {
        "text": "Are any of your nails pitted, thickened or crumbling?",
        "supports": ["psoriasis", "ringworm"],
    },
    "new_contact": {
        "text": "Did it start after touching something new, such as jewelry, soap, cosmetics, a plant or a cleaning product?",
        "supports": ["contact_dermatitis"],
    },
    "exact_spot": {
        "text": "Is the rash only where something touched your skin, matching its shape?",
        "supports": ["contact_dermatitis"],
    },
    "come_and_go": {
        "text": "Do the bumps or welts come and go within hours, popping up in new places?",
        "supports": ["hives"],
    },
    "food_med": {
        "text": "Did it begin soon after eating something, taking a medicine or an insect sting?",
        "supports": ["hives"],
    },
    "blanch": {
        "text": "Do the raised areas turn white when you press on them?",
        "supports": ["hives"],
    },
    "dry_creases": {
        "text": "Is your skin generally very dry, with itch mostly in the creases of elbows or knees?",
        "supports": ["eczema"],
    },
    "allergy_history": {
        "text": "Do you or your family have asthma, hay fever or other allergies?",
        "supports": ["eczema"],
    },
    "oozing": {
        "text": "Does the skin ooze or crust over when scratched?",
        "supports": ["eczema"],
    },
    "greasy_flakes": {
        "text": "Are the flakes greasy or yellowish, mostly on oily areas like the scalp, eyebrows or beside the nose?",
        "supports": ["seborrheic_dermatitis"],
    },
    "dandruff": {
        "text": "Do you also get dandruff?",
        "supports": ["seborrheic_dermatitis"],
    },
    "comedones": {
        "text": "Are there blackheads or whiteheads?",
        "supports": ["acne"],
    },
    "oily_new": {
        "text": "Is your skin oily, with new pimples appearing regularly?",
        "supports": ["acne"],
    },
    "deep_lumps": {
        "text": "Are there deep, painful lumps under the skin?",
        "supports": ["acne"],
    },
    "flushing": {
        "text": "Does your face flush easily with heat, spicy food, alcohol or stress?",
        "supports": ["rosacea"],
    },
    "visible_vessels": {
        "text": "Can you see tiny red blood vessels on your cheeks or nose?",
        "supports": ["rosacea"],
    },
    "white_smooth": {
        "text": "Are the patches completely white or paler than your normal skin, smooth, with no scaling?",
        "supports": ["vitiligo"],
    },
    "white_hair": {
        "text": "Has the hair in or near the patch turned white?",
        "supports": ["vitiligo"],
    },
}

LOCATION_QUESTION = "Where on your body is this? For example face, scalp, hands, elbows, legs or back."

QUICK_LOCATIONS = ["Face", "Scalp", "Hands", "Arms or legs", "Torso", "Skip"]

BODY_PARTS = (
    r"face|cheek|forehead|nose|chin|jaw|scalp|hair|neck|arm|hand|finger|nail|palm|wrist|elbow|"
    r"leg|knee|foot|feet|toe|ankle|thigh|shin|calf|back|chest|stomach|belly|abdomen|groin|armpit|"
    r"underarm|torso|body|eyelid|eyebrow|lip|mouth|ear|buttock|shoulder|waist|everywhere"
)

# Words that show a message is about skin. Used to spot off-topic messages.
SKIN_TERMS = (
    r"skin|rash|itch|scratch|red|redness|pink|patch|bump|spot|pimple|zit|acne|blackhead|whitehead|"
    r"scale|scaly|flak|peel|dry|crack|blister|welt|hive|swell|swollen|burn|sting|oily|greasy|pus|"
    r"crust|ooze|weep|lesion|plaque|wart|scar|sore|ring|circular|rough|tight|bleed|irritat|inflam|"
    r"sunburn|dandruff|scalp|nail|pigment|dark|discolor|bite|sweat|pore|blotch|breakout|bald|"
    r"white|pale|lighter|vein|vessel|flush|lump|cyst|dermat|eczema|psoria|ringworm|vitiligo|rosacea|"
    r"scabies|urticaria|allerg|mole|tinea|fungal|fungus|infection|texture|bumpy|itchi|"
    r"comedo|papul|nodul|pustul|flare|fingernail|toenail|pitted|crumbl|thicken|dent|rubb|chafe|chafing"
)

# Lay wording mapped onto words the model saw in training.
SYNONYMS = {
    "zit": "pimple", "zits": "pimples", "breakout": "pimples", "breakouts": "pimples", "pustule": "pimple",
    "scratchy": "itchy", "itchiness": "itchy",
    "hurts": "painful", "hurt": "painful", "tender": "painful", "aching": "painful",
    "reddish": "red", "pinkish": "red", "flushed": "red",
    "dandruffy": "flaky",
    "blotches": "patches", "blotch": "patch", "marks": "patches", "mark": "patch",
    "pimply": "pimples",
    "kid": "child", "son": "child", "daughter": "child", "toddler": "child",
    "tummy": "stomach",
    "rashes": "rash", "colour": "color", "colourless": "colorless",
    "oozy": "oozing", "weepy": "weeping", "hairloss": "hair loss",
}

# Messages that need a clear safety warning.
RED_FLAGS = [
    r"(can'?t|cannot|trouble|difficulty|hard to|struggl\w+( to)?) breath",
    r"short(ness)? of breath|wheez",
    r"(throat|tongue|lips?|face|eyes?) (is |are |feels? |got )?(swell|swollen|puffy|closing|tight)",
    r"swell\w* (of|in|on) (my )?(throat|tongue|lips?|face)",
    r"(high |a )?fever.*(rash|blister|peel)|(rash|blister|peel).*(high |a )?fever",
    r"skin (is )?peeling (off )?in (sheets|large)",
    r"blisters? (in|on|inside) (my )?(mouth|eyes?|genital)",
    r"red streaks?|spreading (very )?fast|spreading rapidly",
    r"pus.*(fever|chills)|(fever|chills).*pus",
    r"faint(ing)?|dizz(y|iness)|passed out",
]

URGENT_TEXT = (
    "Some of what you wrote can be a sign of a serious reaction or infection. "
    "Please get medical care now: call your local emergency number or go to the nearest emergency room. "
    "Do not wait for an online tool."
)

SERIOUS_SKIN_TEXT = (
    "I can't assess moles or possible skin cancer. A mole or spot that changes in size, shape or color, "
    "bleeds, or looks different from your others should be checked by a dermatologist soon."
)
