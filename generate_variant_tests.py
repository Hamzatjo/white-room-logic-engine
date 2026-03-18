#!/usr/bin/env python3
"""Generate variant test puzzles: real-world names + format variations."""

import json

# ============================================================
# TEST SET 1: Real-world names (same logic patterns)
# ============================================================
real_world = [
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
- If a patient's temperature exceeds 38.5°C, the nurse must administer antipyretics.
- Administering antipyretics requires approval from the attending physician.
- Physician approval automatically triggers a chart update.
- A chart update flags the patient for a 4-hour observation window.
- Patient Garcia has a temperature of 39.1°C.
- Patient Garcia's blood pressure is 120/80, which is within normal range.
</document>

<question>
What is Patient Garcia's current status?
</question>""",
        "answer": "Patient Garcia is flagged for a 4-hour observation window."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
- Any invoice over $10,000 must be reviewed by the Compliance department.
- Compliance review requires documentation from the Procurement team.
- If Procurement cannot provide documentation within 48 hours, the invoice is rejected.
- A rejected invoice triggers a vendor notification.
- Invoice #4521 is for $15,200 from Acme Corp.
- Procurement has not responded in 72 hours.
</document>

<question>
What happens to Invoice #4521?
</question>""",
        "answer": "Invoice #4521 is rejected and a vendor notification is triggered."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
- All employees in the Engineering department have access to the server room.
- Server room access requires a security badge.
- If an employee loses their security badge, all their access permissions are revoked.
- Revoked permissions require manager approval to reinstate.
- Sarah is an engineer who lost her badge yesterday.
- Sarah's manager is currently on vacation.
</document>

<question>
Can Sarah access the server room?
</question>""",
        "answer": "No. Sarah's access permissions were revoked when she lost her badge, and they cannot be reinstated because her manager is on vacation."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
- Flights departing after 10 PM are classified as red-eye flights.
- Red-eye flights receive priority runway access.
- Priority runway access reduces taxi time to under 5 minutes.
- Flight AA-204 departs at 11:30 PM.
- Flight AA-204 has 180 passengers, which is considered full capacity.
</document>

<question>
What is Flight AA-204's expected taxi time?
</question>""",
        "answer": "Under 5 minutes."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
- Students with a GPA above 3.5 qualify for the Dean's List.
- Dean's List students receive a tuition discount of 15%.
- The tuition discount is applied automatically at the start of the next semester.
- James has a GPA of 3.7.
- James is also the captain of the basketball team, which has no bearing on academic honors.
</document>

<question>
What financial benefit does James receive next semester?
</question>""",
        "answer": "James receives a 15% tuition discount."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
- Any chemical stored above 25°C must be moved to cold storage.
- Moving chemicals to cold storage requires a hazmat team.
- The hazmat team operates only on weekdays between 8 AM and 5 PM.
- If hazmat is unavailable, the chemical must be disposed of immediately.
- Chemical X-47 is stored at 28°C.
- Today is Sunday.
</document>

<question>
What happens to Chemical X-47?
</question>""",
        "answer": "Chemical X-47 must be disposed of immediately because the hazmat team is unavailable on Sunday."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
- All packages over 30 kg require freight shipping.
- Freight shipping takes 5-7 business days.
- Packages under 30 kg use standard shipping, which takes 2-3 business days.
- Order #8823 weighs 12 kg.
- Order #8823 contains fragile items, which does not affect the shipping method.
</document>

<question>
How long will Order #8823 take to arrive?
</question>""",
        "answer": "2-3 business days."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
- If a server's CPU usage exceeds 90%, an alert is sent to the DevOps team.
- The DevOps team must acknowledge the alert within 15 minutes.
- If the alert is not acknowledged, the system automatically scales up by adding a new instance.
- Adding a new instance increases the monthly cloud bill by $500.
- Server Alpha's CPU usage is at 95%.
- The DevOps team is currently in a company meeting and cannot respond.
</document>

<question>
What is the financial impact?
</question>""",
        "answer": "The monthly cloud bill increases by $500."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
- Restaurants with a health inspection score below 70 are shut down immediately.
- A shutdown requires a reinspection within 30 days.
- Passing reinspection allows the restaurant to reopen.
- Failing reinspection results in permanent license revocation.
- Mario's Pizzeria scored 62 on its health inspection.
- Mario's Pizzeria has been in business for 25 years, which does not affect inspection outcomes.
</document>

<question>
What is Mario's Pizzeria's current operational status?
</question>""",
        "answer": "Mario's Pizzeria is shut down immediately and must pass a reinspection within 30 days."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
- Vehicles emitting over 200g CO2/km are banned from the city center.
- Banned vehicles that enter the city center receive a $300 fine.
- Fines not paid within 14 days are doubled.
- Tom's truck emits 245g CO2/km.
- Tom drove into the city center 20 days ago and has not paid any fines.
</document>

<question>
How much does Tom owe?
</question>""",
        "answer": "$600."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
- All items priced over $100 include free shipping.
- Items priced under $100 have a flat $12 shipping fee.
- Orders placed before noon qualify for same-day dispatch.
- Lisa ordered a $85 lamp at 10 AM.
- Lisa also used a loyalty coupon, but coupons do not change the item price for shipping calculation purposes.
</document>

<question>
What are Lisa's total shipping costs and when will her order be dispatched?
</question>""",
        "answer": "Lisa's shipping cost is $12 and her order qualifies for same-day dispatch."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
- Employees with over 10 years of tenure are eligible for sabbatical leave.
- Sabbatical leave lasts 3 months.
- During sabbatical, the employee's projects are reassigned to their team lead.
- David has worked at the company for 14 years.
- David is currently leading 3 active projects.
- David's favorite hobby is painting, which is unrelated to his work.
</document>

<question>
If David takes sabbatical, what happens to his projects?
</question>""",
        "answer": "David's 3 active projects are reassigned to his team lead."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
- Trees taller than 15 meters near power lines must be trimmed.
- Trimming requires a permit from the city council.
- Permits take 2 weeks to process.
- During permit processing, a warning sign must be placed near the tree.
- The oak tree on Maple Street is 18 meters tall and is near a power line.
- The oak tree was planted in 1952.
</document>

<question>
What immediate action is required for the oak tree on Maple Street?
</question>""",
        "answer": "A permit must be requested from the city council and a warning sign must be placed near the tree."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
- Loans over $500,000 require board approval.
- Board approval meetings are held on the first Monday of each month.
- If the board rejects a loan, the applicant must wait 6 months to reapply.
- The board rejected First National's loan application of $750,000.
- First National's CEO recently won an industry award, which has no effect on loan decisions.
</document>

<question>
When can First National reapply for the loan?
</question>""",
        "answer": "First National must wait 6 months to reapply."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
- Water samples with bacteria counts above 500 CFU/mL fail the safety test.
- Failed samples trigger a boil-water advisory for the affected area.
- A boil-water advisory remains in effect until two consecutive clean samples are obtained.
- The water sample from District 7 shows 620 CFU/mL.
- District 7 has a population of 45,000 residents.
</document>

<question>
What advisory is issued for District 7?
</question>""",
        "answer": "A boil-water advisory is issued for District 7."
    },
]

# ============================================================
# TEST SET 2: Format variations (same logic, different structure)
# ============================================================
format_variations = [
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

Here are the rules of a system:

When a Thalvex reading goes above 400 Grundels, it causes all connected Spirex units to activate. Spirex units are a type of Morven device. Every Morven device is part of the Quillon array. When any Morven device activates, it sends a pulse to the Zenthar relay. The Zenthar relay shuts down the entire Quillon array when it receives a pulse.

Current situation: A Thalvex reading of 580 Grundels has been recorded.""",
        "answer": "The Quillon array is shut down."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

Rules:
1) Braxil above 50 Kelthons → Vondar activates
2) Vondar activation → Pellisk triggered
3) Pellisk → Drenn cycle begins
4) Drenn cycle → all Frelkast units reset to zero
5) Frelkast Unit C currently holds a value of 200
6) Braxil measured at 30 Kelthons

What happens to Frelkast Unit C?""",
        "answer": "Nothing happens to Frelkast Unit C. The Braxil reading of 30 Kelthons does not exceed the threshold of 50, so no chain of events is triggered."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

Consider the following scenario in the Vexlar containment facility:

The Gorthane protocol states that any Melvux concentration exceeding 12 Prathils will contaminate the Skelvor chamber. A contaminated Skelvor chamber cannot process any further Draxen samples. There are currently 5 Draxen samples queued for processing. The latest Melvux reading shows 8 Prathils. The facility director recently noted that the ventilation system is outdated, but this has no operational impact on the Gorthane protocol.

What is the status of the 5 Draxen samples?""",
        "answer": "The 5 Draxen samples will be processed normally. The Melvux concentration of 8 Prathils does not exceed the threshold of 12, so the Skelvor chamber is not contaminated."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

SYSTEM RULES FOR KRENVOL NETWORK:
* Thaxren emissions > 70 Jolvens = Brenthar nodes go offline
* Brenthar nodes offline = Glorix relay loses signal
* Glorix relay signal loss = Fendrath backup engages
* Fendrath backup = all Skelven transmissions rerouted through Channel 9

CURRENT STATE:
- Thaxren emission level: 85 Jolvens
- Channel 9 capacity: unlimited
- Weather conditions: stormy (no effect on system)

QUERY: Through which channel are Skelven transmissions routed?""",
        "answer": "Channel 9."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
The Plovarx scale determines the behavior of Gendrik particles. Below is the classification:
- Plovarx between 0 and 100: Gendrik particles remain inert.
- Plovarx between 101 and 300: Gendrik particles emit a faint glow (cosmetic only, no functional effect).
- Plovarx above 300: Gendrik particles become volatile and must be contained in a Threlven field.
- A Threlven field requires 50 Kelvax of energy per hour to maintain.
- If energy supply drops below 50 Kelvax/hour, the Threlven field collapses.
- A collapsed Threlven field causes all Gendrik particles to disperse into the Nelvox chamber.

The current Plovarx reading is 450. The energy supply is stable at 80 Kelvax/hour.
</document>

<question>
What is the current state of the Gendrik particles?
</question>""",
        "answer": "The Gendrik particles are volatile and contained in a Threlven field. The energy supply of 80 Kelvax/hour is sufficient to maintain the field."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

In a kingdom, the following laws apply. The Crown Tax applies to all merchants earning more than 1000 gold per season. Merchants subject to the Crown Tax must register with the Royal Treasury. Registered merchants gain access to the King's Trade Routes. The King's Trade Routes reduce travel costs by half. 

Merchant Aldric earned 1,200 gold last season. Merchant Belen earned 800 gold. Aldric has a pet falcon, which is irrelevant to trade law.

Question: Which merchant, if any, gets reduced travel costs?""",
        "answer": "Merchant Aldric gets reduced travel costs. Merchant Belen does not qualify because she earned below 1000 gold."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
System log entry:
- [Rule A] Fenxal pressure > 200 bars → activate Grenthol valve
- [Rule B] Grenthol valve active → Draxen coolant released
- [Rule C] Draxen coolant + Thelvox gas present → neutralization reaction
- [Rule D] Neutralization reaction → system reaches safe state
- [Rule E] Thelvox gas is NOT present in the current chamber
- [Current] Fenxal pressure: 350 bars
</document>

<question>
Does the system reach a safe state?
</question>""",
        "answer": "No. Although the Fenxal pressure triggers the Grenthol valve and releases Draxen coolant, Thelvox gas is not present, so the neutralization reaction cannot occur and the system does not reach a safe state."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

A series of conditional rules governs the Tharven Conservatory:

First, know that any Brellix orchid exposed to more than 6 hours of direct light will bloom. Second, a blooming Brellix orchid produces Felthane pollen. Third, Felthane pollen is toxic to all Draven ferns within a 3-meter radius. Fourth, toxic exposure causes Draven ferns to wilt within 24 hours. Fifth, the Brellix orchid on Shelf A received 8 hours of direct light today. Sixth, there is a Draven fern 2 meters from Shelf A. Also, the gardener watered all plants this morning, but watering does not affect blooming or toxicity.

What will happen to the Draven fern near Shelf A within 24 hours?""",
        "answer": "The Draven fern will wilt within 24 hours."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
Geltharn reactor safety protocol:
A) Core temperature above 1500K triggers emergency coolant flush
B) Emergency coolant flush reduces temperature by 800K
C) If temperature after flush is still above 900K, reactor enters lockdown
D) Lockdown seals all access ports for 48 hours
E) Current core temperature: 1600K
F) The reactor was built in 2019 (irrelevant to safety protocols)
</document>

<question>
What is the final state of the reactor's access ports?
</question>""",
        "answer": "The access ports remain open. The core temperature of 1600K triggers an emergency coolant flush, reducing it by 800K to 800K. Since 800K is below 900K, the reactor does not enter lockdown."
    },
    {
        "prompt": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

Three entities exist in the Prelvax system: Gormund, Thelvian, and Draxos.

The rules are:
- Gormund always copies the state of Thelvian after a 1-cycle delay.
- Thelvian activates whenever Draxos is inactive.
- Draxos is currently inactive.
- Gormund is currently inactive.

After 1 cycle, what is the state of Gormund?""",
        "answer": "Gormund is active. Draxos is inactive, so Thelvian activates. After a 1-cycle delay, Gormund copies Thelvian's active state."
    },
]

# Save both test sets
real_world_jsonl = []
for item in real_world:
    real_world_jsonl.append({
        "conversations": [
            {"from": "human", "value": item["prompt"]},
            {"from": "gpt", "value": f"<reasoning>\n[Expected reasoning chain]\n</reasoning>\n<answer>\n{item['answer']}\n</answer>"}
        ]
    })

format_var_jsonl = []
for item in format_variations:
    format_var_jsonl.append({
        "conversations": [
            {"from": "human", "value": item["prompt"]},
            {"from": "gpt", "value": f"<reasoning>\n[Expected reasoning chain]\n</reasoning>\n<answer>\n{item['answer']}\n</answer>"}
        ]
    })

with open("test_realworld.jsonl", "w") as f:
    for item in real_world_jsonl:
        f.write(json.dumps(item) + "\n")

with open("test_format_variations.jsonl", "w") as f:
    for item in format_var_jsonl:
        f.write(json.dumps(item) + "\n")

print(f"Generated {len(real_world)} real-world puzzles → test_realworld.jsonl")
print(f"Generated {len(format_variations)} format variation puzzles → test_format_variations.jsonl")
