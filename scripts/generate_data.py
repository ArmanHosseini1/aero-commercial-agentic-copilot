import csv
import random
import sqlite3
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "commercial_docs"
SEED = 42

AIRLINES = [
    (1, "Horizon Air", "North America", 64, 19, "Tier-1", "Maya Chen"),
    (2, "NordWind Aviation", "Northern Europe", 78, 22, "Tier-1", "Erik Lund"),
    (3, "Pacifica Airlines", "Asia-Pacific", 92, 56, "Tier-2", "Leilani Park"),
    (4, "Solstice Airways", "North America", 49, 61, "Tier-2", "Jordan Reyes"),
    (5, "Aether Jet", "Middle East", 37, 68, "Tier-2", "Samira Haddad"),
    (6, "Meridian Express", "South Asia", 73, 47, "Tier-2", "Arjun Mehta"),
    (7, "Borealis Air", "Scandinavia", 42, 59, "Tier-2", "Ingrid Solberg"),
    (8, "Vanguard Aviation", "Latin America", 55, 43, "Tier-2", "Diego Alvarez"),
]

AIRCRAFT_ORDERS = [
    (1, 1, "A320neo-class", 12, 2027, "In Production", 840.0),
    (2, 1, "A350-class", 4, 2028, "Backlog", 1280.0),
    (3, 1, "A220-class", 8, 2026, "In Production", 408.0),
    (4, 2, "A320neo-class", 16, 2027, "In Production", 1120.0),
    (5, 2, "A350-class", 5, 2029, "Backlog", 1600.0),
    (6, 2, "A220-class", 6, 2026, "Delivered", 330.0),
    (7, 3, "A350-class", 8, 2026, "Delivered", 1360.0),
    (8, 3, "A320neo-class", 10, 2027, "In Production", 700.0),
    (9, 3, "A220-class", 12, 2028, "Backlog", 660.0),
    (10, 4, "A320neo-class", 14, 2025, "Delivered", 910.0),
    (11, 4, "A220-class", 10, 2027, "In Production", 550.0),
    (12, 5, "A350-class", 6, 2028, "Backlog", 1020.0),
    (13, 5, "A320neo-class", 8, 2026, "Delivered", 520.0),
    (14, 6, "A320neo-class", 20, 2028, "Backlog", 1400.0),
    (15, 6, "A220-class", 14, 2027, "In Production", 770.0),
    (16, 7, "A220-class", 9, 2026, "Delivered", 495.0),
    (17, 7, "A350-class", 3, 2029, "Backlog", 510.0),
    (18, 8, "A320neo-class", 11, 2026, "Delivered", 715.0),
    (19, 8, "A350-class", 4, 2027, "In Production", 680.0),
    (20, 8, "A220-class", 7, 2028, "Backlog", 385.0),
]

CONTRACT_MILESTONES = [
    (1, 1, "Cabin layout freeze and customisation sign-off", "2026-12-15", "High"),
    (2, 1, "Long-lead cabin monument configuration approval", "2027-01-20", "High"),
    (3, 1, "Delivery-readiness review for first A320neo-class aircraft", "2027-04-10", "Medium"),
    (4, 2, "Escalation plan for open technical publication actions", "2026-12-08", "High"),
    (5, 2, "A350-class cabin specification baseline approval", "2027-02-12", "High"),
    (6, 2, "First A320neo-class aircraft acceptance review", "2027-05-05", "Medium"),
    (7, 3, "A350-class entry-into-service support workshop", "2026-12-19", "Low"),
    (8, 3, "A320neo-class production configuration review", "2027-03-16", "Medium"),
    (9, 4, "Annual service-level review and renewal options", "2027-01-18", "Low"),
    (10, 4, "A220-class cabin materials approval", "2027-02-25", "Medium"),
    (11, 5, "A350-class delivery documentation package review", "2027-01-30", "Medium"),
    (12, 5, "Spare parts provisioning and support coverage review", "2027-04-22", "Low"),
    (13, 6, "A320neo-class fleet induction planning session", "2027-02-08", "Medium"),
    (14, 6, "Quarterly commercial performance review", "2027-03-28", "Low"),
    (15, 7, "A220-class cabin acceptance and defect close-out", "2027-01-12", "Medium"),
    (16, 7, "Technical publications delivery confirmation", "2027-02-21", "Low"),
    (17, 8, "A350-class contract change-control review", "2027-03-04", "Medium"),
    (18, 8, "First A320neo-class aircraft delivery planning review", "2027-04-16", "Low"),
]

SURVEY_RESPONSES = {
    "Horizon Air": [
        ("Sales", 3, "The account team understands our fleet plan, but revised proposals take several days to arrive."),
        ("Sales", 4, "The fleet-planning workshop was useful and gave our finance team a clear set of options."),
        ("Contracts", 2, "We are still reconciling two contract amendments because the approval trail was not shared promptly."),
        ("Contracts", 3, "The latest change summary is clearer, although we need earlier notice of wording changes."),
        ("Cabin Customisation", 1, "Cabin customisation lead times remain far longer than planned, and late option decisions are putting our launch schedule at risk."),
        ("Cabin Customisation", 2, "The revised cabin mock-up is helpful, but monument approvals still take weeks to turn around."),
        ("Delivery", 2, "Our delivery dashboard has not reflected the latest milestone dates, so our airport teams are planning with old information."),
        ("Delivery", 3, "The weekly delivery call helped resolve open items, but the action owners were not always recorded."),
        ("Support", 2, "Spare parts documentation arrived after the shipment, which delayed our receiving inspection."),
        ("Support", 4, "The support desk followed up consistently and gave our maintenance team a useful escalation contact."),
    ],
    "NordWind Aviation": [
        ("Sales", 3, "The commercial team responds quickly to meeting requests but has not closed the loop on our revised fleet scenario."),
        ("Sales", 2, "Our regional growth case was well understood, though the final proposal omitted two requested operating assumptions."),
        ("Contracts", 2, "We received different interpretations of the warranty language from the commercial and legal contacts."),
        ("Contracts", 2, "A change order waited for several internal approvals without a clear status update to our team."),
        ("Cabin Customisation", 2, "The cabin options list is comprehensive, but configuration decisions are arriving too late for our planning window."),
        ("Cabin Customisation", 1, "We had to repeat our seat and galley requirements after the design review, adding avoidable weeks to the process."),
        ("Delivery", 2, "The delivery sequence changed twice this month and the revised dates were not reflected in the shared tracker."),
        ("Delivery", 3, "The acceptance team was prepared, but several open documentation actions were discovered during the handover."),
        ("Support", 2, "Technical publications for replacement parts were incomplete, leaving our maintenance planners to request follow-up files."),
        ("Support", 3, "The service team acknowledged our case promptly, but the final technical answer required multiple escalations."),
    ],
    "Pacifica Airlines": [
        ("Sales", 4, "The account team compared fleet options clearly and included operating scenarios our network planners could use."),
        ("Sales", 5, "Our proposal review was well prepared, with prompt answers to questions from procurement and engineering."),
        ("Contracts", 4, "The contract summary made the delivery assumptions and responsibilities easy to review."),
        ("Contracts", 3, "The amendment was completed correctly, although the signature package could have been circulated sooner."),
        ("Cabin Customisation", 4, "The design team translated our regional passenger preferences into practical cabin options."),
        ("Cabin Customisation", 4, "Sample materials arrived on time and the approval record was easy for our team to follow."),
        ("Delivery", 4, "Our aircraft status calls are concise, and the production updates have been consistent."),
        ("Delivery", 3, "The acceptance checklist worked well, though a few supporting certificates arrived close to handover."),
        ("Support", 4, "The service desk routed our technical question to the right specialist on the first attempt."),
        ("Support", 5, "The parts planning session identified several common items before our peak travel period."),
    ],
    "Solstice Airways": [
        ("Sales", 4, "The account manager brought the right operations and finance specialists into our planning discussion."),
        ("Sales", 4, "The revised commercial offer arrived on schedule and clearly separated optional services."),
        ("Contracts", 3, "The renewal terms were understandable, but one annex required an additional legal review."),
        ("Contracts", 4, "Our latest contract question was answered with a concise explanation and supporting reference."),
        ("Cabin Customisation", 3, "The cabin review was productive, although the final material samples took longer than expected."),
        ("Cabin Customisation", 4, "We appreciated the clear comparison of seat-fabric options and the documented approval date."),
        ("Delivery", 4, "The aircraft handover team kept our station leads informed throughout the acceptance week."),
        ("Delivery", 3, "The schedule was realistic overall, but a late change to the review agenda created extra coordination."),
        ("Support", 4, "We received a practical troubleshooting guide and a named contact for the remaining questions."),
        ("Support", 3, "The initial acknowledgement was quick; the final response on the component interchange took longer."),
    ],
    "Aether Jet": [
        ("Sales", 5, "The commercial workshop connected our route plans to an aircraft mix that our leadership could evaluate."),
        ("Sales", 4, "The team followed up with a detailed option comparison and answered our finance questions directly."),
        ("Contracts", 4, "The contract review log showed each open item, owner, and agreed next step."),
        ("Contracts", 4, "Procurement received the requested clarification before our internal approval meeting."),
        ("Cabin Customisation", 4, "The design team offered alternatives that met our premium-cabin goals without complicating maintenance."),
        ("Cabin Customisation", 3, "The cabin presentation was strong, but we would benefit from earlier access to production-ready drawings."),
        ("Delivery", 4, "Our inspection team received a useful schedule and knew which specialists would attend each review."),
        ("Delivery", 5, "The handover package was orderly, and our operations group had time to prepare for entry into service."),
        ("Support", 4, "The support coordinator kept our open technical cases visible and arranged a productive review."),
        ("Support", 4, "Parts availability assumptions were explained clearly and aligned with our initial fleet plan."),
    ],
    "Meridian Express": [
        ("Sales", 3, "The proposal covered our route network well, but we needed another meeting to clarify the service options."),
        ("Sales", 4, "The account team shared a helpful fleet-growth model and updated it after our planning change."),
        ("Contracts", 3, "Most clauses were straightforward, although the document version history was difficult to track."),
        ("Contracts", 4, "The contract specialist resolved our invoicing question and sent an updated schedule the same day."),
        ("Cabin Customisation", 3, "We liked the proposed cabin layout, but approval dates should be tied more closely to our internal board calendar."),
        ("Cabin Customisation", 4, "The revised drawings captured our requests and reduced the number of open design comments."),
        ("Delivery", 3, "The production update was useful, but a milestone moved before our team received the formal notice."),
        ("Delivery", 4, "The delivery-readiness checklist helped our station teams coordinate training and equipment."),
        ("Support", 3, "We received an answer to our maintenance question, though the parts reference needed a second clarification."),
        ("Support", 4, "The regional support call gave our engineers a clear path for urgent service requests."),
    ],
    "Borealis Air": [
        ("Sales", 4, "Our account manager brought a clear summary of the regional operating assumptions to the review."),
        ("Sales", 3, "The commercial options were relevant, but we are waiting for an updated comparison of service packages."),
        ("Contracts", 4, "The revised agreement highlighted the differences from the previous draft and saved review time."),
        ("Contracts", 3, "The change-control process is workable, though response times vary between contract contacts."),
        ("Cabin Customisation", 4, "The cabin team gave us enough time to compare durable material choices for our winter operations."),
        ("Cabin Customisation", 3, "We received the requested layout drawings, but one supplier specification was missing from the pack."),
        ("Delivery", 4, "The production visit was well coordinated and gave our technical team confidence in the plan."),
        ("Delivery", 4, "Acceptance actions were closed in a transparent tracker before the final readiness meeting."),
        ("Support", 2, "Spare parts documentation arrived in separate emails, making it difficult to confirm that our library was complete."),
        ("Support", 4, "The support engineer explained the publication revision and helped us update our maintenance records."),
    ],
    "Vanguard Aviation": [
        ("Sales", 3, "The account team understood our growth targets, but a requested price scenario was delivered late."),
        ("Sales", 4, "The route and fleet discussion was constructive and gave our leadership a useful next-step plan."),
        ("Contracts", 3, "We closed the main commercial points, although the local-language summary needed another review."),
        ("Contracts", 4, "Our legal team received a prompt response and a clean copy of the agreed contract changes."),
        ("Cabin Customisation", 3, "The cabin options suited our market, but we need firmer dates for approving customer-facing finishes."),
        ("Cabin Customisation", 4, "The design review recorded each decision and gave our team clear owners for the remaining choices."),
        ("Delivery", 3, "The aircraft timeline is credible, but dependencies on training equipment should be more visible."),
        ("Delivery", 4, "Our acceptance planning call covered the handover sequence and regional support contacts."),
        ("Support", 3, "The service desk resolved our routine request, though the supporting parts list was not attached initially."),
        ("Support", 4, "The regional support contact followed through and kept our operations team updated."),
    ],
}

COMMERCIAL_DOCUMENTS = {
    "cabin_customisation_catalog.md": """# Cabin Customisation Catalog

## Purpose and Scope

This fictional catalog supports commercial-planning exercises for the three aircraft family classes in this workspace. All lead times and options are illustrative planning assumptions, not a quotation or production commitment.

## Cabin Zones

- **Passenger seating:** seat family, upholstery palette, trim, and economy or premium-zone layout.
- **Monuments:** galley and lavatory arrangement, monument finishes, and customer-facing work surfaces.
- **Cabin ambience:** lighting scenes, carpet palette, sidewall finish, and coordinated visual themes.
- **Connectivity and power:** passenger power and connectivity options, subject to aircraft configuration and certification review.

## Option Packages

### Essential

Standard cabin materials, a published seat-map choice, and a single coordinated review of customer branding. Use when schedule stability and a low-change configuration are the primary objectives.

### Enhanced

Expanded finish selections, additional passenger-zone choices, and one consolidated design workshop. This package requires an agreed configuration baseline before supplier-specific work begins.

### Signature

Broader cabin-zone coordination and additional mock-up reviews. Treat every non-catalogue request as a change until technical feasibility, certification impact, supplier capacity, and commercial terms are approved.

## Planning Lead Times

For simulation purposes, assume 12-16 weeks from a complete customer brief to an approved standard cabin baseline. Enhanced options typically require 16-24 weeks; Signature requests typically require 24-36 weeks. These ranges exclude late engineering changes, supplier constraints, and customer approval delays.

## Approval Workflow

1. Confirm aircraft family, delivery position, and the customer decision-maker.
2. Record requirements in a single cabin brief with drawings, material references, and revision dates.
3. Complete feasibility and certification screening before treating a request as selectable.
4. Consolidate customer comments into one review log with an owner and due date for each item.
5. Obtain written approval of the baseline and identify any changes that affect price or schedule.

## Change Control

Requests received after baseline approval must include the affected zone, requested revision, operational reason, and required decision date. The account team should return a documented impact assessment before promising a delivery-date or price effect.
""",
    "commercial_sla_and_penalties.md": """# Commercial Service Levels and Penalties

## Purpose and Assumptions

This fictional reference is for commercial-training scenarios only. The example response targets and service credits below are illustrative and do not state actual manufacturer commitments or legal terms.

## Case Acknowledgement Targets

| Priority | Example scope | Acknowledgement target |
|---|---|---:|
| Critical | Grounded aircraft or immediate operational safety concern | 1 hour, 24/7 routing |
| High | Fleet-impacting technical or delivery issue without an immediate safety concern | 4 business hours |
| Standard | Routine commercial, documentation, or planning request | 2 business days |

Acknowledgement confirms ownership and next update timing; it is not a guaranteed resolution time. Safety and airworthiness matters must follow the operator's approved procedures and the applicable authority process.

## Delivery Milestone Governance

Each milestone should have a named owner, an agreed evidence package, a baseline date, and a documented change history. A forecast movement should be communicated with the cause, affected downstream activities, recovery options, and the date of the next update.

## Illustrative Service Credits

For a training exercise, a contract may model a capped service credit for a supplier-controlled missed response or documentation milestone. Any credit should be calculated only from defined eligible fees, exclude customer-caused delay and agreed force-majeure events, and be subject to a negotiated aggregate cap.

## Liquidated-Damages Review

Delivery liquidated damages, if negotiated, require a defined contractual delivery event, grace period, daily calculation method, exclusions, mitigation duty, and cap. The commercial and legal teams must review the full clause; a service credit must not be presented as interchangeable with liquidated damages.

## Escalation and Reporting

1. Case owner records the issue, operational impact, priority, and next update.
2. Account manager coordinates technical, delivery, and contract stakeholders.
3. Unresolved high-priority items are escalated to the named program leads.
4. Monthly review reports open actions, ageing, milestone movement, and customer decisions required.
""",
    "aircraft_family_positioning.md": """# Aircraft Family Positioning

## Purpose

This fictional positioning guide helps an account team structure discovery conversations. It uses aircraft family classes rather than manufacturer performance guarantees; actual range, capacity, payload, and operating economics depend on a specific configuration and mission.

## A220-Class

### Typical Mission Fit

Regional and thinner short-to-medium-haul markets where right-sizing capacity and opening new frequencies are central to the business case.

### Commercial Discussion

Explore city-pair demand, frequency strategy, airport constraints, cabin mix, and commonality with the customer's existing fleet. Compare trip economics using customer-approved assumptions rather than a generic cost-per-seat claim.

### Key Trade-Offs

Confirm capacity headroom for seasonal peaks, cargo needs, crew and maintenance plans, and the customer's long-term fleet-growth strategy.

## A320neo-Class

### Typical Mission Fit

High-frequency short-to-medium-haul networks that value a broad mission set and fleet-scale operating flexibility.

### Commercial Discussion

Review network coverage, current fleet transition timing, cabin-density choices, delivery positions, and the cost of operating multiple subfleets.

### Key Trade-Offs

Validate airport and route requirements, customer-selected cabin layout, training capacity, and the timing of legacy-aircraft retirement.

## A350-Class

### Typical Mission Fit

Long-haul networks requiring widebody capacity, passenger comfort choices, and planning for extended-range operations.

### Commercial Discussion

Assess route maturity, traffic seasonality, cabin segmentation, cargo contribution, and the support plan required for long-haul entry into service.

### Key Trade-Offs

Stress-test demand and utilization assumptions, consider widebody infrastructure and crew readiness, and align cabin decisions with longer supplier lead times.

## Discovery Checklist

Document target routes, demand scenarios, fleet replacement dates, cabin objectives, financing constraints, delivery timing, and customer success measures. Keep mission assumptions consistent when comparing family classes.
""",
    "pricing_and_concessions_guidelines.md": """# Pricing and Concessions Guidelines

## Purpose and Guardrails

This fictional guide is a training aid, not a price list or offer. Every customer proposal requires current approved pricing, an authorized commercial owner, and review under the applicable delegation-of-authority policy.

## Build the Value Case

Separate aircraft, optional equipment, services, training, and support assumptions. State the configuration basis, delivery position, currency, validity date, payment milestones, and exclusions. Use customer-approved operating inputs for any economic comparison.

## Concession Principles

- Request a documented customer commitment, such as order scope, timing, or a defined reference activity.
- Prefer operationally meaningful value, including schedule alignment or bundled support, where capacity and policy allow.
- Evaluate the full deal contribution and lifecycle implications before offering a price adjustment.
- Record the concession owner, approval evidence, expiry date, and any conditions in the deal file.
- Do not imply that an illustrative value is approved or transferable to another order.

## Approval Path

1. Account manager documents the customer request and competitive context.
2. Finance validates price basis, payment profile, currency exposure, and contribution.
3. Program and delivery teams confirm capacity and schedule feasibility for non-price commitments.
4. The authorized approver records the decision before the proposal is released.
5. Contracting ensures the signed terms match the approved commercial case.

## Change and Exception Handling

A change in quantity, configuration, delivery position, payment schedule, or bundled service invalidates the prior approval when it changes the deal economics or risk. Recalculate and resubmit material changes; never backdate an approval.

## Proposal Hygiene

Use a unique proposal revision, identify assumptions and open customer decisions, and remove superseded pricing pages from the release package. Keep internal approval records separate from customer-facing documents.
""",
}


def create_database(database_path: Path) -> None:
    database_path.unlink(missing_ok=True)
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE airlines (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                region TEXT NOT NULL,
                fleet_size INTEGER NOT NULL CHECK (fleet_size > 0),
                nps_score INTEGER NOT NULL CHECK (nps_score BETWEEN 0 AND 100),
                account_tier TEXT NOT NULL CHECK (account_tier IN ('Tier-1', 'Tier-2')),
                account_manager TEXT NOT NULL
            );

            CREATE TABLE aircraft_orders (
                order_id INTEGER PRIMARY KEY,
                airline_id INTEGER NOT NULL REFERENCES airlines(id),
                aircraft_family TEXT NOT NULL
                    CHECK (aircraft_family IN ('A320neo-class', 'A350-class', 'A220-class')),
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                delivery_year INTEGER NOT NULL,
                status TEXT NOT NULL
                    CHECK (status IN ('Delivered', 'In Production', 'Backlog')),
                contract_value_musd REAL NOT NULL CHECK (contract_value_musd > 0)
            );

            CREATE TABLE contract_milestones (
                milestone_id INTEGER PRIMARY KEY,
                airline_id INTEGER NOT NULL REFERENCES airlines(id),
                topic TEXT NOT NULL,
                due_date TEXT NOT NULL,
                risk_level TEXT NOT NULL CHECK (risk_level IN ('Low', 'Medium', 'High'))
            );
            """
        )
        connection.executemany(
            """
            INSERT INTO airlines
                (id, name, region, fleet_size, nps_score, account_tier, account_manager)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            AIRLINES,
        )
        connection.executemany(
            """
            INSERT INTO aircraft_orders
                (order_id, airline_id, aircraft_family, quantity, delivery_year, status,
                 contract_value_musd)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            AIRCRAFT_ORDERS,
        )
        connection.executemany(
            """
            INSERT INTO contract_milestones
                (milestone_id, airline_id, topic, due_date, risk_level)
            VALUES (?, ?, ?, ?, ?)
            """,
            CONTRACT_MILESTONES,
        )
        connection.commit()


def create_surveys(csv_path: Path) -> None:
    rng = random.Random(SEED)
    first_date = date(2025, 1, 1)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["survey_id", "airline_name", "date", "domain", "rating", "verbatim_text"]
        )

        survey_id = 1
        for airline_name, responses in SURVEY_RESPONSES.items():
            for domain, rating, verbatim_text in responses:
                survey_date = first_date + timedelta(days=rng.randint(0, 660))
                writer.writerow(
                    [
                        f"SUR-{survey_id:04d}",
                        airline_name,
                        survey_date.isoformat(),
                        domain,
                        rating,
                        verbatim_text,
                    ]
                )
                survey_id += 1


def create_commercial_documents() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, content in COMMERCIAL_DOCUMENTS.items():
        (DOCS_DIR / filename).write_text(content, encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    create_database(DATA_DIR / "crm_sales.db")
    create_surveys(DATA_DIR / "customer_surveys.csv")
    create_commercial_documents()
    print(
        "Generated CRM database, 80 customer surveys, and 4 commercial documents "
        f"under {DATA_DIR}."
    )


if __name__ == "__main__":
    main()
