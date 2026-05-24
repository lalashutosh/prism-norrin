"""
prompts/analysis_prompts.py
────────────────────────────
Raw prompt templates for the analysis agent.

Pure string constants — no logic, no imports, no formatting.
The analysis agent's intelligence functions consume these templates and
fill in the {placeholders} with serialised data at call time.
"""

# ── System prompt ─────────────────────────────────────────────────────────────

ANALYSIS_SYSTEM_PROMPT = """\
You are an expert EU AI Act legal analyst assisting with compliance assessment.
You reason carefully over retrieved legislative text and official guidance to
produce structured, evidence-based findings about a submitted AI use case.

CRITICAL RULES
──────────────
1. Cite only chunk_ids that appear in the RETRIEVED CHUNKS section below.
   Never invent chunk_ids.
2. Label each claim correctly:
   - RETRIEVED : claim is directly supported by legislation / official_guidance chunks
   - FACT      : claim is an explicit statement from the uploaded use-case document
   - ASSUMPTION: claim is reasonably inferred but not directly evidenced
   - UNCERTAIN : evidence is absent, contradictory, or too ambiguous to use
3. Only use confidence HIGH when multiple authoritative chunks directly support
   the claim.  Use LOW or INSUFFICIENT when evidence is weak or absent.
4. Output ONLY valid JSON — no prose outside the JSON object.
5. If you cannot make a finding, set confidence to "INSUFFICIENT" and
   explain why in the summary field.
"""

# ── Per-dimension instruction blocks ─────────────────────────────────────────
#
# Each value is inserted into the user-turn prompt by build_dimension_prompt().
# Placeholders:
#   {facts_text}      — serialised FactSection (use case name, industry, capabilities, etc.)
#   {chunks_text}     — numbered list of retrieved chunks with their chunk_ids
#   {refined_context} — additional context from a synthesis loop-back (may be empty)
#
# Each template ends with a required JSON structure.  The JSON schemas use
# double braces ({{, }}) to escape the curly-bracket characters so that Python's
# str.format() does not interpret them as placeholder slots.
#
# The claim_id prefixes (def_, risk_, proh_, trans_, role_, gov_) are
# dimension-specific to guarantee uniqueness across the session.

DIMENSION_PROMPTS: dict[str, str] = {

    "definition_check": """\
TASK: AI SYSTEM DEFINITION CHECK (Article 3(1) + Annex I, EU AI Act)

Determine whether the described use case constitutes an "AI system" as defined
in Article 3(1) of the EU AI Act and clarified in Annex I.

An AI system is defined as: "a machine-based system designed to operate with
varying levels of autonomy, that may exhibit adaptiveness after deployment,
and that, for explicit or implicit objectives, infers, from the input it
receives, how to generate outputs such as predictions, recommendations,
decisions or content that can influence real or virtual environments."

Key Annex I techniques: machine learning (supervised, unsupervised,
reinforcement), logic- and knowledge-based approaches, statistical approaches.

USE CASE FACTS:
{facts_text}

RETRIEVED CHUNKS:
{chunks_text}

{refined_context}

Respond with ONLY this JSON structure:
{{
  "dimension_id": "definition_check",
  "claims": [
    {{
      "claim_id": "def_<n>",
      "text": "<claim statement>",
      "label": "RETRIEVED|FACT|ASSUMPTION|UNCERTAIN",
      "confidence": "HIGH|MEDIUM|LOW|INSUFFICIENT",
      "chunk_ids": ["<chunk_id>", ...],
      "is_weak": false,
      "weak_reason": null
    }}
  ],
  "confidence": "HIGH|MEDIUM|LOW|INSUFFICIENT",
  "summary": "<concise assessment>",
  "is_ai_system": true|false|null
}}
""",

    "risk_classification": """\
TASK: RISK CLASSIFICATION (Articles 5, 6, Annex III, EU AI Act)

Classify the AI system's risk tier:
  UNACCEPTABLE — prohibited under Article 5
  HIGH         — listed in Annex III or falls under Article 6 criteria
  LIMITED      — subject only to transparency obligations (Article 50)
  MINIMAL      — no specific obligations beyond general product safety

Annex III categories (high-risk): biometric identification, critical
infrastructure, education/training, employment, essential services, law
enforcement, migration/asylum, administration of justice.

Consider: is the system used in a critical area? Does it make decisions with
significant impact on persons? Is it safety-critical?

USE CASE FACTS:
{facts_text}

RETRIEVED CHUNKS:
{chunks_text}

{refined_context}

Respond with ONLY this JSON structure:
{{
  "dimension_id": "risk_classification",
  "claims": [
    {{
      "claim_id": "risk_<n>",
      "text": "<claim statement>",
      "label": "RETRIEVED|FACT|ASSUMPTION|UNCERTAIN",
      "confidence": "HIGH|MEDIUM|LOW|INSUFFICIENT",
      "chunk_ids": ["<chunk_id>", ...],
      "is_weak": false,
      "weak_reason": null
    }}
  ],
  "confidence": "HIGH|MEDIUM|LOW|INSUFFICIENT",
  "summary": "<concise assessment>",
  "risk_level": "unacceptable|high|limited|minimal|unknown"
}}
""",

    "prohibited_practices": """\
TASK: PROHIBITED PRACTICES CHECK (Article 5, EU AI Act)

Determine whether the use case triggers any Article 5 prohibition:
  (a) subliminal techniques to distort behaviour causing harm
  (b) exploiting vulnerabilities of specific groups
  (c) social scoring by public authorities
  (d) real-time remote biometric identification in public spaces (with exceptions)
  (e) emotion recognition in workplace/education
  (f) biometric categorisation inferring sensitive attributes
  (g) predictive policing based solely on profiling
  (h) untargeted facial-image scraping

USE CASE FACTS:
{facts_text}

RETRIEVED CHUNKS:
{chunks_text}

{refined_context}

Respond with ONLY this JSON structure:
{{
  "dimension_id": "prohibited_practices",
  "claims": [
    {{
      "claim_id": "proh_<n>",
      "text": "<claim statement>",
      "label": "RETRIEVED|FACT|ASSUMPTION|UNCERTAIN",
      "confidence": "HIGH|MEDIUM|LOW|INSUFFICIENT",
      "chunk_ids": ["<chunk_id>", ...],
      "is_weak": false,
      "weak_reason": null
    }}
  ],
  "confidence": "HIGH|MEDIUM|LOW|INSUFFICIENT",
  "summary": "<concise assessment>",
  "triggered_articles": ["Article 5(1)(a)", ...],
  "prohibited": true|false|null
}}
""",

    "transparency": """\
TASK: TRANSPARENCY AND GPAI OBLIGATIONS (Articles 13, 50, 52–56, EU AI Act)

Determine what transparency and disclosure obligations apply:
  - High-risk systems: Article 13 (transparency to deployers),
    Article 14 (human oversight)
  - Interaction with natural persons: Article 50(1) notification duty
  - Emotion recognition / biometric categorisation: Article 50(3)
  - Synthetic content (deepfakes): Article 50(4) labelling
  - GPAI models: Articles 52–56 (transparency, copyright, systemic risk)
  - General: any applicable labelling or disclosure requirements

APPLICABILITY GUIDANCE — read before analysing
───────────────────────────────────────────────
• Article 13 applies ONLY to HIGH-RISK AI systems (Annex III / Article 6).
  If the system is classified as minimal or limited risk, Article 13 does
  NOT apply.  State this conclusion with HIGH confidence and cite the article.
• Article 50(1) (notification) applies ONLY when the system directly and
  visibly interacts with natural persons in real time.  B2B tools and
  industrial equipment deployed without human-facing interfaces do NOT trigger
  Article 50(1).  State this conclusion with HIGH confidence.
• Articles 52–56 (GPAI) apply ONLY to foundation models / general-purpose
  AI trained on broad data.  A domain-specific ML model trained on proprietary
  sensor data is NOT a GPAI model.  State this conclusion with HIGH confidence.
• If you can determine that ALL obligations above do not apply, produce a
  claim for each one (stating "not applicable") with confidence HIGH and
  label RETRIEVED (if the corpus provides direct text) or ASSUMPTION (if
  you are inferring from the system description alone).
• Avoid UNCERTAIN for clear non-applicability — "UNCERTAIN" should only be
  used when the use-case description is genuinely ambiguous about a feature
  that would trigger an obligation.

USE CASE FACTS:
{facts_text}

RETRIEVED CHUNKS:
{chunks_text}

{refined_context}

Respond with ONLY this JSON structure:
{{
  "dimension_id": "transparency",
  "claims": [
    {{
      "claim_id": "trans_<n>",
      "text": "<claim statement>",
      "label": "RETRIEVED|FACT|ASSUMPTION|UNCERTAIN",
      "confidence": "HIGH|MEDIUM|LOW|INSUFFICIENT",
      "chunk_ids": ["<chunk_id>", ...],
      "is_weak": false,
      "weak_reason": null
    }}
  ],
  "confidence": "HIGH|MEDIUM|LOW|INSUFFICIENT",
  "summary": "<concise assessment>",
  "applies_to_gpai": true|false,
  "labelling_required": true|false,
  "notification_required": true|false
}}
""",

    "roles": """\
TASK: ROLES DETERMINATION (Articles 3, 25, 26, EU AI Act)

Determine whether the entity is a Provider, Deployer, or both.

Provider: places an AI system on the market or puts it into service under
  own name/brand, or substantially modifies a high-risk system (Article 25).
Deployer: uses an AI system under its authority for a professional purpose
  (Article 3(4)).

Consider: who developed the system? Who deploys it? Are there third-party
vendors? Does the entity modify the system before use?

USE CASE FACTS:
{facts_text}

RETRIEVED CHUNKS:
{chunks_text}

{refined_context}

Respond with ONLY this JSON structure:
{{
  "dimension_id": "roles",
  "claims": [
    {{
      "claim_id": "role_<n>",
      "text": "<claim statement>",
      "label": "RETRIEVED|FACT|ASSUMPTION|UNCERTAIN",
      "confidence": "HIGH|MEDIUM|LOW|INSUFFICIENT",
      "chunk_ids": ["<chunk_id>", ...],
      "is_weak": false,
      "weak_reason": null
    }}
  ],
  "confidence": "HIGH|MEDIUM|LOW|INSUFFICIENT",
  "summary": "<concise assessment>",
  "is_provider": true|false,
  "is_deployer": true|false,
  "is_both": true|false
}}
""",

    "governance": """\
TASK: GOVERNANCE OBSERVATIONS (Articles 9–17, 61–63, EU AI Act)

Identify applicable governance obligations:
  - Article 9: risk management system
  - Article 10: data governance
  - Article 11: technical documentation
  - Article 12: record-keeping
  - Article 13: transparency and information provision
  - Article 14: human oversight measures
  - Articles 16–17: quality management system (providers)
  - Articles 26: deployer obligations
  - Articles 61–63: post-market monitoring, incident reporting

USE CASE FACTS:
{facts_text}

RETRIEVED CHUNKS:
{chunks_text}

{refined_context}

Respond with ONLY this JSON structure:
{{
  "dimension_id": "governance",
  "claims": [
    {{
      "claim_id": "gov_<n>",
      "text": "<claim statement>",
      "label": "RETRIEVED|FACT|ASSUMPTION|UNCERTAIN",
      "confidence": "HIGH|MEDIUM|LOW|INSUFFICIENT",
      "chunk_ids": ["<chunk_id>", ...],
      "is_weak": false,
      "weak_reason": null
    }}
  ],
  "confidence": "HIGH|MEDIUM|LOW|INSUFFICIENT",
  "summary": "<concise assessment>",
  "documentation_required": true|false,
  "oversight_required": true|false,
  "monitoring_required": true|false
}}
""",
}

# ── Retrieval query templates ─────────────────────────────────────────────────
#
# Used by formulate_retrieval_query() to build targeted queries.
# The {context} placeholder is filled with a truncated string of
# (use_case_name, industry, deployment_context) to steer retrieval
# toward legislation relevant to the specific use case, not generic text.

DIMENSION_RETRIEVAL_TEMPLATES: dict[str, str] = {
    "definition_check":     "EU AI Act Article 3 definition AI system Annex I {context}",
    "risk_classification":  "EU AI Act risk classification Annex III Article 6 high-risk {context}",
    "prohibited_practices": "EU AI Act Article 5 prohibited practices {context}",
    "transparency":         "EU AI Act Article 50 transparency obligations GPAI {context}",
    "roles":                "EU AI Act provider deployer Article 25 26 obligations {context}",
    "governance":           "EU AI Act governance documentation Article 9 10 11 14 {context}",
}

# Keywords used for evidence sufficiency checks and chunk filtering.
# check_evidence_sufficiency() requires at least one keyword match plus
# one authoritative chunk to consider evidence sufficient for a dimension.
# _filter_chunks_for_dimension() narrows the chunk list before building
# the LLM prompt so each dimension receives only its relevant context.
DIMENSION_KEYWORDS: dict[str, list[str]] = {
    "definition_check": [
        "ai system", "artificial intelligence", "machine learning", "model",
        "algorithm", "automated", "annex i", "article 3",
    ],
    "risk_classification": [
        "risk", "annex iii", "high-risk", "article 6", "safety", "harm",
        "critical infrastructure", "biometric", "employment",
    ],
    "prohibited_practices": [
        "prohibited", "article 5", "subliminal", "manipulation", "social scoring",
        "biometric identification", "real-time", "emotion recognition",
    ],
    "transparency": [
        "transparency", "article 50", "article 13", "disclosure", "inform",
        "notification", "labelling", "gpai", "general-purpose",
    ],
    "roles": [
        "provider", "deployer", "operator", "article 25", "article 26",
        "places on the market", "puts into service", "importer",
    ],
    "governance": [
        "documentation", "article 9", "article 11", "article 14", "conformity",
        "technical file", "quality management", "oversight", "monitoring",
    ],
}
