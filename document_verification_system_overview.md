# Document Verification & OCR System: Project Overview

## 1. The Problem This Solves

The platform needs users to upload three documents: an **ID document**, a **proof of residency/address certificate**, and an **academic transcript**. It needs to automatically confirm that:

1. Each upload is actually the document type it claims to be
2. The key details on the document (name, ID number, address/region, etc.) match the data the user already provided when registering

Doing this manually for every user doesn't scale. Doing it with a naive "AI reads the image and tells you if it's fine" approach isn't accurate or trustworthy enough for something like identity verification. A wrong auto-approval or auto-rejection has real consequences for users. So the system is built around one central design principle:

> **Automate the clear-cut majority of cases. Route anything uncertain to a human. Never let the system guess with confidence it doesn't actually have.**

That single principle shapes almost every architectural decision below.

---

## 2. High-Level Flow

```
User uploads 3 images (ID, Residency Proof, Transcript) via the website
                    ↓
        Website backend calls this system's API
                    ↓
   For each document, concurrently:
   Preprocess, then OCR, then Classify, then Extract, then Validate Format, then Cross-Check, then Verdict
                    ↓
        Combine into one overall verdict
                    ↓
        Return structured result to the website
```

The three documents are processed at the same time, not one after another. This keeps response time down, which matters since this is a "minimal resources" deployment.

---

## 3. The Pipeline, Stage by Stage

### Stage 1: Preprocessing & Quality Gate
Before anything else, each uploaded image is auto-oriented (corrects phone camera rotation) and checked for basic quality:
- **Blur detection**, a standard computer-vision technique called Laplacian variance. A too-blurry photo is rejected immediately, before wasting time running OCR on something unreadable.
- **Resolution and file-size checks**, which catch obviously broken or empty uploads.

This exists so that garbage-in doesn't silently become garbage-out three steps later. A bad photo gets a clear, immediate "please re-upload, image quality too low" instead of confusing the rest of the pipeline.

### Stage 2: OCR (Optical Character Recognition)
The system uses **PaddleOCR**, a CPU-friendly, accurate open-source OCR engine, to read text out of the image. It doesn't just get a blob of text. It gets each recognized word along with its position on the page (bounding box) and how confident the OCR engine is about that specific word. This positional and confidence data is what makes the next two stages possible.

### Stage 3: Document Classification
The system decides whether a document is an ID document, a residency/address certificate, or a transcript. This is done with **keyword matching**, not AI guessing. Each document type has known telltale phrases ("Identity Card," "Residency Certificate," "Transcript"/"Marksheet," etc.), and ID numbers have a recognizable digit pattern. If nothing matches confidently, the document is marked `unclassified` and routed for human review rather than guessed at.

**Why not use an AI model to classify?** Because these are fixed-format official/institutional documents. A keyword rule is faster, uses no extra resources, and, importantly, is *deterministic*: it either matches or it doesn't, with no risk of an AI model hallucinating a category.

### Stage 4: Field Extraction (the core of the system)
This is where the system pulls out actual data: the ID number, full name, region, roll number, and so on.

**For the ID and residency documents** (fixed official layouts), extraction is **100% deterministic, with no AI model involved.** The technique is called **label-anchored extraction**: the system finds a label on the document (like "Name" or "ID No.") using the OCR results, then looks at the text positioned near that label (to the right or below it, using the bounding box coordinates from Stage 2) and takes that as the value. This mirrors exactly how a human reads the form.

**For transcripts**, which vary in layout between different institutions, the same label-anchor approach is tried first. Only if too few fields are found does the system fall back to a small local AI language model (Phi-3-mini, running quantized on CPU) to parse the harder cases, and even then, the AI's output is validated against a strict schema before being trusted. If the AI's output doesn't fit the expected structure, it's discarded rather than used.

**Why avoid AI for the ID/residency extraction?** Because an AI language model reading OCR text and "guessing" a long numeric ID can produce a plausible-looking wrong answer with full confidence, and there's no way to tell that happened. Deterministic anchor-based extraction on a fixed layout is both more accurate and cheaper to run than AI-based extraction for this specific use case. The AI is reserved only for the one document type where it earns its cost.

Every extracted field carries:
- The extracted **value**
- Its **source location** on the image (bounding box), for auditability
- An **extraction confidence score**, derived from the real OCR confidence of the matched text, not a made-up number

### Stage 5: Format Validation
Before comparing anything against the user's stored data, the system checks that what it extracted is even well-formed: is the ID number the right length? Are name fields a sane length? Malformed extraction is caught here, before it ever reaches the comparison stage. This prevents comparing garbage against the database and getting a nonsensical result.

### Stage 6: Cross-Reference (comparing against the user's stored data)
The extracted fields are compared against the data the user supplied at registration (currently stored in a reference CSV, designed to be swapped for a real database later):

- **ID number**: exact match required, since it's a precise numeric identifier with no room for "close enough"
- **Names and region**: **fuzzy matching** (via the `rapidfuzz` library), because OCR and real-world spelling variation (e.g. "Muhammad" vs "Mohammad") mean exact string matching would wrongly reject legitimate matches. A similarity score (0 to 100%) is calculated for each field.

Each document type has clearly defined required fields that must match:

| Document | Required to match |
|---|---|
| ID document | ID number and full name |
| Residency proof | Full name and region |
| Transcript | Student name |

### Stage 7: Verdict (the decision)
This is the most important design choice in the whole system. Instead of a simple pass/fail, every document gets one of three states:

- **`verified`**: all required fields matched with high confidence, extraction was reliable
- **`needs_review`**: something was ambiguous, such as low OCR/extraction confidence, a fuzzy match score that landed in an uncertain middle band, uncertain classification, or malformed data. This is sent to a human review queue, not auto-decided either way.
- **`rejected`**: the extracted data clearly and confidently does not match (e.g. the ID number is simply a different number), with high confidence that the extraction itself was accurate.

The three individual document verdicts are then combined into one overall verdict for the user. If any document is `rejected`, the overall result is `rejected`. If any needs review, the overall result is `needs_review`. Only if all three are `verified` is the overall result `verified`.

**Why this matters for "very high accuracy":** A binary system is forced to guess on every ambiguous case, and every guess has some error rate. A three-state system only auto-decides the clear cases and defers the ambiguous ones to a human. That is what actually makes a claim of high accuracy honest and defensible for identity verification, rather than aspirational.

### Stage 8: Audit Trail
Every `needs_review` case is saved with full supporting evidence (extracted values, match scores, confidence numbers, reasons) so a human reviewer can see exactly why the system was unsure, not just a bare "unclear" label.

---

## 4. The API

The website's backend (not the browser directly, for security) calls a single endpoint:

```
POST /verify
Headers: X-API-Key: <secret>
Form data: user_id, id_img, residency_img, transcript_img
```

It returns a structured JSON response containing the overall verdict, and, for each document, its classification, every extracted field with its confidence and source location, the cross-reference match details, and the verdict with reasons.

Supporting endpoints/features:
- **`GET /health`**, for uptime monitoring
- **API key authentication**, required on every request
- **Rate limiting**, to prevent abuse
- **Structured error responses**, so bad input never leaks a raw stack trace back to the caller

---

## 5. Why It's "Minimal Resources"

- **CPU-only** throughout. No GPU required anywhere in the pipeline.
- The AI language model is **lazy-loaded** (only pulled into memory when a transcript genuinely needs it) and **quantized** (compressed to run efficiently on CPU), so most requests (ID, residency, and the majority of transcripts) never touch it at all.
- Deterministic extraction for two of the three document types is inherently faster and lighter than running an AI model on every request.
- The whole system is designed to run comfortably on a modest CPU VPS (roughly 4 to 8GB RAM), which keeps hosting costs low.

---

## 6. Security & Data Handling

Since this system processes sensitive personal identity data:
- All processing happens **in memory**. Uploaded images are never written to disk, so there's no leftover file to worry about deleting.
- **PII is redacted in logs.** ID numbers and names are partially masked (e.g. showing only the first few and last few digits/letters) before anything is written to the log file, so a log leak doesn't equal a data leak.
- The API requires authentication and is intended to be called server-to-server from the website's backend, never exposed directly to end-user browsers.
- CORS and the API key must be locked down to production values before launch. This is a known pre-launch checklist item, not yet finalized.

---

## 7. What's Deliberately Not Fully Automated Yet

Being transparent about current scope, since this is worth mentioning honestly in a presentation:
- The reference database is currently a CSV file, designed to be swapped for a real database as the platform scales.
- `needs_review` cases are stored for human review, but the actual admin interface for reviewers to act on them is a next step, not yet built.
- Image encryption-at-rest isn't implemented. It's currently mitigated by not persisting images to disk at all, but this becomes more important once the review queue starts retaining evidence longer-term.

---

## 8. Summary, in One Paragraph

The system takes three uploaded documents, runs each through a quality check, reads the text with OCR, identifies what kind of document it is, and pulls out the key fields using a deterministic, position-based method for the two fixed-format official documents, with a lightweight local AI model reserved only for the more variable transcript format. It then checks that the extracted data is well-formed, fuzzy-matches it against the user's registered information, and produces a three-way verdict: verified, needs review, or rejected, with full supporting evidence for every decision. The result is a system that's fast and cheap to run, and structured so that its claim to high accuracy is actually earned. It only auto-decides when it has real confidence, and hands ambiguous cases to a person rather than guessing.
