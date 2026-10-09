"""career_coach.py - RAG + chains + output parsers for the AI Career Coach.
Generated from the notebook; imported by app.py (Streamlit)."""

import os
import re
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BASE_DIR = Path(os.environ.get(
    "COACH_BASE_DIR", "/kaggle/working" if os.path.exists("/kaggle/working") else "."))
KB_DIR = BASE_DIR / "knowledge_base"          # job descriptions / roadmaps / industry docs
INDEX_DIR = BASE_DIR / "faiss_career_index"   # saved FAISS index
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = os.environ.get("COACH_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
MAX_CV_CHARS = 6000                           # keep prompts short enough for a 7B model

# folder name -> metadata value used for filtered retrieval
DOC_TYPES = {
    "job_descriptions": "job_description",
    "learning_roadmaps": "learning_roadmap",
    "industry_requirements": "industry_requirement",
}


# Filled by load_llm() (inside the notebook they are the model/tokenizer loaded earlier)
tokenizer = None
model = None


def load_llm(model_name=MODEL_NAME, use_4bit=False):
    """Load tokenizer + model into the module globals `tokenizer` / `model`.
    (In the notebook the model is already loaded, so this is only used by the Streamlit app.)"""
    global tokenizer, model
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"device_map": "auto"}
    if use_4bit and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def llm_generate(prompt, max_new_tokens=700):
    """Greedy generation that returns ONLY the new text (the lab's generate_text echoed the prompt too).
    Greedy decoding keeps JSON output stable and repeatable."""
    import torch
    if model is None:
        load_llm()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_kb_documents(kb_dir=KB_DIR):
    """Read every .md / .txt / .pdf under knowledge_base/<doc-type folder>/ into LangChain Documents.
    A file may start with a line 'Role: <role name>'; otherwise the file name is used as the role."""
    docs = []
    for folder, doc_type in DOC_TYPES.items():
        for path in sorted((Path(kb_dir) / folder).glob("*")):
            suffix = path.suffix.lower()
            if suffix in {".md", ".txt"}:
                text = path.read_text(encoding="utf-8")
            elif suffix == ".pdf":
                text = "\n".join(p.page_content for p in PyPDFLoader(str(path)).load())
            else:
                continue
            match = re.search(r"^Role:\s*(.+)$", text, re.MULTILINE)
            role = match.group(1).strip() if match else path.stem.replace("_", " ").title()
            docs.append(Document(
                page_content=text,
                metadata={"role": role, "doc_type": doc_type, "source": path.name},
            ))
    return docs


def list_roles(kb_dir=KB_DIR):
    """All roles that have a job description in the knowledge base."""
    return sorted({d.metadata["role"] for d in load_kb_documents(kb_dir)
                   if d.metadata["doc_type"] == "job_description"})


_VECTORSTORE = None


def get_vectorstore(rebuild=False):
    """Load the FAISS index from disk (or build it from the knowledge base)."""
    global _VECTORSTORE
    if _VECTORSTORE is not None and not rebuild:
        return _VECTORSTORE
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
    if INDEX_DIR.exists() and not rebuild:
        _VECTORSTORE = FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
        return _VECTORSTORE
    docs = load_kb_documents()
    if not docs:
        raise FileNotFoundError(
            f"No knowledge-base files found in {KB_DIR}. Run the 'Build the knowledge base' "
            "cells in the notebook first.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    _VECTORSTORE = FAISS.from_documents(chunks, embeddings)
    _VECTORSTORE.save_local(str(INDEX_DIR))
    return _VECTORSTORE


def retrieve(query, doc_type, role=None, k=4):
    """Semantic search restricted to one document type (and optionally one role)."""
    metadata_filter = {"doc_type": doc_type}
    if role:
        metadata_filter["role"] = role
    return get_vectorstore().similarity_search(query, k=k, filter=metadata_filter, fetch_k=500)


def format_docs(docs):
    return "\n\n".join(f"[{d.metadata['role']} | {d.metadata['doc_type']}]\n{d.page_content}"
                       for d in docs)


def resolve_role(text):
    """Map free text ('senior data scientist', 'NLP engineer') to the closest role in the knowledge base."""
    roles = list_roles()
    wanted = text.strip().lower()
    for role in roles:                                   # exact match
        if role.lower() == wanted:
            return role
    for role in roles:                                   # containment match
        if len(wanted) >= 4 and (role.lower() in wanted or wanted in role.lower()):
            return role
    hit = get_vectorstore().similarity_search(           # semantic fallback
        text, k=1, filter={"doc_type": "job_description"}, fetch_k=500)
    return hit[0].metadata["role"]


def best_fit_roles(query, k=3):
    """Roles whose job descriptions are semantically closest to a skills description."""
    docs = get_vectorstore().similarity_search(
        query, k=30, filter={"doc_type": "job_description"}, fetch_k=500)
    roles = []
    for d in docs:
        if d.metadata["role"] not in roles:
            roles.append(d.metadata["role"])
    return roles[:k]


def role_snapshot(role):
    """One-line description of a role, taken from its job description."""
    docs = retrieve(f"{role} summary", "job_description", role=role, k=1)
    if not docs:
        return role
    m = re.search(r"Summary:\s*(.+)", docs[0].page_content)
    return f"{role}: {m.group(1).strip() if m else docs[0].page_content[:200]}"


def make_parser(fields):
    """Build a StructuredOutputParser from (name, description) pairs."""
    return StructuredOutputParser.from_response_schemas(
        [ResponseSchema(name=n, description=d) for n, d in fields])


# Parser 1 - CV analysis (extends the lab's original schema)
profile_parser = make_parser([
    ("full_name", "The candidate's full name as a string, or \"Unknown\""),
    ("email", "The candidate's email address as a string, or \"Unknown\""),
    ("education", "List of objects, each with the keys degree, institution, year"),
    ("experience", "List of objects, each with the keys role, company, years. Include internships and training programs"),
    ("summary", "Two-sentence professional summary of the candidate, as a string"),
    ("experience_level", "Exactly one of: Student, Fresher, Junior, Mid-level, Senior"),
])

# Parser 2 - skill extraction
skills_parser = make_parser([
    ("technical_skills", "List of strings: programming languages and technical concepts (e.g. Python, SQL, machine learning, NLP)"),
    ("tools_and_technologies", "List of strings: frameworks, libraries, platforms and tools (e.g. PyTorch, Docker, LangChain)"),
    ("soft_skills", "List of strings: soft skills such as communication or teamwork"),
    ("domains", "List of strings: fields the candidate has worked or studied in"),
    ("notable_projects", "List of at most 5 short strings, one per project"),
])

# Parser 3 - CV vs. job requirements
gap_parser = make_parser([
    ("matched_skills", "List of strings: candidate skills that the target role requires"),
    ("missing_skills", "List of objects, each with the keys skill, priority (High, Medium or Low) and reason (one short sentence)"),
    ("match_score", "Integer from 0 to 100: how ready the candidate is for the target role today"),
    ("readiness_summary", "Two or three honest sentences explaining the score"),
])

# Parser 4 - roadmap, jobs and improvement plan
plan_parser = make_parser([
    ("recommended_roadmap", "List of 4 to 6 phases, each an object with the keys phase (short name), duration (e.g. '4 weeks'), topics (list of strings) and project (one practical project)"),
    ("suggested_jobs", "List of exactly 3 objects, each with the keys title, why_it_fits and next_step"),
    ("improvement_plan", "List of 3 objects, each with the keys timeframe (e.g. 'Next 30 days') and actions (list of 2 to 4 concrete strings)"),
])


_EMPTY = {"", "none", "n/a", "unknown", "not provided", "null"}


def _clean(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def _flatten(value):
    if value is None:
        return []
    if isinstance(value, dict):
        name = value.get("skill") or value.get("name") or value.get("title")
        return [name] if name else [x for v in value.values() for x in _flatten(v)]
    if isinstance(value, (list, tuple, set)):
        return [x for v in value for x in _flatten(v)]
    return [value]


def to_str_list(value):
    """Any LLM answer -> clean, de-duplicated list of strings."""
    if isinstance(value, str):
        value = re.split(r"[\n,;\u2022]+", value)
    out, seen = [], set()
    for item in _flatten(value):
        text = _clean(item).strip("-*\u2022 ")
        if text.lower() not in _EMPTY and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out


def to_dict_list(value, primary_key):
    """Any LLM answer -> list of dicts (plain strings become {primary_key: string})."""
    if value is None:
        return []
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, str):
        value = [v for v in re.split(r"\n+", value) if v.strip()]
    out = []
    for item in value if isinstance(value, (list, tuple)) else [value]:
        if isinstance(item, dict):
            out.append({str(k): v for k, v in item.items()})
        elif item is not None and _clean(item):
            out.append({primary_key: _clean(item)})
    return out


def _to_score(value):
    m = re.search(r"\d+(\.\d+)?", str(value))
    return max(0, min(100, int(float(m.group())))) if m else 0


def _text(value, default=""):
    text = _clean(value) if value is not None else ""
    return text if text.lower() not in _EMPTY else default


def normalize_profile(raw):
    return {
        "full_name": _text(raw.get("full_name"), "Unknown"),
        "email": _text(raw.get("email"), "Unknown"),
        "education": to_dict_list(raw.get("education"), "degree"),
        "experience": to_dict_list(raw.get("experience"), "role"),
        "summary": _text(raw.get("summary")),
        "experience_level": _text(raw.get("experience_level"), "Fresher"),
    }


def normalize_skills(raw):
    return {
        "technical_skills": to_str_list(raw.get("technical_skills")),
        "tools_and_technologies": to_str_list(raw.get("tools_and_technologies")),
        "soft_skills": to_str_list(raw.get("soft_skills")),
        "domains": to_str_list(raw.get("domains")),
        "notable_projects": to_str_list(raw.get("notable_projects"))[:5],
    }


def normalize_gap(raw):
    order = {"High": 0, "Medium": 1, "Low": 2}
    missing = []
    for item in to_dict_list(raw.get("missing_skills"), "skill"):
        skill = _text(item.get("skill") or item.get("name"))
        if not skill:
            continue
        priority = _text(item.get("priority"), "Medium").capitalize()
        missing.append({
            "skill": skill,
            "priority": priority if priority in order else "Medium",
            "reason": _text(item.get("reason") or item.get("why")),
        })
    missing.sort(key=lambda m: order[m["priority"]])
    return {
        "matched_skills": to_str_list(raw.get("matched_skills")),
        "missing_skills": missing,
        "match_score": _to_score(raw.get("match_score")),
        "readiness_summary": _text(raw.get("readiness_summary")),
    }


def normalize_plan(raw):
    roadmap = []
    for i, p in enumerate(to_dict_list(raw.get("recommended_roadmap"), "phase"), 1):
        roadmap.append({
            "phase": _text(p.get("phase") or p.get("name") or p.get("title"), f"Phase {i}"),
            "duration": _text(p.get("duration")),
            "topics": to_str_list(p.get("topics")),
            "project": _text(p.get("project")),
        })
    jobs = []
    for j in to_dict_list(raw.get("suggested_jobs"), "title"):
        title = _text(j.get("title") or j.get("job_title") or j.get("role"))
        if title:
            jobs.append({"title": title,
                         "why_it_fits": _text(j.get("why_it_fits") or j.get("why")),
                         "next_step": _text(j.get("next_step"))})
    plan = []
    for p in to_dict_list(raw.get("improvement_plan"), "timeframe"):
        plan.append({"timeframe": _text(p.get("timeframe") or p.get("period"), "Next steps"),
                     "actions": to_str_list(p.get("actions") or p.get("steps"))})
    return {"recommended_roadmap": roadmap, "suggested_jobs": jobs, "improvement_plan": plan}


def read_cv_pdf(path):
    """Extract plain text from a CV PDF."""
    pages = PyPDFLoader(str(path)).load()
    return "\n".join(p.page_content for p in pages).strip()


def extract_json_block(text):
    """Robust version of the lab helper: accepts a ```json fence, or falls back to the outermost {...}."""
    blocks = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if blocks:
        body = blocks[-1]
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in the model output")
        body = text[start:end + 1]
    return f"```json\n{body}\n```"

def parse_lenient(parser, text):
    fenced = extract_json_block(text)
    inner = re.search(r"```json\s*(.*?)\s*```", fenced, re.DOTALL).group(1)
    data = json.loads(inner, strict=False)          # turn the text into a dict
    names = [s.name for s in parser.response_schemas]
    if not any(n in data for n in names):           # still reject answers that are totally wrong
        raise ValueError("None of the expected keys were found")
    for n in names:
        data.setdefault(n, None)                    # a missing key becomes None instead of an error
    return data

def generate_and_parse(prompt_text, parser, max_new_tokens=700, retries=1):
    """LLM call + output parser. If the JSON is invalid, ask once more with a stricter instruction."""
    last_error, raw = None, ""
    for attempt in range(retries + 1):
        suffix = "" if attempt == 0 else (
            "\n\nIMPORTANT: your previous answer was not valid JSON. "
            "Reply with ONE valid JSON object inside a ```json code block and nothing else.")
        raw = llm_generate(prompt_text + suffix, max_new_tokens)
        try:
            return parse_lenient(parser, raw)
        except Exception as error:                       # noqa: BLE001
            last_error = error
    raise ValueError(f"Could not parse the model output: {last_error}\n--- RAW OUTPUT ---\n{raw}")


def make_chain(template, parser, max_new_tokens=700):
    """LCEL chain:  prompt template  ->  LLM + output parser  ->  Python dict."""
    prompt = PromptTemplate.from_template(template).partial(
        format_instructions=parser.get_format_instructions())
    return prompt | RunnableLambda(
        lambda prompt_value: generate_and_parse(prompt_value.to_string(), parser, max_new_tokens))


ANALYZE_CV_TEMPLATE = """You are an expert HR recruiter and CV analyst.
Analyse the CV below and extract the candidate profile.
Use ONLY information found in the CV or the extra notes. If something is missing write "Unknown" or an empty list.

CV:
{cv_text}

Extra notes from the candidate about their experience:
{extra_info}

{format_instructions}
Return only the JSON code block."""

EXTRACT_SKILLS_TEMPLATE = """You are a technical recruiter who extracts skills from CVs.
Read the CV, the skills the candidate listed themselves and their experience notes.
Extract a complete, de-duplicated skill inventory. Use short, standard skill names (for example "PyTorch", not "I used PyTorch to train models").
Do not invent skills that are not supported by the text.

CV:
{cv_text}

Skills listed by the candidate:
{stated_skills}

Experience notes:
{extra_info}

{format_instructions}
Return only the JSON code block."""

COMPARE_TEMPLATE = """You are a career coach comparing a candidate with a target job.

Target role: {target_role}

Candidate skills (JSON):
{candidate_skills}

Job requirements from the knowledge base:
{job_context}

Industry requirements from the knowledge base:
{industry_context}

Instructions:
- matched_skills: skills the candidate HAS that the role requires.
- missing_skills: skills the role requires that the candidate does NOT show. Use only skills mentioned in the requirements above, most important first. Priority High = required for the role, Medium = expected, Low = nice to have.
- match_score: honest readiness score from 0 to 100.
- Never claim the candidate has a skill that is not in their skill list.

{format_instructions}
Return only the JSON code block."""

PLAN_TEMPLATE = """You are an experienced career coach. Write a practical plan for the candidate.

Target role: {target_role}
Candidate level: {experience_level}
Candidate summary: {summary}
Current skills: {current_skills}
Missing skills (most important first): {missing_skills}
Readiness score: {match_score} / 100

Reference learning roadmap:
{roadmap_context}

Industry expectations:
{industry_context}

Other roles that match the candidate's skills (from the job knowledge base):
{related_roles}

Instructions:
- recommended_roadmap: ordered phases that close the missing skills. Adapt the reference roadmap and skip what the candidate already knows.
- suggested_jobs: exactly 3 realistic job titles. Include the target role (as a stretch goal if the score is low) and the best-matching alternatives.
- improvement_plan: three timeframes (30, 60 and 90 days) with concrete actions such as projects, portfolio, certificates and CV improvements.

{format_instructions}
Return only the JSON code block."""

analyze_cv_chain = make_chain(ANALYZE_CV_TEMPLATE, profile_parser, max_new_tokens=700)
extract_skills_chain = make_chain(EXTRACT_SKILLS_TEMPLATE, skills_parser, max_new_tokens=700)
compare_chain = make_chain(COMPARE_TEMPLATE, gap_parser, max_new_tokens=800)
plan_chain = make_chain(PLAN_TEMPLATE, plan_parser, max_new_tokens=1400)


def _json(data):
    return json.dumps(data, ensure_ascii=False, indent=1)


def run_career_coach(cv_text, skills_text="", experience_text="", target_role="", progress=None):
    """Full pipeline: CV + skills + experience + target role  ->  structured career report."""
    say = progress or print
    cv_text = (cv_text or "").strip()[:MAX_CV_CHARS]
    skills_text = (skills_text or "").strip()
    experience_text = (experience_text or "").strip()
    if not cv_text and not skills_text:
        raise ValueError("Please provide a CV or at least a list of skills.")

    # Chain 1 - analyse the CV
    say("Step 1/4 - Analysing the CV ...")
    profile = normalize_profile(analyze_cv_chain.invoke({
        "cv_text": cv_text or "Not provided",
        "extra_info": experience_text or "None"}))

    # Chain 2 - extract skills
    say("Step 2/4 - Extracting skills ...")
    skills = normalize_skills(extract_skills_chain.invoke({
        "cv_text": cv_text or "Not provided",
        "stated_skills": skills_text or "None",
        "extra_info": experience_text or "None"}))
    hard_skills = skills["technical_skills"] + skills["tools_and_technologies"]
    skills_query = ", ".join(hard_skills[:25]) or skills_text

    # Pick the knowledge-base role that matches the target (or the best fit if none was given)
    requested = (target_role or "").strip()
    kb_role = resolve_role(requested) if requested else \
        best_fit_roles(f"{skills_query}. {profile['summary']}", k=1)[0]

    # RAG (job descriptions + industry requirements) -> Chain 3 - compare with job requirements
    say(f"Step 3/4 - Retrieving requirements for '{kb_role}' and comparing ...")
    job_docs = retrieve(f"{kb_role} required skills", "job_description", role=kb_role, k=3)
    industry_docs = retrieve(f"{kb_role} industry requirements", "industry_requirement", role=kb_role, k=3)
    gap = normalize_gap(compare_chain.invoke({
        "target_role": kb_role,
        "candidate_skills": _json(skills),
        "job_context": format_docs(job_docs),
        "industry_context": format_docs(industry_docs)}))

    # RAG (learning roadmaps) -> Chain 4 - roadmap, suggested jobs, improvement plan
    say("Step 4/4 - Building the roadmap and improvement plan ...")
    roadmap_docs = retrieve(f"{kb_role} learning roadmap", "learning_roadmap", role=kb_role, k=3)
    related = best_fit_roles(skills_query or kb_role, k=4)
    plan = normalize_plan(plan_chain.invoke({
        "target_role": kb_role,
        "experience_level": profile["experience_level"],
        "summary": profile["summary"] or "Not available",
        "current_skills": ", ".join(hard_skills) or "None listed",
        "missing_skills": ", ".join(m["skill"] for m in gap["missing_skills"]) or "None identified",
        "match_score": gap["match_score"],
        "roadmap_context": format_docs(roadmap_docs),
        "industry_context": format_docs(industry_docs),
        "related_roles": "\n".join(role_snapshot(r) for r in related)}))

    report = {
        "target_role": requested or kb_role,
        "kb_role": kb_role,
        "candidate": {"name": profile["full_name"], "level": profile["experience_level"],
                      "summary": profile["summary"]},
        "match_score": gap["match_score"],
        "readiness_summary": gap["readiness_summary"],
        "current_skills": {"technical": skills["technical_skills"],
                           "tools": skills["tools_and_technologies"],
                           "soft": skills["soft_skills"]},
        "matched_skills": gap["matched_skills"],
        "missing_skills": gap["missing_skills"],
        "recommended_roadmap": plan["recommended_roadmap"],
        "suggested_jobs": plan["suggested_jobs"],
        "improvement_plan": plan["improvement_plan"],
    }
    sources = [{"role": d.metadata["role"], "type": d.metadata["doc_type"], "file": d.metadata["source"]}
               for d in job_docs + industry_docs + roadmap_docs]
    return {"report": report, "profile": profile, "skills": skills, "gap": gap, "sources": sources}
