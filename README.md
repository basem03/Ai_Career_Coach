# AI Career Coach

An AI career coach that helps users choose a suitable career path. It reads a CV, compares the candidate with real job requirements, and returns a personalised skill-gap analysis, learning roadmap, job suggestions and improvement plan.

Built with **RAG + LangChain chains + output parsers**, **Mistral-7B-Instruct**, **FAISS** and **Streamlit**.

## What it does

**Input**
- CV (PDF or pasted text)
- Skills
- Experience
- Target role (or "not sure, suggest one")

**Output**
- Current skills
- Missing skills (with priority)
- Recommended roadmap
- Suggested jobs
- Improvement plan (30 / 60 / 90 days)
- Estimated readiness score (0-100)

## How it works

```
 CV + skills + experience + target role
              |
   Chain 1: Analyse CV ............ candidate profile
   Chain 2: Extract skills ........ skill inventory
              |
   RAG  <-  job descriptions + industry requirements
   Chain 3: Compare with job requirements ... skill gap
              |
   RAG  <-  learning roadmaps
   Chain 4: Roadmap + jobs + improvement plan
              |
        Final report  ->  Streamlit UI
```

| Part | Implementation |
|---|---|
| LLM | `mistralai/Mistral-7B-Instruct-v0.2` (Hugging Face Transformers) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS, with `role` and `doc_type` metadata filters |
| Chains | LangChain LCEL: `prompt \| LLM + parser` |
| Output parsers | `StructuredOutputParser`, one per chain, with type clean-up |
| UI | Streamlit |

## Project files

| File | Purpose |
|---|---|
| `lab-4-career-coach.ipynb` | Kaggle notebook: builds and tests everything step by step |
| `career_coach.py` | All logic as an importable module (written by the notebook) |
| `app.py` | Streamlit interface |
| `requirements.txt` | Python dependencies |
| `knowledge_base/` | RAG documents (created by the notebook) |

## Run on Kaggle

1. Import `ai-career-coach.ipynb` into a new Kaggle notebook.
2. Settings: **GPU** (T4 x2 or P100) and **Internet on**.
3. Add your CV dataset as an input, so `pdf_path` in the first cells is valid.
4. Create a free [ngrok](https://ngrok.com) account and add your token as a Kaggle secret named `NGROK_TOKEN`.
5. Run the cells **top to bottom** (do not use "Run All"):
   - Part 1: load Mistral and read the CV.
   - Part 2: build the knowledge base, parsers and chains, then test with `run_career_coach(...)`.
   - Part 3: write the app files, free the GPU, and launch Streamlit. It prints a public URL.
6. Open the URL. The first load takes a few minutes while the model loads.

## Run locally

Put `career_coach.py`, `app.py`, `requirements.txt` and the `knowledge_base/` folder in one directory:

```bash
pip install -r requirements.txt
streamlit run app.py
```

A GPU is recommended. Without one, use a smaller model (answers will be weaker):

```bash
export COACH_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export COACH_4BIT=0
streamlit run app.py
```

### Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `COACH_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | Hugging Face model to use |
| `COACH_4BIT` | `1` | `1` = load in 4-bit to save GPU memory, `0` = full precision |
| `COACH_BASE_DIR` | `/kaggle/working` on Kaggle, otherwise `.` | Where `knowledge_base/` and the FAISS index live |

## Customise the knowledge base

The starter set covers 7 roles: Data Analyst, Data Scientist, Machine Learning Engineer, AI / LLM Engineer, Backend Developer, Frontend Developer and DevOps Engineer.

To add roles or real job postings, put `.md`, `.txt` or `.pdf` files in:

```
knowledge_base/
  job_descriptions/
  learning_roadmaps/
  industry_requirements/
```

Start each file with a line such as `Role: Cloud Engineer`, then rebuild the index:

```python
get_vectorstore(rebuild=True)
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Expected key ... to be present` | The model left out a JSON field. Make sure `career_coach.py` uses `parse_lenient`, then **restart** Streamlit. |
| App still shows an old error | Streamlit keeps the old module in memory. Stop the server and start it again. |
| GPU out of memory | Free the notebook's model before launching the app and keep `COACH_4BIT=1`. |
| `NGROK_TOKEN` not found | Check the secret name, that it is attached to the notebook, and that Internet is on. |
| "No text could be read from this PDF" | The PDF is a scan. Paste the CV text instead. |
| "Could not parse the model output" | Run the analysis again and check the raw output shown in the error. |

## Limitations

- One report makes 4 LLM calls, so it takes a few minutes on a T4.
- The readiness score is an LLM estimate, not a measured value.
- Results depend on the knowledge base. If a target role is not in it, the closest known role is used.
- The starter knowledge base is general guidance. Replace or extend it with current job postings for your market.
- This is a guidance tool, not a hiring decision. Treat its output as a starting point.
