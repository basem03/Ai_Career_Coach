"""Simple Streamlit UI for the AI Career Coach.   Run:  streamlit run app.py"""
import json
import os
import tempfile

import pandas as pd
import streamlit as st

import career_coach as cc

st.set_page_config(page_title="AI Career Coach", page_icon="🧭", layout="wide")

NOT_SURE = "Not sure - suggest the best fit for me"
OTHER = "Other (type it below)"


@st.cache_resource(show_spinner="Loading the language model and knowledge base (first run takes a few minutes)...")
def init_backend():
    """Load the LLM and the FAISS index once per server process."""
    cc.load_llm(use_4bit=os.environ.get("COACH_4BIT", "1") == "1")
    cc.get_vectorstore()
    return cc.list_roles()


def render_report(result):
    r = result["report"]

    st.subheader(f"Target role: {r['target_role']}")
    if r["target_role"].lower() != r["kb_role"].lower():
        st.caption(f"Closest role in the knowledge base: **{r['kb_role']}**")
    left, right = st.columns([1, 3])
    left.metric("Estimated readiness", f"{r['match_score']} / 100")
    right.progress(r["match_score"] / 100)
    right.write(r["readiness_summary"])
    st.caption("The score is an LLM estimate based on retrieved job requirements, not an exact measurement.")

    tabs = st.tabs(["✅ Current skills", "⚠️ Missing skills", "🗺️ Roadmap",
                    "💼 Suggested jobs", "📈 Improvement plan", "🔍 Under the hood"])

    with tabs[0]:
        for label, key in [("Technical", "technical"), ("Tools & technologies", "tools"), ("Soft skills", "soft")]:
            st.markdown(f"**{label}**")
            items = r["current_skills"][key]
            st.write(" · ".join(f"`{s}`" for s in items) if items else "_None found_")
        if r["matched_skills"]:
            st.markdown("**Already relevant for this role**")
            st.write(" · ".join(f"`{s}`" for s in r["matched_skills"]))

    with tabs[1]:
        if r["missing_skills"]:
            df = pd.DataFrame(r["missing_skills"]).rename(
                columns={"skill": "Skill", "priority": "Priority", "reason": "Why it matters"})
            st.dataframe(df, hide_index=True)
        else:
            st.success("No major gaps were found for this role.")

    with tabs[2]:
        for i, p in enumerate(r["recommended_roadmap"], 1):
            with st.expander(f"{i}. {p['phase']}  ·  {p['duration']}", expanded=(i == 1)):
                for t in p["topics"]:
                    st.markdown(f"- {t}")
                if p["project"]:
                    st.markdown(f"**Project:** {p['project']}")

    with tabs[3]:
        for j in r["suggested_jobs"]:
            st.markdown(f"#### {j['title']}")
            st.write(j["why_it_fits"])
            if j["next_step"]:
                st.markdown(f"**Next step:** {j['next_step']}")
            st.divider()

    with tabs[4]:
        for step in r["improvement_plan"]:
            st.markdown(f"#### {step['timeframe']}")
            for a in step["actions"]:
                st.markdown(f"- {a}")

    with tabs[5]:
        st.caption("Intermediate results of each chain, and the knowledge-base documents that were retrieved.")
        with st.expander("Chain 1 - CV analysis"):
            st.json(result["profile"])
        with st.expander("Chain 2 - Extracted skills"):
            st.json(result["skills"])
        with st.expander("Chain 3 - Comparison with job requirements"):
            st.json(result["gap"])
        with st.expander("RAG sources used"):
            st.dataframe(pd.DataFrame(result["sources"]), hide_index=True)
        st.download_button("Download full report (JSON)",
                           json.dumps(result["report"], indent=2, ensure_ascii=False),
                           file_name="career_report.json", mime="application/json")


# ----------------------------------------------------------------------------- UI
st.title("🧭 AI Career Coach")
st.write("Upload your CV, add your skills and target role, and get a personalised gap analysis, "
         "learning roadmap, suggested jobs and improvement plan.")

try:
    roles = init_backend()
except Exception as error:  # noqa: BLE001
    st.error(f"Could not start the backend: {error}")
    st.stop()

with st.form("career_form"):
    col1, col2 = st.columns(2)
    with col1:
        cv_file = st.file_uploader("CV (PDF)", type=["pdf"])
        cv_pasted = st.text_area("...or paste your CV text", height=150)
        skills_text = st.text_area("Your skills (comma separated)", height=90,
                                   placeholder="Python, SQL, PyTorch, LangChain ...")
    with col2:
        experience_text = st.text_area("Experience (internships, projects, years of work)", height=150)
        choice = st.selectbox("Target role", [NOT_SURE] + roles + [OTHER])
        custom_role = st.text_input("Custom target role", disabled=(choice != OTHER))
    submitted = st.form_submit_button("Analyse my career path", type="primary")

if submitted:
    cv_text = cv_pasted
    if cv_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(cv_file.getvalue())
        try:
            cv_text = cc.read_cv_pdf(tmp.name)
        finally:
            os.unlink(tmp.name)
        if not cv_text:
            st.warning("No text could be read from this PDF (is it a scan?). Paste the text instead.")

    target = "" if choice == NOT_SURE else (custom_role if choice == OTHER else choice)

    try:
        with st.status("Working on your career report...", expanded=True) as status:
            st.session_state["result"] = cc.run_career_coach(
                cv_text, skills_text, experience_text, target, progress=st.write)
            status.update(label="Done!", state="complete", expanded=False)
    except Exception as error:  # noqa: BLE001
        st.session_state.pop("result", None)
        st.error(str(error))

if "result" in st.session_state:
    render_report(st.session_state["result"])
