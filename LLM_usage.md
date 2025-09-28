# 🤖 LLM Usage Report

This document transparently outlines where and how Large Language Models (LLMs) were used in this project.

---

## 🔹 Tools Used
- **ChatGPT (Free Tier, GPT-4o / GPT-5 depending on availability)**  
  Used mainly for **report writing assistance**, including:
  - Drafting and refining text sections (Q1–Q4 summaries, explanations, background).  
  - Rewording and polishing content to maintain clarity and readability.  
  - Integrating facts and outputs from experiments into coherent prose.  

- **Claude (Anthropic)**  
  Used specifically for **inference setup and experiment design** in the *paper section*, particularly:  
  - Helping plan the evaluation protocol for Q1–Q4.  
  - Structuring inference pipelines.  
  - Providing insights into experimental setup choices.

---

## 🔹 Auxiliary Script Generation
LLMs were also used to generate supporting code/scripts, which were later verified and adapted by hand:
- **Evaluation metric scripts** for Q5 (Multilingual & Code-Switch Stress Test) and Q6 (Robustness).  
- **Perturbed sentence generation scripts**:  
  - Automatically creating noisy/altered variants from normal sentences (typos, casing errors, spacing issues, etc.).  
  - Used for robustness testing.

---

## 🔹 Human Oversight
- All outputs from LLMs were **reviewed, validated, and edited** before inclusion.  
- No raw/unverified generations were used directly in the final submission.  
- LLMs served as **assistants** to accelerate workflow, not as final arbiters of correctness.

---

## 🔹 Ethical Note
This project followed responsible usage practices:  
- LLMs were used for support, not as ground-truth data generators.  
- Sensitive or potentially biased content was handled with care, especially in multilingual and cultural translation tasks.  
- The main experimental results are based on **actual runs/inference outputs**, not hallucinated model claims.

