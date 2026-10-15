# AI Market Research & Strategy Engine

An AI-powered market research application that turns a business question into a structured strategy brief. Built with **Python, Streamlit, Gemini, web search, and a multi-agent research workflow**, it collects source material, extracts claims, validates evidence, identifies themes, and generates recommendations with traceable URLs.

> **Important:** This project is a research assistant, not a substitute for expert judgment. Verify important claims and source links before using findings for business decisions.

## Highlights

- **Research planning:** Breaks a broad business question into focused research tasks and search queries.
- **Web discovery:** Searches the web through configurable providers and collects relevant pages or search snippets.
- **Evidence extraction:** Converts source content into structured claims, excerpts, entities, and topics.
- **Evidence validation:** Flags weak or unsupported evidence and records validation notes and confidence scores.
- **Insight aggregation:** Groups accepted evidence into themes and market signals.
- **Strategy brief generation:** Produces an executive summary, key findings, competitor observations, implications, and recommendations.
- **Evidence review UI:** Inspect accepted and rejected evidence, source URLs, fetch status, and workflow logs.
- **Research memory / retrieval:** Retrieves related prior research where available.
- **Streamlit interface:** Runs as a single Python application without a separate JavaScript frontend.

## Workflow

```text
Business Question
       |
       v
Research Planner
       |
       v
Web Search & Page Fetching
       |
       v
Evidence Extraction
       |
       v
Evidence Validation
       |
       v
Theme & Insight Aggregation
       |
       v
Strategy Brief + Evidence Appendix
```

## Tech Stack

- **Language:** Python
- **UI:** Streamlit
- **LLM:** Google Gemini (`google-genai`)
- **Web search:** Gemini Grounding with Google Search, `googlesearch-python`, or DuckDuckGo fallback
- **Page parsing:** Requests, Beautiful Soup
- **Research retrieval/memory:** Embedding-based retrieval components
- **Configuration:** `python-dotenv`

## Repository Structure

```text
McKinsey-AI-Market-Research-Strategy-Engine/
└── app/
    ├── app.py                    # Streamlit interface and pipeline orchestration
    ├── requirements.txt          # Python dependencies
    ├── .env.example              # Example environment configuration
    ├── check_search.py           # Search-provider diagnostic
    ├── agents/
    │   ├── planner.py            # Research planning
    │   ├── browser.py            # Search and source collection
    │   ├── extractor.py          # Structured evidence extraction
    │   ├── validator.py          # Evidence checks
    │   ├── aggregator.py         # Theme and insight aggregation
    │   └── report_writer.py      # Strategy brief generation
    ├── utils/
    │   ├── llm_client.py         # Gemini client and model fallback handling
    │   ├── search.py             # Search-provider integrations
    │   ├── rag_store.py          # Embedding/retrieval utilities
    │   ├── memory.py             # Research memory
    │   └── models.py             # Shared data models
    ├── data/
    │   └── sample_research_requests.json
    ├── Evidence SS.png
    └── REPORT SS.png
```

## Getting Started

### 1. Prerequisites

- Python **3.11 or 3.12** recommended for broad package compatibility
- A Google Gemini API key: [Get an API key](https://aistudio.google.com/apikey)
- Git (optional, for cloning)

### 2. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/McKinsey-AI-Market-Research-Strategy-Engine.git
cd McKinsey-AI-Market-Research-Strategy-Engine/app
```

Replace `YOUR-USERNAME` with your GitHub username. If you downloaded the project as a ZIP, open a terminal in the `app` directory instead.

### 3. Create and activate a virtual environment

**Windows PowerShell**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for this terminal, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure environment variables

Copy `.env.example` to `.env`.

**Windows PowerShell**
```powershell
Copy-Item .env.example .env
```

**macOS / Linux**
```bash
cp .env.example .env
```

Open `.env` and set your API key:

```env
GEMINI_API_KEY=your_actual_api_key
```

Optional settings are available in `.env.example`, including the primary Gemini model, fallback models, embedding model, and search provider. Use model IDs currently supported by your Google AI account and SDK. Model availability and quotas can change; if a model returns a 404 or 429 error, check the current Gemini model list and your API quota.

**Never commit `.env` or publish your API key.** Keep secrets in local environment variables or your deployment platform's secret manager.

### 6. Run the application

From the `app` directory, with your virtual environment active:

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## Using the App

1. Open the research intake section.
2. Enter a focused business question, such as a market-entry, competitor, or trend-analysis question.
3. Start the research workflow.
4. Review the **Workflow Monitor** for research tasks, source counts, and execution logs.
5. Use the **Evidence Review** section to inspect claims, excerpts, confidence, validation notes, and source links.
6. Read the generated **Strategy Brief** and evidence appendix.
7. Independently verify critical facts before relying on the recommendations.

### Example research questions

- Should we enter the fintech-for-freelancers market in Southeast Asia?
- How are Stripe, Adyen, and Checkout.com positioning themselves in embedded finance?
- What are the emerging signals in enterprise AI agent adoption?
- Which competitors, customer needs, and regulatory factors should be considered before entering a new market?

## Search Configuration & Troubleshooting

The application supports configurable search providers. Review the comments in `.env.example` and `utils/search.py` for the supported values and provider order.

### No sources found
- Run the diagnostic from the `app` directory:
  ```bash
  python check_search.py
  ```
- Confirm that `.env` is in the same `app` directory from which you run Streamlit.
- Check internet access and whether your selected search provider is rate-limiting requests.
- Try the configured fallback provider and simplify the research query.

### Gemini 404 / model not found
- Confirm that the model ID is currently available to your API key.
- Update `GEMINI_MODEL` and `GEMINI_MODEL_FALLBACK` in `.env` to supported model IDs.
- Check that the installed `google-genai` SDK is up to date.

### Gemini 429 / quota exceeded
- Check Google AI Studio quota and billing/rate limits for your account.
- Reduce repeated or concurrent requests and retry after the quota window resets.
- A fallback model does not guarantee that quota limits will be avoided.

### Dependency or Python errors
- Use a fresh virtual environment with Python 3.11 or 3.12.
- Install packages from `app/requirements.txt` while your terminal is in the `app` directory.
- If installation fails, review the first package-specific error in the terminal output.

## Screenshots

The repository includes these screenshots:

- `app/Evidence SS.png` — evidence review interface
- `app/REPORT SS.png` — generated strategy report

If you move the screenshots into a `screenshots/` directory, update the image paths below accordingly.

### Evidence Review
![Evidence Review](app/Evidence%20SS.png)

### Strategy Report
![Strategy Report](app/REPORT%20SS.png)

## Deployment

The app can be deployed to a Streamlit-compatible hosting service or another Python hosting platform.

At minimum, configure the deployment with:
- The app entry point: `app.py`
- The dependencies from `requirements.txt`
- `GEMINI_API_KEY` and any optional configuration as platform secrets/environment variables

Do not upload `.env`, virtual environments, API keys, or generated private research data to a public repository. Confirm that the selected host's Python version and outbound network access are compatible with the app.

## Limitations

- Search results and webpage availability depend on third-party services, network access, and rate limits.
- Some sites block automated fetching or render content with JavaScript; the app may fall back to search snippets.
- Evidence validation reduces risk but cannot guarantee that every claim is correct or unbiased.
- LLM-generated summaries and recommendations can contain errors. Check original sources.
- API model availability, quotas, and pricing may change.

## Future Improvements

- Add domain allowlists and source-quality scoring.
- Improve source deduplication and recency checks.
- Add stronger cross-source claim verification.
- Add export options for Markdown/PDF and downloadable evidence tables.
- Add automated tests, structured logs, and configurable research-depth limits.
- Add caching and clearer handling of API quota failures.

## License

No license file was included in the supplied project archive. Before publishing, choose a license (for example, MIT) and add a `LICENSE` file if you want others to reuse the code. Without a license, standard copyright restrictions apply.

## Disclaimer

This project is an independent portfolio project and is not affiliated with, endorsed by, or sponsored by McKinsey & Company. The name is used descriptively to indicate a consulting-style market research workflow.
