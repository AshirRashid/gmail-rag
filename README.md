# Email Event Finder

Local semantic search over your Gmail inbox. It surfaces emails that contain events, meetings, appointments, deadlines, and other scheduling information, using a locally-run embedding model and ChromaDB. No email content leaves the machine, and there is no external API call anywhere in the pipeline.

## Try it in one command

No Gmail account, no setup:

```bash
./demo.sh
```

This bootstraps a virtual environment, installs dependencies, starts ChromaDB on localhost, ingests a bundled synthetic inbox, and launches the query UI at http://127.0.0.1:7860. Use it to see how the search works before pointing it at your own mail.

## How it works

1. Fetches emails from Gmail via the API (read-only OAuth scope)
2. Skips replies, so only the original (first) email in each thread is indexed
3. Cleans each email body (strips quoted replies and footers)
4. Embeds each email as a single whole-email document with `BAAI/bge-base-en-v1.5` (runs locally)
5. Stores the embeddings in ChromaDB, keyed by message ID
6. Queries the collection semantically to find the emails you describe

## Evidence

The `eval/` directory holds a benchmark against a BM25 baseline, latency and cost measurements, a failure analysis, and a security review. See [`eval/README.md`](eval/README.md). Headline: semantic recall@5 of 0.849 (0.688 for BM25), query p50 of 60.7ms, and $0 marginal cost.

---

## Run it on your own inbox

### Prerequisites

- Python 3.11+
- A Google Cloud project with the Gmail API enabled

### 1. Setup

```bash
./setup.sh
```

This creates a `.venv`, installs dependencies, starts ChromaDB in the background (logs go to `chroma.log`), and tells you what to do if Gmail credentials are missing.

### 2. Gmail credentials

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project and enable the **Gmail API**
3. Create an **OAuth 2.0 Client ID** (Desktop app)
4. Download the JSON file and place it in the repo root
5. If the filename differs from the default, set it in `config.py`:

```python
# config.py
CREDENTIALS_FILE = "your-credentials-file.json"
```

### 3. Ingest and search

```bash
./run.sh
```

`run.sh` starts ChromaDB if it is not already running, ingests your inbox if the collection is empty, and launches the UI. On first run a browser window opens for Gmail OAuth; a `token.json` is saved so later runs skip it.

To fetch more emails on a manual ingest:

```bash
N_EMAILS=200 python pipeline.py
```

### Command-line query (without the UI)

```bash
python query.py
python query.py -q "an email inviting me to a team lunch or dinner with a date and time"
N_RESULTS=10 python query.py -q "an email about an upcoming deadline or submission date"
```

### Writing good queries

The model matches meaning, not keywords. Queries work best phrased as a description of the email you are looking for, in natural language.

| Goal | Good query |
|---|---|
| Find event invites | `an email inviting me to an event, party, or gathering with a specific date` |
| Find meetings | `an email asking me to join a meeting or call at a scheduled time` |
| Find appointments | `an email confirming a doctor, dentist, or personal appointment` |
| Find deadlines | `an email mentioning an upcoming deadline, due date, or submission cutoff` |
| Find travel | `an email with flight, hotel, or travel booking confirmation and dates` |

**Tips:**
- Describe the email, not what you want to do. *"an email confirming..."* works better than *"find confirmations..."*
- Include the type of information you expect. Dates, times, and locations make queries more precise.
- Avoid single keywords. *"an email about a job interview with a scheduled time"* outperforms *"interview"*.

---

## Configuration

All settings can be overridden with environment variables:

| Variable | Default | Description |
|---|---|---|
| `CREDENTIALS_FILE` | `gmail_main_new_secret.json` | OAuth credentials filename |
| `CHROMA_HOST` | `localhost` | ChromaDB host |
| `CHROMA_PORT` | `8000` | ChromaDB port |
| `CHROMA_COLLECTION` | `emails` | Collection name |
| `EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding model |
| `N_EMAILS` | `500` | Emails to fetch per pipeline run (replies are filtered out automatically) |
| `N_RESULTS` | `5` | Results returned per query |
