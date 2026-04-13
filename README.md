# Email Event Finder (v2)

Scans your Gmail inbox and surfaces emails that contain events, meetings, or scheduling information — using a local embedding model and ChromaDB.

## How it works

1. Fetches emails from Gmail via the API
2. Skips replies — only the original (first) email in each thread is indexed
3. Cleans each email body (strips quoted replies, footers)
4. Embeds each email as a single document with `BAAI/bge-base-en-v1.5` (runs locally)
5. Stores embeddings in ChromaDB
6. Queries the DB semantically to find event-related emails

---

## Prerequisites

- Python 3.11+
- Google Cloud project with the Gmail API enabled

---

## Setup

### Automated (recommended)

```bash
cd v2
bash setup.sh
```

This will:
- Create a `.venv` virtual environment and install all dependencies
- Start ChromaDB in the background (logs → `chroma.log`)
- Tell you what to do if Gmail credentials are missing

### Manual

#### 1. Create a virtual environment and install dependencies

```bash
cd v2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 2. Start ChromaDB

```bash
chroma run --path ./chroma-data --host localhost --port 8000
```

Leave this running in a separate terminal.

#### 3. Get Gmail credentials

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project → enable the **Gmail API**
3. Create an **OAuth 2.0 Client ID** (Desktop app)
4. Download the JSON file and place it in the `v2/` directory
5. Set the filename in `config.py` if it differs from the default:

```python
# config.py
CREDENTIALS_FILE = "your-credentials-file.json"
```

---

## Usage

### Ingest emails

```bash
cd v2
python pipeline.py
```

On first run, a browser window will open for Gmail OAuth. A `token.json` file is saved so subsequent runs skip this step.

To fetch more emails:

```bash
N_EMAILS=200 python pipeline.py
```

### Query for events

```bash
python query.py
```

With a custom query:

```bash
python query.py -q "an email inviting me to a team lunch or dinner with a date and time"
```

To return more results:

```bash
N_RESULTS=10 python query.py -q "an email about an upcoming deadline or submission date"
```

### Writing good queries

The model matches meaning, not keywords. Queries work best when phrased as a description of the email you're looking for, written in natural language.

| Goal | Good query |
|---|---|
| Find event invites | `an email inviting me to an event, party, or gathering with a specific date` |
| Find meetings | `an email asking me to join a meeting or call at a scheduled time` |
| Find appointments | `an email confirming a doctor, dentist, or personal appointment` |
| Find deadlines | `an email mentioning an upcoming deadline, due date, or submission cutoff` |
| Find travel | `an email with flight, hotel, or travel booking confirmation and dates` |

**Tips:**
- Describe the email, not what you want to do — *"an email confirming..."* not *"find confirmations..."*
- Include the type of information you expect — dates, times, locations make queries more precise
- Avoid single keywords — *"an email about a job interview with a scheduled time"* outperforms *"interview"*

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
| `N_EMAILS` | `50` | Emails to fetch per pipeline run (replies are filtered out automatically) |
| `N_RESULTS` | `5` | Results returned per query |
