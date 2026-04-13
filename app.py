import gradio as gr
from query import DEFAULT_QUERY, query


def search(q: str) -> str:
    results = query(q)
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"**[{i}] {r['subject']}**  \n"
            f"{r['sender']} · {r['date']} · score {r['score']}  \n"
            f"{r['snippet']}..."
        )
    return "\n\n---\n\n".join(lines)


gr.Interface(
    fn=search,
    inputs=gr.Textbox(label="Query", placeholder=DEFAULT_QUERY, lines=2),
    outputs=gr.Markdown(label="Results"),
    title="Email Event Finder",
    allow_flagging="never",
).launch()
