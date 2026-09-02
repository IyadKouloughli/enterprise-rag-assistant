import os
import asyncio
from pathlib import Path
import gradio as gr
import spaces

from hybrid_search import load_index_and_metadata, hybrid_search
from generate_answer import rewrite_query, build_prompt, stream_llm

INDEX_DIR = Path("data/index")

@spaces.GPU
def run_hybrid_search_gpu(query, role, top_k, rerank):
    """Wrapper to run hybrid_search inside a ZeroGPU context."""
    return hybrid_search(
        index_dir=INDEX_DIR,
        query=query,
        role=role,
        top_k=top_k,
        rerank=rerank,
        return_audit=True,
    )

async def chat_wrapper(message, history, role, top_k, rerank):
    # Immediate feedback to the user so they know the server is processing
    yield "🔍 *Processing your query and searching the knowledge base...*"
    
    try:
        # Convert Gradio history (list of [user, bot]) to list of dicts for rewrite_query
        history_dicts = []
        for user_msg, bot_msg in history:
            history_dicts.append({"role": "user", "content": user_msg or ""})
            history_dicts.append({"role": "assistant", "content": bot_msg or ""})
            
        effective_query = await rewrite_query(message, history=history_dicts)
        
        # Run the GPU-decorated retrieval
        sources, audit_stats = run_hybrid_search_gpu(effective_query, role, top_k, rerank)
        
        if not sources:
            yield f"No relevant sources found in the knowledge base for role(s) [{role}]."
            return
            
        prompt = build_prompt(effective_query, sources)
        accumulated_answer = []
        
        # Stream from LLM
        async for token in stream_llm(
            system="You are an internal enterprise knowledge assistant. Answer using ONLY provided sources with [1], [2] citations.",
            prompt=prompt,
        ):
            accumulated_answer.append(token)
            yield "".join(accumulated_answer)
            
        # Append sources to the final answer
        final_answer = "".join(accumulated_answer)
        final_answer += "\n\n### Sources Used:\n"
        for i, s in enumerate(sources, start=1):
            final_answer += f"- **[{i}]** {s.get('title', 'Unknown')} ({s.get('source_path', 'Unknown')})\n"
        
        yield final_answer
        
    except Exception as e:
        import traceback
        yield f"⚠️ **System Error:** {str(e)}\n\n```python\n{traceback.format_exc()}\n```"

# Cleaned up custom CSS
custom_css = ""

# Build the Gradio UI
with gr.Blocks(title="NexusAI Copilot", theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🚀 NexusAI — Enterprise Knowledge Copilot")
    gr.Markdown("Welcome to the **Native Gradio Interface**. This interface securely connects you to internal enterprise knowledge with Hybrid Search and Neural Reranking.")
    
    with gr.Tab("Copilot Chat"):
        with gr.Row():
            with gr.Column(scale=1):
                role_input = gr.Textbox(label="Your Role (ACL)", value="engineer", info="e.g., hr, engineer, manager")
                top_k_input = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Top K Retrieval")
                rerank_input = gr.Checkbox(label="Enable Neural Reranking", value=True)
            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=chat_wrapper,
                    additional_inputs=[role_input, top_k_input, rerank_input]
                )
                
    with gr.Tab("Document Statistics"):
        def get_stats():
            try:
                index, meta = load_index_and_metadata(INDEX_DIR)
                dept_counts = {}
                for m in meta:
                    dept = m.get("department", "general")
                    dept_counts[dept] = dept_counts.get(dept, 0) + 1
                stats = f"**Total Indexed Documents:** {len(meta)}\n\n**By Department:**\n"
                for dept, count in dept_counts.items():
                    stats += f"- {dept.capitalize()}: {count}\n"
                return stats
            except Exception as e:
                return f"Error loading index: {e}"
                
        stats_btn = gr.Button("Refresh Statistics")
        stats_out = gr.Markdown()
        stats_btn.click(fn=get_stats, inputs=[], outputs=[stats_out])
        
if __name__ == "__main__":
    demo.queue().launch()
