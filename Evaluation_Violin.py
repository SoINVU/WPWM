import os
import argparse
import logging
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import faiss
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class RAGComparisonExperiment:
    def __init__(self, config):
        self.config = config
        self._setup_logging()
        self.system_prompt = "You are a professional equipment troubleshooting assistant. Based on the provided references (if any) and your expertise, analyze the fault below and provide precise, brief troubleshooting suggestions. /no think"

        self.logger.info("Starting experiment system initialization...")
        self._init_models()
        self._load_vector_db()
        self.logger.info("System initialization complete!")

    def _setup_logging(self):
        log_dir = self.config.get('output_dir', './experiment_results')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                            handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
        self.logger = logging.getLogger("RAG_Experiment")

    def _init_models(self):
        self.embed_tokenizer = AutoTokenizer.from_pretrained(self.config['embed_path'], trust_remote_code=True)
        self.embed_model = AutoModel.from_pretrained(self.config['embed_path'], trust_remote_code=True,
                                                     torch_dtype=torch.float16, device_map="auto").eval()
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.config['llm_path'], trust_remote_code=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config['llm_path'], trust_remote_code=True,
                                                              torch_dtype=torch.float16, device_map="auto",
                                                              low_cpu_mem_usage=True).eval()
        self.is_qwen_model = "Qwen" in self.config['llm_path']

    def _load_vector_db(self):
        self.faiss_index = faiss.read_index(f"{self.config['index_path']}.faiss")
        with open(f"{self.config['index_path']}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunks, self.metadata = data['chunks'], data['metadata']

    def get_embedding(self, text: str) -> np.ndarray:
        if not text: return np.zeros(self.embed_model.config.hidden_size)
        inputs = self.embed_tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(
            self.embed_model.device)
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
            mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            embeddings = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy().flatten()

    def retrieve_context(self, query: str) -> str:
        q_emb = self.get_embedding(query)
        q_emb_search = (q_emb / np.linalg.norm(q_emb)).reshape(1, -1).astype('float32')
        search_k = min(self.config['top_k'] * 2, len(self.chunks))
        distances, indices = self.faiss_index.search(q_emb_search, search_k)

        documents, metadata_list = [], []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and float(distances[0][i]) >= self.config['similarity_threshold']:
                documents.append(self.chunks[idx])
                metadata_list.append(self.metadata[idx])
                if len(documents) >= self.config['use_top_k']: break

        if len(documents) < 2:
            query_words = query.lower().split()
            for chunk, meta in zip(self.chunks, self.metadata):
                if any(len(w) > 2 and w in chunk.lower() for w in query_words):
                    documents.append(chunk)
                    metadata_list.append(meta)
                    if len(documents) >= self.config['use_top_k']: break

        if not documents: return ""

        return "".join([f"Reference: {m.get('REPAIR_SUGGESTION', d[:200])}\n" for d, m in
                        zip(documents[:self.config['use_top_k']], metadata_list[:self.config['use_top_k']])])

    def generate_llm_response(self, query: str, context: str = None) -> str:
        prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n" + (
            f"Reference:\n{context}\n" if context else "") + f"Query:{query}<|im_end|>\n<|im_start|>assistant\n" if self.is_qwen_model else f"{self.system_prompt}\n\n" + (
            f"Reference:\n{context}\n" if context else "") + f"Query:{query}\nAnswer:"
        inputs = self.llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(
            self.llm_model.device)
        with torch.no_grad():
            outputs = self.llm_model.generate(**inputs, max_new_tokens=150, temperature=0.1, do_sample=False,
                                              pad_token_id=self.llm_tokenizer.eos_token_id)
        resp = self.llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return re.sub(r'<think>.*?</think>\n*', '', resp, flags=re.DOTALL).strip()

    def run_experiments(self):
        try:
            df = pd.read_csv(self.config['csv_path'], encoding='utf-8')
        except:
            df = pd.read_csv(self.config['csv_path'], encoding='gbk')

        grouped = df.dropna(subset=['FAULT', 'REPAIR_SUGGESTION']).groupby('FAULT')[
            'REPAIR_SUGGESTION'].agg(lambda x: '. '.join([str(i) for i in x])).reset_index()
        if self.config.get('max_queries'): grouped = grouped.head(self.config['max_queries'])

        queries, truths = grouped['FAULT'].tolist(), grouped['REPAIR_SUGGESTION'].tolist()
        self.logger.info(f"Starting experiment. Total {len(queries)} queries. Recording query-level distribution characteristics.")

        truth_embs = [self.get_embedding(str(t)) for t in tqdm(truths, desc="Truth Embeddings")]
        rag_scores, non_rag_scores = [], []

        # List to save detailed results
        detailed_results = []

        for q_idx, query in enumerate(tqdm(queries, desc="Evaluating Queries")):
            truth_emb = truth_embs[q_idx]
            truth_text = truths[q_idx]

            # 1. Non-RAG Pipeline
            resp_non_rag = self.generate_llm_response(query, context=None)
            emb_non_rag = self.get_embedding(resp_non_rag)
            score_non_rag = float(np.dot(emb_non_rag, truth_emb)) if not np.all(emb_non_rag == 0) else 0.0
            non_rag_scores.append(score_non_rag)

            # 2. RAG Pipeline
            retrieved_ctx = self.retrieve_context(query)
            resp_rag = self.generate_llm_response(query, context=retrieved_ctx)
            emb_rag = self.get_embedding(resp_rag)
            score_rag = float(np.dot(emb_rag, truth_emb)) if not np.all(emb_rag == 0) else 0.0
            rag_scores.append(score_rag)

            # 3. Record details for each data point
            detailed_results.append({
                "Query_ID": q_idx + 1,
                "Query_Text": query,
                "Ground_Truth": truth_text,
                "Non_RAG_Response": resp_non_rag,
                "Non_RAG_Similarity": round(score_non_rag, 4),
                "RAG_Retrieved_Context": retrieved_ctx.strip() if retrieved_ctx else "No relevant content retrieved",
                "RAG_Response": resp_rag,
                "RAG_Similarity": round(score_rag, 4),
                "Similarity_Improvement": round(score_rag - score_non_rag, 4)  # RAG improvement margin
            })

        # Save detailed results to a CSV file
        results_df = pd.DataFrame(detailed_results)
        csv_filename = f'Detailed_Results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        csv_filepath = os.path.join(self.config.get('output_dir', '.'), csv_filename)
        # Using utf-8-sig encoding to prevent garbled text when opened in Excel
        results_df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
        self.logger.info(f"Detailed experiment records (including responses and similarities) saved to: {csv_filepath}")

        # Plot charts
        self.plot_academic_violin(rag_scores, non_rag_scores, len(queries))

    def plot_academic_violin(self, rag_scores, non_rag_scores, n_queries):
        """Generate academic violin plot + internal boxplot"""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        data = [non_rag_scores, rag_scores]

        # 1. Draw violin plot (showing kernel density distribution)
        parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False, widths=0.7)
        colors = ['#801F3F', '#544795']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.6)

        # 2. Draw slim boxplot inside violin plot (showing quartiles)
        bp = ax.boxplot(data, patch_artist=True, widths=0.15,
                        boxprops=dict(facecolor='white', color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5),
                        medianprops=dict(color='black', linewidth=2.5), zorder=3)

        # 3. Annotate mean and statistical info
        text_bbox = dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA', edgecolor='gray', alpha=0.8)
        ax.text(1, np.max(non_rag_scores) + 0.05,
                f"Mean: {np.mean(non_rag_scores):.3f}\nSD: {np.std(non_rag_scores):.3f}", ha='center', fontsize=11,
                bbox=text_bbox)
        ax.text(2, np.max(rag_scores) + 0.05, f"Mean: {np.mean(rag_scores):.3f}\nSD: {np.std(rag_scores):.3f}",
                ha='center', fontsize=11, bbox=text_bbox)

        # 4. Chart aesthetics
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Non-RAG System', 'RAG System'], fontsize=13, fontweight='bold')
        ax.set_ylabel('Cosine Similarity', fontsize=14, fontweight='bold')
        ax.set_title(f'Performance Distribution: RAG vs. Non-RAG\n(n={n_queries} Unique Queries)', fontsize=15,
                     fontweight='bold', pad=20)

        ax.set_ylim(max(0.0, min(min(non_rag_scores), min(rag_scores)) - 0.1), 1.1)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        plot_path = os.path.join(self.config.get('output_dir', '.'),
                                 f'ViolinPlot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Academic violin plot successfully saved: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--llm_path", type=str, default="/mnt/workspace/models/Qwen/Qwen3-8B")
    parser.add_argument("--embed_path", type=str, default="/mnt/workspace/models/Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--index_path", type=str, default="/mnt/workspace/data/faiss_index")
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./experiment_results")
    args = parser.parse_args()

    config = {
        "csv_path": args.csv,
        "llm_path": args.llm_path,
        "embed_path": args.embed_path,
        "index_path": args.index_path,
        "max_queries": args.max_queries,
        "output_dir": args.output_dir,
        "top_k": 5,
        "use_top_k": 3,
        "similarity_threshold": 0.5
    }

    RAGComparisonExperiment(config).run_experiments()