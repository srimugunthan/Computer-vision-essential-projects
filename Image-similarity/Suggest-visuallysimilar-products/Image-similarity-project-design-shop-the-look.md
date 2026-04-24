# Project Design: Shop the Look – Visual Style & Recommendation Engine

**Version:** 1.0  
**Status:** Draft  
**Author:** Srimugunthan  
**Domain:** Deep Metric Learning · Multimodal AI · Vector Search · Recommendation Systems

---

## 1. Overview

**Shop the Look** is a local, CPU-friendly visual recommendation engine that transforms a product image catalog into a searchable "style space" using CLIP embeddings and FAISS vector indexing. A user uploads a query image (or a query image combined with a text refinement), and the system returns the most stylistically similar products from the catalog — finding minimalist wooden furniture when shown a minimalist oak chair, not just exact chair duplicates.

The project demonstrates the full foundation model → vector database → recommendation pipeline that powers image search at Pinterest, Amazon, and Myntra — implemented entirely on a laptop with no GPU and no cloud dependency. It bridges computer vision and recommender systems, connecting directly to the analytics and behavioral modeling work that underpins finserv product recommendation and fraud similarity matching.

---

## 2. Goals and Non-Goals

### Goals

- Extract 512-dimensional CLIP (ViT-B/32) embeddings for a product image catalog (500–5,000 images).
- Normalize embeddings to unit length and index them in FAISS for sub-millisecond similarity search.
- Support image-only queries and multimodal queries (image + text refinement).
- Cluster the catalog into style groups using K-Means for unsupervised style taxonomy discovery.
- Evaluate retrieval quality using Recall@K against a hand-labeled ground truth set.
- Expose the system via a Streamlit UI for interactive querying and result visualization.
- Run entirely offline on a laptop CPU with no external API calls.

### Non-Goals

- Training or fine-tuning CLIP (zero-shot inference only for v1; fine-tuning is a v2 extension).
- Real-time catalog ingestion (batch indexing at setup time is sufficient).
- Multi-user session management or authentication (single-user local tool).
- Price or inventory data integration (pure visual similarity; metadata enrichment is a v2 feature).

---

## 3. System Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        OFFLINE: Catalog Indexing Pipeline                  │
│                                                                            │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐  │
│  │  Product Catalog │───▶│  CLIP Image      │───▶│  L2 Normalizer       │  │
│  │  (images + meta) │    │  Encoder         │    │  (unit vectors)      │  │
│  └──────────────────┘    │  (ViT-B/32)      │    └──────────┬───────────┘  │
│                          └──────────────────┘               │             │
│                                                  ┌──────────▼───────────┐  │
│                                                  │  FAISS Index         │  │
│                                                  │  (IndexFlatIP)       │  │
│                                                  └──────────┬───────────┘  │
│                                                             │             │
│                                                  ┌──────────▼───────────┐  │
│                                                  │  K-Means Clustering  │  │
│                                                  │  (style taxonomy)    │  │
│                                                  └──────────┬───────────┘  │
│                                                             │             │
│                                                  ┌──────────▼───────────┐  │
│                                                  │  Index Store         │  │
│                                                  │  (faiss.index +      │  │
│                                                  │   metadata.json +    │  │
│                                                  │   clusters.json)     │  │
│                                                  └──────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                        ONLINE: Query & Retrieval Pipeline                  │
│                                                                            │
│  ┌───────────────────┐    ┌───────────────────┐                            │
│  │  Query Image      │───▶│  CLIP Image       │                            │
│  │  (user upload)    │    │  Encoder          │──┐                         │
│  └───────────────────┘    └───────────────────┘  │                         │
│                                                  │  Vector Fusion         │
│  ┌───────────────────┐    ┌───────────────────┐  │  (weighted blend)      │
│  │  Text Refinement  │───▶│  CLIP Text        │──┘                         │
│  │  (optional)       │    │  Encoder          │                            │
│  └───────────────────┘    └───────────────────┘                            │
│                                                  │                         │
│                                       ┌──────────▼───────────┐             │
│                                       │  FAISS Search        │             │
│                                       │  (top-K dot product) │             │
│                                       └──────────┬───────────┘             │
│                                                  │                         │
│                                       ┌──────────▼───────────┐             │
│                                       │  Result Ranker       │             │
│                                       │  (score + cluster    │             │
│                                       │   diversity filter)  │             │
│                                       └──────────┬───────────┘             │
│                                                  │                         │
│                                       ┌──────────▼───────────┐             │
│                                       │  Streamlit UI        │             │
│                                       │  (image grid +       │             │
│                                       │   similarity scores) │             │
│                                       └──────────────────────┘             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Model Selection: CLIP (ViT-B/32)

### 4.1 Why CLIP

CLIP (Contrastive Language–Image Pretraining) was trained on 400 million image-text pairs using a contrastive objective that forces image and text embeddings of the same concept into nearby positions in the same 512-dimensional vector space. This means:

- A photo of a "minimalist oak chair" and the text "minimalist wooden furniture" map to nearby vectors.
- Style attributes (vintage, industrial, Scandinavian) that appear in alt-text during training are implicitly encoded in the image embeddings.
- Zero-shot: no fine-tuning needed for reasonable style similarity on product images.

### 4.2 Model Variants and Trade-offs

| Variant | Embedding Dim | CPU Inference (per image) | Notes |
|---|---|---|---|
| **ViT-B/32** (recommended) | 512 | 100–200 ms | Best laptop balance |
| ViT-B/16 | 512 | 300–500 ms | Higher visual fidelity, 3× slower |
| ViT-L/14 | 768 | 800–1500 ms | Near-SOTA, impractical on CPU |
| RN50 (ResNet backbone) | 1024 | 150–250 ms | Alternative if ViT unavailable |

**Default:** ViT-B/32 via `sentence-transformers` (`clip-ViT-B-32`).

### 4.3 Embedding Space Properties

All CLIP image embeddings are L2-normalized to unit length before indexing. Under this normalization:

```
cosine_similarity(V_Q, V_P) = V_Q · V_P
```

So cosine similarity becomes a simple dot product — the most efficient operation in FAISS. Similarity scores range from 0 (orthogonal, no style overlap) to 1 (identical style representation).

---

## 5. Component Design

### 5.1 Catalog Preprocessor (`catalog_preprocessor.py`)

Prepares the product image catalog for embedding extraction.

- **Input:** Directory of product images + optional metadata CSV (product_id, name, category, price).
- **Resize:** All images resized to 224×224 (CLIP's expected input) using PIL with `LANCZOS` resampling.
- **Center crop:** Apply center crop to remove watermarks and borders common in e-commerce images.
- **Deduplication:** Compute perceptual hash (`imagehash.phash`) for each image; skip near-duplicates (Hamming distance < 8) before embedding to avoid redundant index entries.
- **Output:** List of `(image_id, preprocessed_PIL_image, metadata_dict)` tuples passed to the embedder.

```python
@dataclass
class CatalogItem:
    image_id: str
    image_path: str
    preprocessed: PIL.Image
    metadata: dict          # {name, category, price, source_url, ...}
```

---

### 5.2 Embedding Extractor (`embedder.py`)

Runs CLIP image encoder in batches across the full catalog.

- **Batch size:** 32 images per forward pass (fits within 4–8 GB RAM comfortably).
- **Backend:** `sentence-transformers` `SentenceTransformer('clip-ViT-B-32')` for the image encoder; same model instance provides the text encoder, ensuring both live in the same embedding space.
- **Normalization:** Apply L2 normalization after extraction using `sklearn.preprocessing.normalize` or `torch.nn.functional.normalize`.
- **Progress:** `tqdm` progress bar for catalog indexing — visible when running `python index_catalog.py`.

**Throughput estimate (CPU):** At 150 ms/image and batch size 32, a 500-image catalog indexes in ~75 seconds. A 5,000-image catalog indexes in ~12 minutes — run once and cache.

```python
def embed_images(images: list[PIL.Image], batch_size: int = 32) -> np.ndarray:
    """
    Returns: float32 array of shape (N, 512), L2-normalized.
    """
```

---

### 5.3 FAISS Index Builder (`index_builder.py`)

Builds and persists the searchable vector index.

**Index type selection:**

| Index Type | Use Case | Notes |
|---|---|---|
| `IndexFlatIP` | < 10K items (exact search, inner product) | Recommended default; brute-force but fast at this scale |
| `IndexIVFFlat` | 10K–1M items | Approximate; requires training step; 10–100× faster |
| `IndexHNSWFlat` | Any size, low latency | Graph-based ANN; best recall-speed balance at scale |

**Default for this project:** `IndexFlatIP` (exact inner product search). At 500–5,000 items and 512 dimensions, a brute-force search takes < 1 ms, making approximate search unnecessary.

```python
def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]        # 512 for ViT-B/32
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)            # embeddings must be float32, L2-normalized
    return index
```

**Persisted artifacts:**

```
index_store/
├── catalog.faiss          # FAISS index file (~2 MB for 5K items at 512 dims)
├── metadata.json          # Ordered list of {image_id, name, category, path, ...}
├── embeddings.npy         # Raw embeddings matrix (for clustering, offline analysis)
└── clusters.json          # K-Means cluster assignments per image_id
```

**Memory footprint:** 5,000 items × 512 dims × 4 bytes = ~10 MB in RAM during search. Negligible for a laptop with 8+ GB RAM.

---

### 5.4 Multimodal Query Encoder (`query_encoder.py`)

Encodes user queries — image-only, text-only, or image + text combined — into a single query vector in the shared CLIP embedding space.

**Image-only query:**

```python
query_vector = clip.encode_image(query_image)
query_vector = l2_normalize(query_vector)
```

**Text-only query:**

```python
query_vector = clip.encode_text(query_text)
query_vector = l2_normalize(query_vector)
```

**Multimodal query (image + text refinement):**

CLIP's key property is that image and text embeddings live in the same space. A multimodal query vector is computed as a weighted linear combination:

```
V_query = normalize(α × V_image + (1 - α) × V_text)
```

where `α` (default: 0.7) controls how much the image drives the result vs. the text refinement. This is configurable via the UI slider.

**Examples:**

| Query Image | Text Refinement | α | Effect |
|---|---|---|---|
| Minimalist oak chair | (none) | 1.0 | Style-pure image search |
| Minimalist oak chair | "blue" | 0.7 | Similar style but biased toward blue items |
| Minimalist oak chair | "outdoor" | 0.6 | Style + material bias toward outdoor-suitable furniture |
| (none) | "Scandinavian bedroom" | 0.0 | Pure text-to-image search |

This multimodal interpolation is the distinguishing "senior-level" feature — it is the same mechanism used in production by systems like DALL-E's image variation endpoint and Pinterest's visual + text search.

---

### 5.5 Retrieval Engine (`retrieval_engine.py`)

Executes the similarity search and applies post-retrieval ranking logic.

**Core search:**

```python
def search(query_vector: np.ndarray, k: int = 20) -> list[RetrievalResult]:
    scores, indices = index.search(query_vector.reshape(1, -1), k)
    return [
        RetrievalResult(
            image_id=metadata[idx]['image_id'],
            score=float(scores[0][i]),
            metadata=metadata[idx],
            cluster_id=clusters[metadata[idx]['image_id']]
        )
        for i, idx in enumerate(indices[0])
    ]
```

**Diversity filter (optional):** Without diversity filtering, the top-K results tend to cluster around the same style sub-group — you get 10 nearly-identical minimalist chairs instead of a diverse set of minimalist furniture. A simple diversity pass ensures at most `max_per_cluster` results from any single K-Means cluster:

```python
def diversify(results: list[RetrievalResult],
              max_per_cluster: int = 2) -> list[RetrievalResult]:
    seen_clusters = defaultdict(int)
    diverse = []
    for r in results:
        if seen_clusters[r.cluster_id] < max_per_cluster:
            diverse.append(r)
            seen_clusters[r.cluster_id] += 1
    return diverse
```

This is a greedy MMR (Maximal Marginal Relevance) approximation — a technique worth naming explicitly in the portfolio write-up.

---

### 5.6 Style Clustering (`clustering.py`)

Runs K-Means on the full embedding matrix to discover unsupervised style clusters in the catalog.

**Configuration:**

```python
kmeans_params = {
    "n_clusters": 20,         # tunable; ~√N for N=500 gives ~22
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42
}
```

**Cluster interpretation:** After clustering, sample 5 representative images from each cluster and manually label them. In practice, CLIP-based clusters on product images tend to align with interpretable style attributes (e.g., "dark wood traditional," "white minimalist," "industrial metal"). These labels are added to `clusters.json` as `cluster_name` for display in the UI.

**Silhouette score:** Compute `sklearn.metrics.silhouette_score` across k ∈ {10, 15, 20, 25, 30} and plot to select the optimal cluster count. Include this plot in the evaluation notebook.

**FAISS-accelerated K-Means (optional):** For catalogs > 50K items, use `faiss.Kmeans` instead of sklearn — it is 10–50× faster and GPU-optional.

---

### 5.7 Evaluation Framework (`evaluate.py`)

#### Recall@K

For a ground truth set of (query, relevant_items) pairs, Recall@K measures how often at least one relevant item appears in the top-K results:

```
Recall@K = |{queries where ≥ 1 relevant item appears in top-K}| / |total queries|
```

**Ground truth construction:** For a 500-item catalog, manually annotate 50 query images with 3–5 "acceptable similar" items each. This is 3–5 hours of work and produces a realistic evaluation benchmark. Alternatively, use category labels from the dataset as weak ground truth (items in the same category should retrieve each other).

**Target metrics (ViT-B/32 zero-shot):**

| Metric | Target |
|---|---|
| Recall@1 | ≥ 0.40 |
| Recall@5 | ≥ 0.70 |
| Recall@10 | ≥ 0.85 |

#### Mean Reciprocal Rank (MRR)

For each query, MRR captures the rank of the first relevant result:

```
MRR = (1/|Q|) × Σ (1 / rank_of_first_relevant_result)
```

MRR is a tighter metric than Recall@K — it penalizes systems that find the right answer but bury it at position 5 rather than position 1.

#### Inference Latency Benchmark

```python
benchmark = {
    "catalog_size": [500, 1000, 5000],
    "query_latency_ms": {
        "clip_encode": measured_clip_latency,
        "faiss_search": measured_faiss_latency,
        "total": measured_total_latency
    }
}
```

**Expected results:** FAISS search latency at 5,000 items is < 1 ms regardless of catalog size (IndexFlatIP scales linearly but the constant is tiny). CLIP encoding dominates at 100–200 ms — this is the only bottleneck.

---

### 5.8 Streamlit UI (`app.py`)

A minimal interactive frontend for querying the engine.

**UI components:**

- **Image upload widget:** `st.file_uploader` accepting PNG/JPEG.
- **Text refinement field:** `st.text_input("Refine by style or color...")` — empty by default.
- **Alpha slider:** `st.slider("Image weight", 0.0, 1.0, 0.7)` — controls image vs. text blend.
- **Results grid:** Display top-K results as a 3-column image grid with similarity scores and product names beneath each image.
- **Cluster explorer tab:** A separate tab showing all style clusters as collapsible sections with representative thumbnail grids.
- **Latency display:** `st.caption(f"Query processed in {latency_ms:.0f} ms")` displayed below results.

**State management:** The FAISS index and metadata are loaded once into `st.session_state` at startup — not reloaded on each query. This keeps query latency at 100–200 ms (just CLIP encoding) rather than re-indexing.

---

## 6. Dataset Strategy

### 6.1 Recommended Datasets

| Dataset | Size | Notes |
|---|---|---|
| **Fashion Product Images (Kaggle)** | ~44K images, ~789 MB | Best for demo; has category labels for Recall@K ground truth |
| **Amazon Product Images (subset)** | Variable | Requires scraping or use of academic API |
| **Open Images V7 (household subset)** | ~5K relevant images | General objects; less e-commerce specific |
| **Custom scraped catalog** | DIY | Scrape 500 product images from any public e-commerce site |

**Recommended for local dev:** Download the Fashion Product Images small subset (~500 images, ~50 MB). This is enough to demonstrate the full pipeline and fits within a 1 GB development footprint.

### 6.2 Catalog Structure

```
catalog/
├── images/
│   ├── item_001.jpg
│   ├── item_002.jpg
│   └── ...
└── metadata.csv          # columns: image_id, name, category, subcategory, price
```

---

## 7. Project Structure

```
shop-the-look/
├── config.yaml                     # Model, index paths, clustering params, UI settings
├── index_catalog.py                # CLI: preprocess catalog + extract embeddings + build FAISS index
├── app.py                          # Streamlit UI entry point
├── catalog_preprocessor.py         # Image resizing, deduplication, metadata loading
├── embedder.py                     # CLIP batch embedding extractor
├── index_builder.py                # FAISS index construction and persistence
├── query_encoder.py                # Image / text / multimodal query encoding
├── retrieval_engine.py             # FAISS search + diversity filter
├── clustering.py                   # K-Means style clustering + silhouette analysis
├── evaluate.py                     # Recall@K, MRR, latency benchmarking
├── catalog/
│   ├── images/                     # Product images (gitignored)
│   └── metadata.csv
├── index_store/
│   ├── catalog.faiss               # Persisted FAISS index (gitignored)
│   ├── metadata.json
│   ├── embeddings.npy              # Raw embedding matrix (gitignored)
│   └── clusters.json
├── ground_truth/
│   └── eval_pairs.json             # Hand-labeled (query, relevant_items) pairs
├── notebooks/
│   ├── 01_eda_catalog.ipynb        # Catalog exploration and image distribution
│   ├── 02_embedding_analysis.ipynb # t-SNE / UMAP visualization of style space
│   ├── 03_cluster_explorer.ipynb   # K-Means cluster analysis and labeling
│   └── 04_evaluation.ipynb         # Recall@K, MRR, benchmark results
├── tests/
│   ├── test_embedder.py            # Unit: output shape and normalization
│   ├── test_retrieval.py           # Unit: top-1 result for known similar pair
│   └── test_clustering.py          # Unit: cluster count matches config
├── pyproject.toml                  # uv-managed dependencies
└── README.md
```

---

## 8. Setup and Run

```bash
# 1. Create environment
uv venv
source .venv/bin/activate

# 2. Install dependencies
uv pip install torch torchvision sentence-transformers faiss-cpu \
               streamlit pillow imagehash pandas scikit-learn tqdm

# 3. Optional: UMAP for embedding visualization notebook
uv pip install umap-learn matplotlib

# 4. Index the catalog (run once)
python index_catalog.py --catalog catalog/ --output index_store/ --clusters 20

# 5. Launch the Streamlit UI
streamlit run app.py

# 6. Run evaluation
python evaluate.py --ground-truth ground_truth/eval_pairs.json \
                   --k 1 5 10 \
                   --report outputs/eval_report.json

# 7. Run latency benchmark
python evaluate.py --benchmark --catalog-sizes 500 1000 5000
```

---

## 9. Configuration (`config.yaml`)

```yaml
model:
  clip_variant: clip-ViT-B-32     # sentence-transformers model name
  embedding_dim: 512
  batch_size: 32

catalog:
  images_dir: catalog/images/
  metadata_csv: catalog/metadata.csv
  dedup_hamming_threshold: 8       # perceptual hash distance; lower = stricter dedup

index:
  store_dir: index_store/
  faiss_index_type: IndexFlatIP    # options: IndexFlatIP | IndexIVFFlat | IndexHNSWFlat
  nprobe: 10                       # only for IndexIVFFlat (number of clusters to probe)

clustering:
  n_clusters: 20
  n_init: 10
  random_state: 42
  max_per_cluster_in_results: 2   # diversity filter cap

query:
  default_k: 12                   # number of results returned
  default_alpha: 0.7              # image weight in multimodal fusion (0=text only, 1=image only)
  diversity_filter: true

evaluation:
  k_values: [1, 5, 10]
  ground_truth_path: ground_truth/eval_pairs.json
```

---

## 10. Key Design Decisions

### 10.1 IndexFlatIP over IndexFlatL2

Both are exact brute-force search. `IndexFlatIP` (inner product) is preferred because after L2 normalization, inner product equals cosine similarity — the natural metric for style comparison. `IndexFlatL2` (Euclidean distance) on normalized vectors gives equivalent rankings but requires a sign flip and is slightly less interpretable (higher L2 = more different, vs. lower IP = more different). Using IP means the similarity score is directly human-readable: 0.95 means "very similar style."

### 10.2 Why L2 Normalize Before Indexing

If embeddings are not normalized, the magnitude of a vector (driven by how "confident" the model is about an image's features) pollutes the similarity score. Two minimalist chairs with different lighting conditions might have very different magnitudes but near-identical directions. Normalization ensures only direction — i.e., style — contributes to similarity.

### 10.3 Alpha Weighting in Multimodal Fusion

The weighted blend `α × V_image + (1-α) × V_text` works because CLIP's contrastive training objective explicitly aligns image and text embeddings in the same space. This is not true of arbitrary image and text encoders — it is a property specific to CLIP-family models. This is worth explaining in the README as it demonstrates understanding of the model's architecture, not just its API.

### 10.4 Greedy Diversity Filter over Pure Score Ranking

Returning the top-12 results by raw score typically returns 12 nearly-identical items (the catalog's closest neighbors cluster tightly). A greedy MMR-style diversity filter that caps results per cluster at `max_per_cluster = 2` produces a more useful recommendation grid with varied items that all match the style query. This is the same principle used in news feed diversification and playlist generation.

---

## 11. Embedding Space Visualization

The `02_embedding_analysis.ipynb` notebook reduces the 512-dimensional embedding space to 2D using UMAP for visual inspection:

```python
import umap
reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

# Plot colored by category label
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=category_colors, alpha=0.6)
```

**Expected output:** Products cluster by visual style rather than category — minimalist white chairs and minimalist white tables appear near each other, even though they are different product types. This is the visual proof that the embedding space captures "style" rather than just object identity. It is the single most compelling visualization for this project's portfolio page.

---

## 12. Learning Objectives Mapped to Implementation

| Learning Objective | Where Implemented |
|---|---|
| Foundation model as feature extractor | `embedder.py` — CLIP ViT-B/32 zero-shot embedding |
| Vector normalization and cosine similarity | `embedder.py` — L2 normalize; `index_builder.py` — IndexFlatIP |
| Vector database indexing and search | `index_builder.py` + `retrieval_engine.py` — FAISS |
| Multimodal query fusion | `query_encoder.py` — α-weighted image + text vector blend |
| Unsupervised style clustering | `clustering.py` — K-Means + silhouette analysis |
| Retrieval evaluation metrics | `evaluate.py` — Recall@K and MRR |
| Diversity in recommendation results | `retrieval_engine.py` — greedy MMR-style cluster cap |
| Interactive ML product demo | `app.py` — Streamlit UI with image upload and result grid |

---

## 13. Transferable Skills Framing

**Vector similarity → Fraud entity matching.** FAISS nearest-neighbor search over CLIP embeddings is architecturally identical to entity resolution in AML: embed merchant descriptions into a vector space, index known fraud merchants, and flag new merchants whose embedding falls within a similarity threshold of a known bad actor. The same pipeline — encode, normalize, index, search — applies directly.

**Style clusters → Behavioral segmentation.** K-Means over CLIP embeddings produces interpretable style groups without labeled training data. This is the same technique used to segment customers by transaction behavior or to group fraud alerts by attack pattern — unsupervised clustering over dense representation vectors with post-hoc human labeling.

**Recall@K → Information retrieval evaluation in NLP.** Recall@K is the standard metric for retrieval-augmented generation (RAG) pipelines — the same metric used to evaluate whether a document retriever surfaces the right context for an LLM. Knowing how to construct ground truth pairs, compute Recall@K, and interpret the results is directly transferable to evaluating any RAG system.

**Multimodal fusion → Tabular + text feature fusion.** The α-weighted CLIP vector blend is a simple instance of multimodal feature fusion. The same principle applies in finserv: blending a structured transaction embedding (tabular features encoded via an MLP) with a text embedding (merchant description via a language model) to improve fraud classification. The vector addition is identical.

---

## 14. Portfolio Narrative

**Foundation model literacy:** This project demonstrates fluency with CLIP — not just "I called the API" but understanding why L2 normalization is required, why IP search is preferred over L2, and why the shared embedding space enables multimodal fusion. These are the kinds of architectural understanding questions that distinguish a practitioner from a tutorial follower.

**Production-grade vector search:** FAISS is the production vector search library used at Meta, not a toy demonstration. Documenting the trade-offs between IndexFlatIP (exact, small catalog) and IndexIVFFlat / IndexHNSWFlat (approximate, large catalog) shows awareness of scalability constraints that arise in real deployments.

**Quantitative evaluation:** Recall@K and MRR with a hand-labeled ground truth set makes this measurably better or worse, not just subjectively "looks good." The evaluation notebook is the artifact that elevates this from a demo into a DS project.

**Bangalore AI ecosystem relevance:** Visual search and style recommendation are active engineering problems at Flipkart, Myntra, Meesho, and Urban Ladder — all headquartered in Bangalore. Framing the README explicitly in that context makes this a locally resonant portfolio piece.

---

## 15. Open Questions and Future Work

- **v1.1:** Add product metadata filtering as a hard constraint on top of soft similarity — e.g., "only return items in category=furniture with price < ₹15,000." This is a standard re-ranking pattern in production recommender systems.
- **v1.2:** Fine-tune CLIP on a domain-specific dataset (e.g., Indian ethnic wear) using a contrastive loss on triplets (anchor, positive, negative). Measure Recall@K improvement vs. zero-shot baseline — the delta is the portfolio-worthy finding.
- **v2.0:** Replace K-Means with HDBSCAN (density-based clustering) to automatically discover the number of style clusters rather than fixing k. HDBSCAN is more appropriate for non-spherical style manifolds.
- **v2.1:** Add a "negative feedback" mechanism: if the user clicks "not like this," compute the embedding of the rejected item and subtract it from the query vector before re-searching. This is the vector arithmetic approach to preference refinement.
- **Research angle:** Treat the retrieval problem as an adversarial robustness question — can a small adversarial perturbation to a product image change its nearest neighbors entirely without visibly changing the image? Connects directly to your adversarial ML background and is a publishable research direction in the context of e-commerce fraud (fake product listings that evade visual similarity filters).
