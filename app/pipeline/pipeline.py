# app/pipeline/pipeline.py
import warnings
import logging
import datetime
import transformers
from sentence_transformers import SentenceTransformer, LoggingHandler
from keybert import KeyBERT

from app.mongodb import papers_col
from app.nlp.summarizer import SummarizerBigBirdPegasus
from app.nlp.translator import TranslatorM2M100
from app.pipeline.text_utils import (
    clean_summary_en,
    chunk_text,
    build_raw_text,
)
from app.nlp.refiner import LocalKoreanRefiner

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()

def run_sota_pipeline(limit: int = 10, top_k: int = 10) -> None:
    print("\n===== LOADING MODELS =====")

    summarizer = SummarizerBigBirdPegasus()
    translator = TranslatorM2M100()
    refiner = LocalKoreanRefiner()

    embed_model = SentenceTransformer("models/all-mpnet-base-v2")
    print("[INFO] Embedding model loaded: all-mpnet-base-v2")

    kw_model = KeyBERT("models/scibert")
    print("[INFO] KeyBERT using SciBERT loaded")
    
    cs_categories = [
        "cs.AI","cs.AR","cs.CC","cs.CE","cs.CG","cs.CL","cs.CR","cs.CV","cs.CY",
        "cs.DB","cs.DC","cs.DL","cs.DM","cs.DS","cs.ET","cs.FL","cs.GL","cs.GR",
        "cs.GT","cs.HC","cs.IR","cs.IT","cs.LG","cs.LO","cs.MA","cs.MM","cs.MS",
        "cs.NA","cs.NE","cs.NI","cs.OH","cs.OS","cs.PF","cs.PL","cs.RO","cs.SC",
        "cs.SD","cs.SE","cs.SI","cs.SY"
    ]
    
    cs_filter = {
        "categories": {
            "$elemMatch": {"$in": cs_categories}
        }
    }
    # count = papers_col.count_documents(cs_filter)
    # print("CS 논문 개수:", count)
    if limit is None:
        cursor = (
            papers_col
            .find(cs_filter)
            .sort("view_count", -1)
        )
        print("[INFO] Prepared cursor for ALL cs.* papers (sorted by view_count desc)")
    else:
        cursor = (
            papers_col
            .find(cs_filter)
            .sort("view_count", -1)
            .limit(limit)
        )
        print(f"[INFO] Prepared cursor for up to {limit} cs.* papers (sorted by view_count desc)")

    # ============================ PROCESS ============================
    for doc in cursor:
        print("\n" + "=" * 50)

        paper_id = doc.get("id") or doc.get("_id")
        print("[ID]", paper_id)
        
        if doc.get("summary_refined") is True:
            print(f"[INFO] Skip (already refined): {paper_id}")
            continue
        
     
        
        title = doc.get("title", "") or ""
        raw_text = build_raw_text(doc)

        chunks = chunk_text(raw_text, max_chars=4000)
        chunk_summaries: list[str] = []
        for i, ch in enumerate(chunks):
            s = summarizer.summarize(ch)
            if s:
                chunk_summaries.append(s)

        summary_input = " ".join(chunk_summaries)
        summary_en_raw = summarizer.summarize(summary_input)
        summary_en = clean_summary_en(summary_en_raw)
        print("\n[SUMMARY_EN]\n", summary_en)

        summary_ko_raw = translator.translate(summary_en)
        # print("\n[SUMMARY_KO_RAW]\n", summary_ko_raw)

        keywords_en = [
            w
            for w, score in kw_model.extract_keywords(
                summary_en,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=top_k,
            )
        ]
        print("\n[KEYWORDS_EN]\n", keywords_en)
        
        summary_en_refined, summary_ko = refiner.refine_both(
            summary_en_raw=summary_en,    
            summary_ko_raw=summary_ko_raw,
            keywords_en=keywords_en,
        
        )
        print("\n[SUMMARY_EN_refined]\n", summary_en_refined)
        
        print("\n[SUMMARY_KO]\n", summary_ko)
       
        text_for_emb = "\n".join(keywords_en).strip()

        emb = embed_model.encode(
            [text_for_emb],
            normalize_embeddings=True,
        )[0]
        # print("\n[EMBEDDING] first 10 dims\n", emb[:10])
        
        update_doc = {
            "keywords": keywords_en,
            "summary": {
                "ko": summary_ko,
                "en": summary_en_refined,
            },
            "embedding_vector": emb.tolist(),
            "summary_refined": True,
        }

        papers_col.update_one(
            {"_id": doc["_id"]},
            {"$set": update_doc}
        )

        print(f"[INFO] Updated document: {paper_id}")

print("\n[INFO] COMPLETE (DB UPDATED)\n")
