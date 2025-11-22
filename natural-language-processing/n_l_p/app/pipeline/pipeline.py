# app/pipeline/sota_pipeline.py

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

from app.mongodb import papers_col
from app.nlp.summarizer import SummarizerBigBirdPegasus
from app.nlp.translator import TranslatorM2M100
from app.pipeline.text_utils import clean_summary_en, chunk_text, build_raw_text, postprocess_ko_summary


def run_sota_pipeline(limit: int = 10, top_k: int = 10) -> None:
    """
    RSRS 종합설계 SOTA 파이프라인 테스트 함수.

    - BigBird-Pegasus로 영문 요약 생성 (SUMMARY_EN)
    - NLLB로 한글 요약 생성 (SUMMARY_KO, 표시용)
    - KeyBERT + SciBERT로 영문 키워드 추출 (KEYWORDS_EN)
    - 키워드 한글 번역 (KEYWORDS_KO, 표시용)
    - sentence-transformers(all-mpnet-base-v2)로 영문 임베딩 생성
    - DB에는 어떤 것도 업데이트/저장하지 않고 결과만 출력
    """
    print("\n===== LOADING MODELS =====")

    # 1. summarizer (BigBird-Pegasus)
    summarizer = SummarizerBigBirdPegasus()

    # 2. translator (NLLB) – 임베딩은 영어만, 이건 한글 표시용
    translator = TranslatorM2M100()

    # 3. embedding model (영문 임베딩)
    embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    print("[INFO] Embedding model loaded: all-mpnet-base-v2")

    # 4. KeyBERT + SciBERT for 키워드 (영문)
    kw_model = KeyBERT("allenai/scibert_scivocab_uncased")
    print("[INFO] KeyBERT using SciBERT loaded")

    # 5. DB에서 논문 읽기 (업로드/업데이트 X)
    cursor = papers_col.find({}).limit(limit)
    total = papers_col.count_documents({})
    print(f"[INFO] Loaded {limit}/{total} papers from MongoDB")

    # ============================ PROCESS ============================
    for doc in cursor:
        print("\n" + "=" * 50)

        paper_id = doc.get("id") or doc.get("arxiv_id")
        print("[ID]", paper_id)

        title = doc.get("title", "") or ""
        raw_text = build_raw_text(doc)

        if not raw_text:
            print("[WARN] No text (abstract/body) for this paper. Skipping.")
            continue

        # 🔹 1단계: 청크별 요약
        chunks = chunk_text(raw_text, max_chars=4000)
        chunk_summaries = []
        for i, ch in enumerate(chunks):
            print(f"[DEBUG] summarizing chunk {i+1}/{len(chunks)} (len={len(ch)})")
            s = summarizer.summarize(ch)
            chunk_summaries.append(s)

        # 🔹 2단계: 청크 요약들을 다시 한 번 요약 (최종 SUMMARY_EN)
        summary_input = " ".join(chunk_summaries)
        summary_en_raw = summarizer.summarize(summary_input)
        summary_en = clean_summary_en(summary_en_raw)
        print("\n[SUMMARY_EN]\n", summary_en)

        # -------- SUMMARY KO (표시용, 임베딩 X) ----------
        summary_ko_raw = translator.translate(summary_en)
        summary_ko = postprocess_ko_summary(summary_ko_raw)
        print("\n[SUMMARY_KO]\n", summary_ko)

        # -------- KEYWORDS EN ----------
        keywords_en = [
            w for w, score in kw_model.extract_keywords(
                summary_en,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=top_k,
            )
        ]
        print("\n[KEYWORDS_EN]\n", keywords_en)

        # -------- KEYWORDS KO (표시용) ----------
        keywords_ko = [translator.translate(w) for w in keywords_en]
        print("\n[KEYWORDS_KO]\n", keywords_ko)

        # -------- EMBEDDING (영문만 사용) ------------
        text_for_emb = (title + "\n" + summary_en).strip()
        emb = embed_model.encode(
            [text_for_emb],
            normalize_embeddings=True
        )[0]
        print("\n[EMBEDDING] first 10 dims\n", emb[:10])

    print("\n[INFO] COMPLETE (NO DB UPDATE)\n")
