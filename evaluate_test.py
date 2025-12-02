import pymongo
import evaluate
from comet import download_model, load_from_checkpoint
import pandas as pd

MONGO_URI = "mongodb://rsrs-root:KIQu3jebjHNhTEE6mm5tgj2oNjYr7J805k2JLbE0AVo@35.87.92.19:27017/arxiv?authSource=admin"
DB_NAME = "arxiv"
COLLECTION_NAME = "papers"

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
col = db[COLLECTION_NAME]

CS_CATEGORIES = [
    "cs.AI","cs.AR","cs.CC","cs.CE","cs.CG","cs.CL","cs.CR","cs.CV","cs.CY",
    "cs.DB","cs.DC","cs.DL","cs.DM","cs.DS","cs.ET","cs.FL","cs.GL","cs.GR",
    "cs.GT","cs.HC","cs.IR","cs.IT","cs.LG","cs.LO","cs.MA","cs.MM","cs.MS",
    "cs.NA","cs.NE","cs.NI","cs.OH","cs.OS","cs.PF","cs.PL","cs.RO","cs.SC",
    "cs.SD","cs.SE","cs.SI","cs.SY"
]

ABSTRACT_FIELDS = ["abstract", "summary_en_raw", "raw_abstract"]

def get_abstract(doc):
    for f in ABSTRACT_FIELDS:
        if f in doc and isinstance(doc[f], str) and doc[f].strip():
            return doc[f].strip()
    return ""

cursor = (
    col.find(
        {
            "categories": {"$in": CS_CATEGORIES},
            "summary.en": {"$exists": True},
            "summary.ko": {"$exists": True},
            "abstract": {"$exists": True},  
            "view_count": {"$exists": True}
        }
    )
    .sort("view_count", -1)
    .limit(3000)
)

print("[INFO] Number of docs to evaluate:", cursor.count() if hasattr(cursor, "count") else "cursor ready")

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

comet_path = download_model("Unbabel/wmt20-comet-qe-da")
comet_model = load_from_checkpoint(comet_path)

records = []

for doc in cursor:
    paper_id = doc.get("paper_id") or str(doc.get("_id"))
    abstract = get_abstract(doc)
    summary_en = doc["summary"]["en"].strip()
    summary_ko = doc["summary"]["ko"].strip()

    if not abstract or not summary_en or not summary_ko:
        continue
    
    try:
        rouge_out = rouge.compute(predictions=[summary_en], references=[abstract])
        rouge_l = float(rouge_out["rougeL"])
    except Exception as e:
        print(f"[WARN] ROUGE 계산 실패: {paper_id}, 이유: {e}")
        rouge_l = None
        
    try:
        bert_out = bertscore.compute(
            predictions=[summary_en],
            references=[abstract],
            lang="en"
        )
        bert_f1 = float(bert_out["f1"][0])

    except Exception as e:
        print(f"[WARN] BERTScore 계산 실패: {paper_id}, 이유: {e}")
        bert_f1 = None   # 또는 0.0

    

    try:
        comet_data = [{"src": abstract, "mt": summary_ko}]
        comet_score = float(comet_model.predict(comet_data, batch_size=8).scores[0])
    except Exception as e:
        print(f"[WARN] COMET 계산 실패: {paper_id}, 이유: {e}")
        comet_score = None

    records.append({
        "paper_id": paper_id,
        "view_count": doc["view_count"],
        "rougeL": rouge_l,
        "bert_f1": bert_f1,
        "comet_qe": comet_score,
    })


df = pd.DataFrame(records)
df.to_csv("cs_eval_1000_sorted.csv", index=False, encoding="utf-8-sig")

print("===== DONE =====")
print(df.head())
print("평균 ROUGE-L:", df["rougeL"].mean())
print("평균 BERTScore F1:", df["bert_f1"].mean())
print("평균 COMET-QE:", df["comet_qe"].mean())