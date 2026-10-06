"""Flask API 路由。

把所有 NLP 能力、存储与流水线编排暴露为 REST 接口，
前端 10 个页面通过 ``fetch`` 调用这些接口。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Optional

from flask import Blueprint, current_app, jsonify, request

from nlp import (get_constituency_parser, get_embeddings, get_keywords, get_ner,
                 get_parser, get_segmenter, get_sentiment, get_summarizer,
                 get_tagger, get_translator, get_entity_resolver,
                 ENTITY_TYPE_NAMES, TAG_NAMES, SENSE_TYPE_NAMES,
                 DEP_REL_NAMES, PHRASE_NAMES, POLARITY_NAMES)
from nlp.entitylink import BUILTIN_KB
from nlp.lexicon import STOPWORDS
from storage import StoreRegistry


api = Blueprint("api", __name__, url_prefix="/api")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _registry() -> StoreRegistry:
    return current_app.config["STORE_REGISTRY"]


def _engine():
    return current_app.config["PIPELINE_ENGINE"]


def _models_dir() -> str:
    import os
    path = os.path.join(current_app.config["DATA_ROOT"], "models")
    os.makedirs(path, exist_ok=True)
    return path


def _store_result(task: str, text: str, result: dict,
                  corpus_id: Optional[str] = None) -> str:
    record = {"text": text, "result": result, "created_at": time.time()}
    if corpus_id:
        record["corpus_id"] = corpus_id
    return _registry().task(task).insert(record)


def _payload() -> dict:
    data = request.get_json(silent=True) or {}
    return data


def _resolve_text(data: dict) -> tuple[str, Optional[str]]:
    """从请求中取文本：优先 text，其次 corpus_id。"""
    if data.get("text"):
        return data["text"], data.get("corpus_id")
    corpus_id = data.get("corpus_id")
    if corpus_id:
        record = _registry().task("corpus").get(corpus_id)
        if record:
            return record.get("text", ""), corpus_id
        return "", corpus_id
    return "", None


def _clean(text: str, remove_stopwords: bool = True) -> dict:
    text = re.sub(r"\s+", " ", text).strip()
    seg = get_segmenter()
    words = seg.cut(text)
    if remove_stopwords:
        kept = [w for w in words if w not in STOPWORDS]
    else:
        kept = words
    removed = len(words) - len(kept)
    return {
        "text": text,
        "cleaned": " ".join(kept),
        "tokens": kept,
        "original_tokens": words,
        "removed_stopwords": removed,
    }


# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------

@api.get("/status")
def status():
    return jsonify({
        "ok": True,
        "version": "1.0.0",
        "tasks": _registry().tasks(),
        "time": time.time(),
    })


@api.get("/meta")
def meta():
    """给前端提供标签集合与可配置参数。"""
    return jsonify({
        "tag_names": TAG_NAMES,
        "dep_rel_names": DEP_REL_NAMES,
        "phrase_names": PHRASE_NAMES,
        "entity_type_names": ENTITY_TYPE_NAMES,
        "entity_sense_names": SENSE_TYPE_NAMES,
        "polarity_names": POLARITY_NAMES,
        "directions": [{"id": "zh2en", "name": "中文 → 英文"},
                       {"id": "en2zh", "name": "英文 → 中文"}],
    })


# ---------------------------------------------------------------------------
# 语料库管理
# ---------------------------------------------------------------------------

@api.get("/corpus")
def list_corpus():
    records = _registry().task("corpus").all()
    items = [{
        "id": r.get("id"),
        "name": r.get("name", "未命名"),
        "length": len(r.get("text", "")),
        "created_at": r.get("created_at"),
        "preview": r.get("text", "")[:80],
    } for r in records if not r.get("_deleted")]
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return jsonify({"corpora": items})


@api.post("/corpus")
def create_corpus():
    data = _payload()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "语料内容不能为空"}), 400
    record = {
        "name": data.get("name") or f"语料_{int(time.time())}",
        "text": text,
        "created_at": time.time(),
    }
    rid = _registry().task("corpus").insert(record)
    return jsonify({"id": rid, "ok": True})


@api.post("/corpus/upload")
def upload_corpus():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "未接收到文件"}), 400
    raw = file.read()
    text = None
    for enc in ("utf-8", "gbk", "gb18030", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        return jsonify({"error": "无法解码文件内容"}), 400
    name = data_name = file.filename or "上传文件"
    record = {"name": name, "text": text.strip(), "created_at": time.time()}
    rid = _registry().task("corpus").insert(record)
    return jsonify({"id": rid, "name": name, "length": len(text), "ok": True})


@api.get("/corpus/<cid>")
def get_corpus(cid: str):
    record = _registry().task("corpus").get(cid)
    if not record:
        return jsonify({"error": "语料不存在"}), 404
    return jsonify(record)


@api.delete("/corpus/<cid>")
def delete_corpus(cid: str):
    ok = _registry().task("corpus").delete(cid)
    return jsonify({"ok": ok})


@api.post("/corpus/<cid>/clean")
def clean_corpus(cid: str):
    record = _registry().task("corpus").get(cid)
    if not record:
        return jsonify({"error": "语料不存在"}), 404
    data = _payload()
    result = _clean(record.get("text", ""), data.get("remove_stopwords", True))
    _store_result("clean", record.get("text", ""), result, corpus_id=cid)
    return jsonify(result)


# ---------------------------------------------------------------------------
# 分词与词性标注
# ---------------------------------------------------------------------------

@api.post("/segment")
def segment():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    seg = get_segmenter()
    words = seg.cut(text)
    result = {"words": words, "count": len(words)}
    rid = _store_result("segment", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


@api.post("/pos")
def pos_tag():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    tagger = get_tagger()
    tokens = [[w, t] for w, t in tagger.tag(text)]
    result = {"tokens": tokens, "tag_names": TAG_NAMES}
    rid = _store_result("pos", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 句法分析
# ---------------------------------------------------------------------------

@api.post("/parse")
def parse():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    dep = get_parser().parse(text)
    const = get_constituency_parser().parse(text)
    result = {
        "dependency": dep,
        "constituency": const,
        "dep_rel_names": DEP_REL_NAMES,
        "phrase_names": PHRASE_NAMES,
    }
    rid = _store_result("parse", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 命名实体识别与标注
# ---------------------------------------------------------------------------

@api.post("/ner")
def ner():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    entities = get_ner().recognize(text)
    result = {"entities": entities, "entity_type_names": ENTITY_TYPE_NAMES}
    rid = _store_result("ner", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


@api.post("/ner/annotate")
def ner_annotate():
    data = _payload()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    record = {
        "text": text,
        "entities": data.get("entities", []),
        "note": data.get("note", ""),
        "created_at": time.time(),
    }
    rid = _registry().task("annotation").insert(record)
    return jsonify({"id": rid, "ok": True})


@api.get("/ner/annotations")
def ner_annotations():
    records = _registry().task("annotation").all()
    return jsonify({"annotations": records})


# ---------------------------------------------------------------------------
# 实体消歧 / 链接 / 跨文档对齐
# ---------------------------------------------------------------------------
#
# 存储布局（复用分片 JSON 存储）：
# - ``entity`` 任务：规范实体记录（内置知识库 + 动态新建），id 即 entity_id，
#   画像与特征词随链接增量更新；
# - ``mention`` 任务：实体提及记录，含语境快照（context/domains/co_entities），
#   status 为 linked / pending（待定）。已链接记录稳定不改判，
#   待定记录可在知识库扩充后重估升级。

def _entity_resolver():
    """返回与 ``entity`` 存储同步过的实体对齐器（进程内单例）。"""
    resolver = get_entity_resolver()
    if not current_app.config.get("ENTITY_STORE_SYNCED"):
        store = _registry().task("entity")
        if not store.all():
            # 首次使用：把内置知识库落库，保证 entity_id 跨重启稳定
            store.insert_many([dict(e, id=e["entity_id"]) for e in BUILTIN_KB])
        resolver.load_entries(
            [r for r in store.all() if not r.get("_deleted")])
        current_app.config["ENTITY_STORE_SYNCED"] = True
    return resolver


def _persist_entity_mutations(mutations: dict) -> None:
    """把对齐器返回的变更集写入 ``entity`` 分片存储。"""
    store = _registry().task("entity")
    for entry in mutations.get("new_entities", []):
        store.insert(dict(entry, id=entry["entity_id"]))
    for entity_id, entry in mutations.get("updated_entities", {}).items():
        store.update(entity_id, entry)


def _entity_view(record: dict) -> dict:
    return {
        "entity_id": record.get("entity_id") or record.get("id"),
        "canonical": record.get("canonical"),
        "aliases": record.get("aliases", []),
        "sense_type": record.get("sense_type"),
        "sense_name": SENSE_TYPE_NAMES.get(record.get("sense_type"), "未知"),
        "domains": record.get("domains", []),
        "mention_count": record.get("mention_count", 0),
        "source": record.get("source"),
        "merged_into": record.get("merged_into"),
    }


@api.post("/entity/resolve")
def entity_resolve():
    """消歧并链接一段文本中的实体提及（结果持久化为 mention 记录）。"""
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    resolver = _entity_resolver()
    doc_id = data.get("doc_id") or cid
    result = resolver.resolve_document(text, doc_id=doc_id)
    _persist_entity_mutations(result["mutations"])
    mentions = result["mentions"]
    if mentions and data.get("persist", True):
        now = time.time()
        ids = _registry().task("mention").insert_many(
            [dict(m, created_at=now) for m in mentions])
        for m, mid in zip(mentions, ids):
            m["id"] = mid
    result["sense_type_names"] = SENSE_TYPE_NAMES
    return jsonify(result)


@api.post("/entity/ingest")
def entity_ingest():
    """把语料库文档批量入库做实体对齐（幂等：已入库的文档默认跳过）。

    同一对象散落在多篇文档时聚到同一 entity_id 下；同名不同对象各自
    另立实体。已链接的关系不因新文档入库而改变。
    """
    data = _payload()
    store = _registry().task("corpus")
    corpus_ids = data.get("corpus_ids")
    if corpus_ids:
        records = [store.get(c) for c in corpus_ids]
        records = [r for r in records if r and not r.get("_deleted")]
    else:
        records = [r for r in store.all() if not r.get("_deleted")]
    if not records:
        return jsonify({"error": "没有可处理的语料"}), 400

    resolver = _entity_resolver()
    mention_store = _registry().task("mention")
    force = bool(data.get("force"))
    summary = {"docs": 0, "skipped": 0, "mentions": 0,
               "linked": 0, "pending": 0, "new_entities": 0}
    for rec in records:
        doc_id = rec["id"]
        if not force and mention_store.query(
                where=[("doc_id", "eq", doc_id)], limit=1):
            summary["skipped"] += 1
            continue
        result = resolver.resolve_document(rec.get("text", ""), doc_id=doc_id)
        _persist_entity_mutations(result["mutations"])
        mentions = result["mentions"]
        if mentions:
            now = time.time()
            mention_store.insert_many([dict(m, created_at=now) for m in mentions])
        summary["docs"] += 1
        summary["mentions"] += len(mentions)
        summary["linked"] += sum(1 for m in mentions if m["status"] == "linked")
        summary["pending"] += sum(1 for m in mentions if m["status"] == "pending")
        summary["new_entities"] += len(result["mutations"]["new_entities"])
    return jsonify({"ok": True, **summary})


@api.get("/entities")
def list_entities():
    store = _registry().task("entity")
    _entity_resolver()  # 确保已同步
    items = [_entity_view(r) for r in store.all() if not r.get("_deleted")]
    items.sort(key=lambda x: (x["merged_into"] is not None, -x["mention_count"]))
    return jsonify({"entities": items, "sense_type_names": SENSE_TYPE_NAMES})


@api.get("/entities/<eid>")
def get_entity(eid: str):
    resolver = _entity_resolver()
    store = _registry().task("entity")
    record = store.get(eid)
    if not record or record.get("_deleted"):
        return jsonify({"error": "实体不存在"}), 404
    view = _entity_view(record)
    view["cues"] = record.get("cues", {})
    view["profile"] = record.get("profile", {})
    view["related"] = record.get("related", [])
    # 跨文档提及：沿重定向链取最终实体名下的所有提及
    final = resolver.get_entry(eid)
    final_id = final["entity_id"] if final else eid
    mentions = _registry().task("mention").query(
        where=[("entity_id", "eq", final_id)], order_by="created_at")
    view["mentions"] = mentions
    view["doc_count"] = len({m.get("doc_id") for m in mentions})
    return jsonify(view)


@api.get("/entity/mentions")
def list_mentions():
    where = []
    for key in ("status", "doc_id", "entity_id"):
        val = request.args.get(key)
        if val:
            where.append((key, "eq", val))
    records = _registry().task("mention").query(
        where=where or None, order_by="created_at", order="desc",
        limit=request.args.get("limit", 200, type=int))
    return jsonify({"mentions": records, "count": len(records)})


@api.post("/entity/reresolve")
def entity_reresolve():
    """重估全部待定提及：知识库扩充后只升级、不改判已链接结果。"""
    resolver = _entity_resolver()
    mention_store = _registry().task("mention")
    pending = mention_store.query(where=[("status", "eq", "pending")])
    upgraded = 0
    for rec in pending:
        out = resolver.reresolve(rec)
        if not out:
            continue
        _persist_entity_mutations(out["mutations"])
        mention_store.update(rec["id"], {
            "status": "linked", "entity_id": out["entity_id"],
            "canonical": out["canonical"], "sense_type": out["sense_type"],
            "score": out["score"], "margin": out["margin"],
            "confidence": out["confidence"], "candidates": out["candidates"],
            "reresolved": True,
        })
        upgraded += 1
    return jsonify({"ok": True, "checked": len(pending), "upgraded": upgraded})


@api.post("/entities/<eid>/assign")
def entity_assign(eid: str):
    """人工把一条（待定）提及指派给实体，画像加倍吸收该语境。"""
    data = _payload()
    mention_id = data.get("mention_id")
    if not mention_id:
        return jsonify({"error": "缺少 mention_id"}), 400
    resolver = _entity_resolver()
    entity_store = _registry().task("entity")
    mention_store = _registry().task("mention")
    entity = entity_store.get(eid)
    mention = mention_store.get(mention_id)
    if not entity or entity.get("_deleted") or entity.get("merged_into"):
        return jsonify({"error": "实体不存在或已被合并"}), 404
    if not mention or mention.get("_deleted"):
        return jsonify({"error": "提及不存在"}), 404
    mutations = resolver.manual_link(mention, eid)
    _persist_entity_mutations(mutations)
    mention_store.update(mention_id, {
        "status": "linked", "entity_id": eid,
        "canonical": entity.get("canonical"),
        "sense_type": entity.get("sense_type"),
        "confidence": 1.0, "source": "manual",
    })
    return jsonify({"ok": True})


@api.post("/entities/<eid>/alias")
def entity_add_alias(eid: str):
    """给实体登记别名：同一对象的另一种叫法对上号。"""
    data = _payload()
    alias = (data.get("alias") or "").strip()
    if not alias:
        return jsonify({"error": "缺少别名"}), 400
    resolver = _entity_resolver()
    try:
        mutations = resolver.add_alias(eid, alias)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    _persist_entity_mutations(mutations)
    return jsonify({"ok": True})


@api.post("/entities/merge")
def entity_merge():
    """显式合并两个实体：源实体的提及全部改挂目标，源留重定向。"""
    data = _payload()
    src_id, dst_id = data.get("src_id"), data.get("dst_id")
    if not src_id or not dst_id:
        return jsonify({"error": "缺少 src_id 或 dst_id"}), 400
    resolver = _entity_resolver()
    try:
        mutations = resolver.merge_entities(src_id, dst_id)
    except (KeyError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    _persist_entity_mutations(mutations)
    dst = mutations["updated_entities"][dst_id]
    mention_store = _registry().task("mention")
    moved = 0
    for m in mention_store.query(where=[("entity_id", "eq", src_id)]):
        mention_store.update(m["id"], {
            "entity_id": dst_id, "canonical": dst.get("canonical"),
            "sense_type": dst.get("sense_type"), "merged_from": src_id,
        })
        moved += 1
    return jsonify({"ok": True, "moved_mentions": moved})


@api.get("/entity/stats")
def entity_stats():
    resolver = _entity_resolver()
    mention_store = _registry().task("mention")
    mentions = mention_store.all()
    live = [m for m in mentions if not m.get("_deleted")]
    return jsonify({
        "entities": resolver.stats(),
        "mentions": {
            "total": len(live),
            "linked": sum(1 for m in live if m.get("status") == "linked"),
            "pending": sum(1 for m in live if m.get("status") == "pending"),
            "docs": len({m.get("doc_id") for m in live}),
        },
        "sense_type_names": SENSE_TYPE_NAMES,
    })


# ---------------------------------------------------------------------------
# 情感分析
# ---------------------------------------------------------------------------

@api.post("/sentiment")
def sentiment():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_sentiment().analyze(text)
    result["polarity_name"] = POLARITY_NAMES.get(result["polarity"], "")
    rid = _store_result("sentiment", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 文本摘要
# ---------------------------------------------------------------------------

@api.post("/summary")
def summary():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_summarizer().summarize(
        text, ratio=data.get("ratio", 0.3),
        max_sentences=data.get("max_sentences"))
    rid = _store_result("summary", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 机器翻译（模拟）
# ---------------------------------------------------------------------------

@api.post("/translate")
def translate():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_translator().translate(text, direction=data.get("direction", "zh2en"))
    rid = _store_result("translate", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 关键词提取
# ---------------------------------------------------------------------------

@api.post("/keywords")
def keywords():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_keywords().extract(text, top_k=data.get("top_k", 10),
                                    method=data.get("method", "hybrid"))
    rid = _store_result("keywords", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 词向量
# ---------------------------------------------------------------------------

def _embedding_path() -> str:
    import os
    return os.path.join(_models_dir(), "embeddings.json")


@api.post("/embeddings/train")
def train_embeddings():
    data = _payload()
    corpus_ids = data.get("corpus_ids")
    store = _registry().task("corpus")
    if corpus_ids:
        texts = [store.get(c)["text"] for c in corpus_ids if store.get(c)]
    else:
        texts = [r["text"] for r in store.all() if not r.get("_deleted")]
    if not texts:
        return jsonify({"error": "没有可用语料，请先上传语料"}), 400

    emb = get_embeddings()
    emb.train(texts, vocab_size=data.get("vocab_size", 200),
              dim=data.get("dim", 20), window=data.get("window", 5),
              min_count=data.get("min_count", 1))

    payload = {
        "vocab": emb.vocab,
        "vectors": emb.vectors,
        "dim": emb.dim,
        "trained_at": time.time(),
        "corpus_count": len(texts),
    }
    with open(_embedding_path(), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return jsonify(emb.stats())


@api.get("/embeddings/vectors")
def embeddings_vectors():
    emb = get_embeddings()
    if not emb.vectors:
        _load_embeddings()
        emb = get_embeddings()
    if not emb.vectors:
        return jsonify({"error": "尚未训练词向量"}), 404
    n_clusters = int(request.args.get("clusters", 5))
    proj = emb.project_2d()
    clusters = emb.cluster(n_clusters)
    return jsonify({
        "points": [{"word": w, "x": round(p[0], 4), "y": round(p[1], 4),
                    "cluster": clusters.get(w, 0)} for w, p in proj.items()],
        "stats": emb.stats(),
    })


@api.get("/embeddings/neighbors")
def embeddings_neighbors():
    word = request.args.get("word", "")
    k = int(request.args.get("k", 10))
    emb = get_embeddings()
    if not emb.vectors:
        _load_embeddings()
        emb = get_embeddings()
    return jsonify({"word": word, "neighbors": emb.nearest(word, k)})


def _load_embeddings():
    import os
    path = _embedding_path()
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        emb = get_embeddings()
        emb.vocab = data.get("vocab", [])
        emb.vectors = data.get("vectors", {})
        emb.dim = data.get("dim", 0)
    except (json.JSONDecodeError, OSError):
        pass


# ---------------------------------------------------------------------------
# 流水线配置与执行
# ---------------------------------------------------------------------------

@api.get("/pipeline/stages")
def pipeline_stages():
    return jsonify({"stages": _engine().list_stages()})


@api.post("/pipeline")
def save_pipeline():
    data = _payload()
    config = data.get("config") or data
    if not config.get("stages"):
        return jsonify({"error": "流水线至少需要一个阶段"}), 400
    name = config.get("name") or f"流水线_{int(time.time())}"
    record = {"name": name, "config": config, "created_at": time.time()}
    rid = _registry().task("pipeline_config").insert(record)
    return jsonify({"id": rid, "name": name, "ok": True})


@api.get("/pipeline")
def list_pipelines():
    records = _registry().task("pipeline_config").all()
    items = [{"id": r["id"], "name": r.get("name"), "config": r.get("config"),
              "created_at": r.get("created_at")}
             for r in records if not r.get("_deleted")]
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return jsonify({"pipelines": items})


@api.get("/pipeline/<pid>")
def get_pipeline(pid: str):
    record = _registry().task("pipeline_config").get(pid)
    if not record:
        return jsonify({"error": "流水线不存在"}), 404
    return jsonify(record)


@api.post("/pipeline/preview")
def pipeline_preview():
    """对单条文本跑流水线（不持久化），供配置页预览。"""
    data = _payload()
    text = (data.get("text") or "").strip()
    config = data.get("config")
    if not text or not config:
        return jsonify({"error": "缺少文本或配置"}), 400
    try:
        result = _engine().build(config).run({"text": text})
        return jsonify({"ok": True, "output": result})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


@api.post("/pipeline/<pid>/run")
def run_pipeline(pid: str):
    record = _registry().task("pipeline_config").get(pid)
    if not record:
        return jsonify({"error": "流水线不存在"}), 404
    config = record.get("config")
    data = _payload()

    run_id = uuid.uuid4().hex[:12]
    started = time.time()

    if data.get("batch"):
        # 批量：对语料库中的多篇文档执行
        corpus_ids = data.get("corpus_ids") or []
        store = _registry().task("corpus")
        docs = []
        if corpus_ids:
            docs = [store.get(c)["text"] for c in corpus_ids if store.get(c)]
        else:
            docs = [r["text"] for r in store.all() if not r.get("_deleted")]
        if not docs:
            return jsonify({"error": "没有可处理的文档"}), 400

        progress_state = {"done": 0, "total": len(docs)}

        def _progress(done, total):
            progress_state["done"] = done
            progress_state["total"] = total

        results = _engine().run_batch(
            config, docs, shared=data.get("shared"),
            max_workers=data.get("max_workers", 4),
            chunk_size=data.get("chunk_size", 16),
            progress=_progress)
        succeeded = sum(1 for r in results if r and r["ok"])
        failed = len(results) - succeeded
        run_record = {
            "run_id": run_id, "pipeline_id": pid, "batch": True,
            "doc_count": len(docs), "succeeded": succeeded, "failed": failed,
            "started": started, "finished": time.time(),
            "results": results,
        }
        rid = _registry().task("pipeline_run").insert(run_record)
        return jsonify({"run_id": run_id, "id": rid, "succeeded": succeeded,
                        "failed": failed, "doc_count": len(docs)})
    else:
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "缺少文本"}), 400
        try:
            output = _engine().build(config).run({"text": text})
            run_record = {
                "run_id": run_id, "pipeline_id": pid, "batch": False,
                "text": text, "output": output,
                "started": started, "finished": time.time(),
            }
            rid = _registry().task("pipeline_run").insert(run_record)
            return jsonify({"run_id": run_id, "id": rid, "ok": True,
                            "output": output})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc)}), 400


@api.get("/pipeline/run/<run_id>")
def get_pipeline_run(run_id: str):
    records = _registry().task("pipeline_run").query(
        where=[("run_id", "eq", run_id)])
    if not records:
        return jsonify({"error": "执行记录不存在"}), 404
    return jsonify(records[0])


# ---------------------------------------------------------------------------
# 结果查询（分片合并与查询）
# ---------------------------------------------------------------------------

@api.get("/results")
def list_result_tasks():
    registry = _registry()
    tasks = []
    for name in registry.tasks():
        if name in ("corpus", "pipeline_config", "annotation"):
            continue
        stats = registry.task(name).stats()
        tasks.append(stats)
    return jsonify({"tasks": tasks})


@api.get("/results/<task>")
def query_results(task: str):
    registry = _registry()
    if task not in registry.tasks():
        return jsonify({"error": "任务不存在"}), 404
    store = registry.task(task)
    where = []
    for key in ("type", "corpus_id"):
        val = request.args.get(key)
        if val:
            where.append((key, "eq", val))
    order_by = request.args.get("order_by")
    order = request.args.get("order", "desc")
    limit = request.args.get("limit", type=int)
    offset = request.args.get("offset", 0, type=int)
    records = store.query(where=where or None, order_by=order_by,
                          order=order, limit=limit, offset=offset)
    return jsonify({
        "task": task,
        "count": len(records),
        "stats": store.stats(),
        "records": records,
    })


@api.post("/results/<task>/compact")
def compact_results(task: str):
    registry = _registry()
    if task not in registry.tasks():
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(registry.task(task).compact())


@api.get("/results/<task>/merge")
def merge_results(task: str):
    registry = _registry()
    if task not in registry.tasks():
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(registry.task(task).merge())


@api.post("/results/compact_all")
def compact_all():
    return jsonify({"compacted": _registry().compact_all()})
