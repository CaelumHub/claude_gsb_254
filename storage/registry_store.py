"""跨文档实体注册表（Entity Registry / 实体对齐存储）。

消歧解决「这一句里它是谁」，本模块解决「把散落在多篇文档里的同一对象
对到同一个名下」，并保证持续入库时关系**稳得住、越积越不乱**。

设计要点
--------
1. **规范对象（canonical entity）**：每个真实世界对象一条，拥有永不复用的
   稳定 id（``E000001``）。画像（``aliases / industries / context_entities``）
   只做**并集累加**，不覆盖、不删除旧证据。
2. **提及（mention）**：每次文档入库产生的一次具体出现，落在普通分片存储
   （``entity_mention`` 任务）里，携带 ``entity_id`` 指向规范对象；
   判不了的标 ``status=PENDING``、``entity_id=None``，绝不乱挂。
3. **同名不同对象不混**：匹配时先过硬约束——语义类型冲突或行业互斥直接
   出局，再对名字、别名、共现实体、行业做加权评分，低于阈值不合并。
4. **稳定性**：id 一经分配不再改变；纠错（人工/自动合并）用 ``merge``
   把失败者并入胜者并留下 ``redirect`` 重定向，任何旧引用都能找到归宿；
   ``split`` 可把误并的对象拆开，且只影响明确指定的提及。
5. **知识库锚点**：``kb_id`` 相同的对象视为同一真实对象（苹果公司不会
   因为措辞不同而建两份）。

注册表本身是一份带版本号的 JSON（``registry.json``），所有读-改-写都在
文件锁 + 原子替换内完成，与分片存储共用同一套并发原语。
"""

from __future__ import annotations

import time
from typing import Any, Optional

from .lock import FileLock, lock_path_for
from .sharded import ShardedStore, _atomic_write_json, _read_json


# 提及状态
STATUS_LINKED = "LINKED"        # 已链接到知识库对象
STATUS_RESOLVED = "RESOLVED"    # 已对齐到注册表对象（知识库未收录）
STATUS_PENDING = "PENDING"      # 证据不足，暂不挂靠

# 匹配阈值
MATCH_MIN_SCORE = 4.0       # 总分低于该值 -> 不并入，新建/待定
MATCH_MARGIN = 2.0          # 多个候选时，冠亚军分差不足 -> 待定，避免误并

# 打分权重
W_EXACT_NAME = 6.0          # 名字就是规范名/主别名
W_ALIAS = 3.0               # 名字出现在别名集
W_INDUSTRY = 1.5            # 行业画像重合
W_CONTEXT = 2.0             # 共现实体画像重合
W_TYPE = 2.0                # 语义类型相同


class EntityRegistry:
    """规范实体注册表 + 提及对齐。"""

    def __init__(self, root: str, mention_store: ShardedStore,
                 filename: str = "registry.json"):
        self.dir = root if root.endswith("entity_registry") \
            else f"{root.rstrip('/')}/entity_registry"
        import os
        os.makedirs(self.dir, exist_ok=True)
        self.path = f"{self.dir}/{filename}"
        self.mentions = mention_store
        self._lock_path = lock_path_for(self.path)

    # ------------------------------------------------------------------
    # 底层读写
    # ------------------------------------------------------------------
    def _blank(self) -> dict:
        return {
            "version": 1,
            "next_seq": 1,
            "entities": {},        # entity_id -> 规范对象
            "redirect": {},        # 旧 entity_id -> 现行 entity_id
            "updated_at": time.time(),
        }

    def _read(self) -> dict:
        data = _read_json(self.path, None)
        if not data or "entities" not in data:
            return self._blank()
        data.setdefault("redirect", {})
        data.setdefault("next_seq", 1)
        return data

    def _write(self, reg: dict) -> None:
        reg["updated_at"] = time.time()
        _atomic_write_json(self.path, reg)

    def resolve_id(self, reg: dict, entity_id: str) -> str:
        """跟随 redirect 链找到现行 id。"""
        seen = set()
        cur = entity_id
        while cur in reg.get("redirect", {}) and cur not in seen:
            seen.add(cur)
            cur = reg["redirect"][cur]
        return cur

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get(self, entity_id: str) -> Optional[dict]:
        with FileLock(self._lock_path, mode="shared"):
            reg = self._read()
            eid = self.resolve_id(reg, entity_id)
            entity = reg["entities"].get(eid)
            if entity:
                entity = dict(entity)
                entity["id"] = eid
            return entity

    def all_entities(self) -> list[dict]:
        with FileLock(self._lock_path, mode="shared"):
            reg = self._read()
            out = []
            for eid, ent in reg["entities"].items():
                item = dict(ent)
                item["id"] = eid
                out.append(item)
        out.sort(key=lambda e: (-e.get("mention_count", 0), e["id"]))
        return out

    def stats(self) -> dict:
        with FileLock(self._lock_path, mode="shared"):
            reg = self._read()
            entities = reg["entities"]
            pending = sum(1 for e in entities.values()
                          if e.get("status") == STATUS_PENDING)
            return {
                "entity_count": len(entities),
                "pending_entity_count": pending,
                "mention_count": self.mentions.stats().get("total", 0),
                "kb_anchored": sum(1 for e in entities.values() if e.get("kb_id")),
            }

    def list_mentions(self, entity_id: Optional[str] = None,
                      status: Optional[str] = None,
                      limit: Optional[int] = None) -> list[dict]:
        where = []
        if entity_id:
            where.append(("entity_id", "eq", entity_id))
        if status:
            where.append(("status", "eq", status))
        return self.mentions.query(
            where=where or None, order_by="_created",
            order="desc", limit=limit)

    # ------------------------------------------------------------------
    # 单篇文档的提及入库与对齐
    # ------------------------------------------------------------------
    def ingest(self, linked: list[dict], doc_id: Optional[str] = None,
               text: str = "", source: str = "") -> dict:
        """把一篇文档的链接结果对齐入库。

        :param linked: :class:`nlp.linking.EntityLinker.link` 的返回
        :param doc_id: 文档标识（如语料 id），用于跨文档追溯
        :return: 统计 ``{resolved, pending, created, created_ids, mentions}``
        """
        stats = {"resolved": 0, "pending": 0, "created": 0,
                 "created_ids": [], "mentions": 0}
        if not linked:
            return stats

        mention_records: list[dict] = []
        with FileLock(self._lock_path):
            reg = self._read()
            for item in linked:
                stats["mentions"] += 1
                # 实际出现的字串（别名召回时可能是 Apple/红富士…）
                surface = item.get("surface_form") or item["text"]
                sem_type = item.get("sem_type")
                features = {
                    "industries": [item["industry"]] if item.get("industry") else [],
                    "context_entities": [
                        e for e in _context_from_evidence(item.get("evidence", []))
                    ],
                }

                entity_id, status, is_new = self._align(
                    reg, surface=surface, sem_type=sem_type,
                    kb_id=item.get("kb_id"), features=features,
                    link_status=item.get("status", ""))

                if entity_id is None:
                    stats["pending"] += 1
                else:
                    stats["resolved"] += 1
                    if is_new:
                        stats["created"] += 1
                        stats["created_ids"].append(entity_id)

                mention_records.append({
                    "text": text,
                    "doc_id": doc_id,
                    "source": source,
                    "surface": surface,
                    "entity_id": entity_id,
                    "kb_id": item.get("kb_id"),
                    "sem_type": sem_type,
                    "ner_type": item.get("ner_type"),
                    "industry": item.get("industry", ""),
                    "status": status,
                    "confidence": item.get("confidence", 0.0),
                    "start": item.get("start", -1),
                    "end": item.get("end", -1),
                    "evidence": item.get("evidence", []),
                    "_created": time.time(),
                })

            self._write(reg)

        if mention_records:
            self.mentions.insert_many(mention_records)
        return stats

    # -- 对齐决策（在锁内调用） ------------------------------------------
    def _align(self, reg: dict, *, surface: str, sem_type: str,
               kb_id: Optional[str], features: dict,
               link_status: str) -> tuple[Optional[str], str, bool]:
        """返回 (entity_id, mention_status, is_new)。

        entity_id 为 None 表示待定；is_new 表示本次是否新建了规范对象。
        """
        entities = reg["entities"]

        # 1) 知识库锚点：同一 kb_id 必然同一对象
        if kb_id:
            kb_name = _KB_NAMES.get(kb_id)
            target = self._find_by_kb(reg, kb_id)
            if target is None:
                target = self._new_id(reg)
                entities[target] = self._new_entity(
                    kb_name or surface, sem_type, kb_id=kb_id,
                    features=features, status=STATUS_LINKED)
                # 实际字串若与规范名不同，记为别名（苹果/红富士/Apple）
                if surface and surface != entities[target]["canonical"]:
                    self._absorb_alias(entities[target], surface)
                return target, STATUS_LINKED, True
            self._absorb(entities[target], surface, features)
            return target, STATUS_LINKED, False

        # 待定（知识库多候选证据不足，或粗归类也判不了）
        if link_status == "KB_PENDING" or sem_type in (None, "PENDING"):
            return None, STATUS_PENDING, False

        # 2) 在现有对象中找最佳匹配
        scored = self._score_candidates(
            reg, surface, sem_type, features)
        best = scored[0] if scored else None
        second = scored[1] if len(scored) > 1 else None
        margin = (best[1] - second[1]) if second else (best[1] if best else 0)

        if best and best[1] >= MATCH_MIN_SCORE and margin >= MATCH_MARGIN:
            eid = best[0]
            self._absorb(entities[eid], surface, features)
            return eid, STATUS_RESOLVED, False

        # 3) 没有任何候选 -> 可以放心新建；有近似候选但分不够 -> 待定，
        #    宁可挂起也不制造误并（后续证据充分时再由 resolve_pending 归位）
        if best is None:
            eid = self._new_id(reg)
            entities[eid] = self._new_entity(
                surface, sem_type, kb_id=None, features=features,
                status=STATUS_RESOLVED)
            return eid, STATUS_RESOLVED, True
        return None, STATUS_PENDING, False

    def _score_candidates(self, reg: dict, surface: str, sem_type: str,
                          features: dict) -> list[tuple[str, float, list[str]]]:
        scored: list[tuple[str, float, list[str]]] = []
        inds = set(features.get("industries", ()))
        ctx = set(features.get("context_entities", ()))
        for eid, ent in reg["entities"].items():
            # 已挂起的对象不参与吸附
            if ent.get("status") == STATUS_PENDING:
                continue
            # 硬约束：语义类型冲突直接出局（同名不同类绝不混）
            if not _types_compatible(ent.get("sem_type"), sem_type):
                continue
            # 硬约束：行业互斥出局（两边行业都明确且不相容）。
            # 注意「同名 + 同类型」是更强的同一性证据，不因单次行业票
            # 不同就拒绝——行业判错比同名同类撞车常见得多，故这里
            # 只对「连名字都不同」的候选施加行业硬约束。
            ent_inds = set(ent.get("industries", ()))
            name_hit = (surface == ent.get("canonical")
                        or surface in set(ent.get("aliases", ())))
            if inds and ent_inds and not name_hit and \
                    not _industries_compatible(inds, ent_inds):
                continue

            score = 0.0
            why: list[str] = []
            names = {ent.get("canonical", "")} | set(ent.get("aliases", ()))
            if surface == ent.get("canonical"):
                score += W_EXACT_NAME
                why.append("同名")
            elif surface in names:
                score += W_ALIAS
                why.append("别名")

            if ent.get("sem_type") == sem_type:
                score += W_TYPE
                why.append("同类")

            overlap_ind = inds & ent_inds
            if not overlap_ind and inds and ent_inds and \
                    _industries_compatible(inds, ent_inds):
                # 细类不同但同属一个大领域，给一半行业分
                score += W_INDUSTRY * 0.5
                why.append("行业相近")
            elif overlap_ind:
                score += W_INDUSTRY
                why.append("行业:" + "/".join(sorted(overlap_ind)))

            overlap_ctx = ctx & set(ent.get("context_entities", ()))
            if overlap_ctx:
                score += W_CONTEXT * min(len(overlap_ctx), 2)
                why.append("共现:" + "/".join(sorted(overlap_ctx)))

            if score > 0:
                scored.append((eid, round(score, 2), why))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # 待定提及的再处理 / 人工干预
    # ------------------------------------------------------------------
    def resolve_pending(self, limit: int = 500) -> dict:
        """对所有 PENDING 提及用当前画像重新对齐一次。

        新文档不断入库会带来新证据；重判只改变提及的挂靠关系，
        不删除任何历史，已确定的对象 id 不受影响。
        """
        pending = self.list_mentions(status=STATUS_PENDING, limit=100000)
        created = resolved = still = 0
        with FileLock(self._lock_path):
            reg = self._read()
            updates: list[tuple[str, Optional[str], str]] = []
            new_ids: set[str] = set()
            for m in pending[:limit]:
                features = {
                    "industries": [m["industry"]] if m.get("industry") else [],
                    "context_entities": _context_from_evidence(
                        m.get("evidence", [])),
                }
                before = set(reg["entities"])
                eid, status, _ = self._align(
                    reg, surface=m["surface"], sem_type=m.get("sem_type"),
                    kb_id=m.get("kb_id"), features=features,
                    link_status="")
                if eid and eid not in before:
                    new_ids.add(eid)
                updates.append((m["id"], eid, status))
                if eid is None:
                    still += 1
                elif eid in new_ids:
                    created += 1
                else:
                    resolved += 1
            self._write(reg)

        # 提及记录在分片存储里，逐条改写（量小、低频运维操作）
        for mid, eid, status in updates:
            self._update_mention(mid, {"entity_id": eid, "status": status})
        return {"checked": len(updates), "resolved": resolved,
                "created": created, "still_pending": still}

    def assign(self, mention_id: str, entity_id: Optional[str] = None,
               create_name: Optional[str] = None,
               sem_type: Optional[str] = None) -> dict:
        """人工把一条提及指定给某对象；不给对象则按名字新建。"""
        mention = self.mentions.get(mention_id)
        if not mention:
            raise KeyError(f"提及不存在: {mention_id}")

        with FileLock(self._lock_path):
            reg = self._read()
            if entity_id is None:
                entity_id = self._new_id(reg)
                reg["entities"][entity_id] = self._new_entity(
                    create_name or mention["surface"],
                    sem_type or mention.get("sem_type", "OTHER"),
                    kb_id=mention.get("kb_id"),
                    features={"industries": [mention["industry"]]
                              if mention.get("industry") else [],
                              "context_entities": _context_from_evidence(
                                  mention.get("evidence", []))},
                    status=STATUS_RESOLVED)
                created = True
            else:
                entity_id = self.resolve_id(reg, entity_id)
                if entity_id not in reg["entities"]:
                    raise KeyError(f"对象不存在: {entity_id}")
                self._absorb(reg["entities"][entity_id], mention["surface"],
                             {"industries": [mention["industry"]]
                              if mention.get("industry") else [],
                              "context_entities": _context_from_evidence(
                                  mention.get("evidence", []))})
                created = False
            self._write(reg)

        old_eid = mention.get("entity_id")
        self._update_mention(mention_id, {
            "entity_id": entity_id, "status": STATUS_RESOLVED})
        self._adjust_count(old_eid, entity_id)
        return {"entity_id": entity_id, "created": created}

    def merge(self, source_id: str, target_id: str) -> dict:
        """把 ``source`` 并入 ``target``（纠错/证据充分后的聚合）。

        - 画像做并集合并；
        - source 的提及全部改挂 target；
        - source 不删除，写入 redirect，旧引用永远可解析。
        """
        with FileLock(self._lock_path):
            reg = self._read()
            source_id = self.resolve_id(reg, source_id)
            target_id = self.resolve_id(reg, target_id)
            if source_id == target_id:
                return {"source": source_id, "target": target_id,
                        "moved_mentions": 0}
            if source_id not in reg["entities"] or \
                    target_id not in reg["entities"]:
                raise KeyError("待合并对象不存在")

            src, tgt = reg["entities"][source_id], reg["entities"][target_id]
            tgt["aliases"] = sorted(
                set(tgt.get("aliases", [])) | {src.get("canonical", "")}
                | set(src.get("aliases", [])))
            tgt["industries"] = sorted(
                set(tgt.get("industries", [])) | set(src.get("industries", [])))
            tgt["context_entities"] = sorted(
                set(tgt.get("context_entities", []))
                | set(src.get("context_entities", [])))
            tgt["mention_count"] = tgt.get("mention_count", 0) + \
                src.get("mention_count", 0)
            if not tgt.get("kb_id") and src.get("kb_id"):
                tgt["kb_id"] = src["kb_id"]

            moved = 0
            for m in self.mentions.query(
                    where=[("entity_id", "eq", source_id)]):
                self._update_mention(m["id"], {"entity_id": target_id})
                moved += 1

            reg["entities"].pop(source_id, None)
            reg["redirect"][source_id] = target_id
            self._write(reg)
        return {"source": source_id, "target": target_id,
                "moved_mentions": moved}

    def split(self, entity_id: str, mention_ids: list[str],
              new_name: Optional[str] = None) -> dict:
        """把若干提及从 ``entity_id`` 拆出为一个新对象（纠正误并）。"""
        with FileLock(self._lock_path):
            reg = self._read()
            entity_id = self.resolve_id(reg, entity_id)
            if entity_id not in reg["entities"]:
                raise KeyError("对象不存在")
            src = reg["entities"][entity_id]

            new_id = self._new_id(reg)
            sample = self.mentions.get(mention_ids[0]) if mention_ids else None
            name = new_name or (sample["surface"] if sample else src["canonical"])
            industries, ctx = set(), set()
            moved = 0
            for mid in mention_ids:
                m = self.mentions.get(mid)
                if not m or m.get("entity_id") != entity_id:
                    continue
                self._update_mention(mid, {
                    "entity_id": new_id, "status": STATUS_RESOLVED})
                if m.get("industry"):
                    industries.add(m["industry"])
                ctx |= set(_context_from_evidence(m.get("evidence", [])))
                moved += 1

            if moved == 0:
                raise ValueError("没有可拆分的提及")
            reg["entities"][new_id] = self._new_entity(
                name, sample.get("sem_type", "OTHER") if sample else "OTHER",
                kb_id=None,
                features={"industries": sorted(industries),
                          "context_entities": sorted(ctx)},
                status=STATUS_RESOLVED)
            reg["entities"][new_id]["mention_count"] = moved
            src["mention_count"] = max(
                0, src.get("mention_count", 0) - moved)
            self._write(reg)
        return {"new_id": new_id, "moved_mentions": moved}

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _new_id(self, reg: dict) -> str:
        seq = reg["next_seq"]
        reg["next_seq"] = seq + 1
        return f"E{seq:06d}"

    @staticmethod
    def _find_by_kb(reg: dict, kb_id: str) -> Optional[str]:
        for eid, ent in reg["entities"].items():
            if ent.get("kb_id") == kb_id:
                return eid
        return None

    @staticmethod
    def _new_entity(canonical: str, sem_type: str, *, kb_id: Optional[str],
                    features: dict, status: str) -> dict:
        return {
            "canonical": canonical,
            "sem_type": sem_type,
            "kb_id": kb_id,
            "status": status,
            "aliases": [],
            "industries": sorted(set(features.get("industries", []))),
            "context_entities": sorted(
                set(features.get("context_entities", []))),
            "mention_count": 1,
            "first_seen": time.time(),
        }

    @staticmethod
    def _absorb(entity: dict, surface: str, features: dict) -> None:
        """把一次新提及的证据并入画像（只加不减）。"""
        if surface and surface != entity.get("canonical"):
            aliases = set(entity.get("aliases", []))
            aliases.add(surface)
            entity["aliases"] = sorted(aliases)
        entity["industries"] = sorted(
            set(entity.get("industries", []))
            | set(features.get("industries", [])))
        entity["context_entities"] = sorted(
            set(entity.get("context_entities", []))
            | set(features.get("context_entities", [])))
        entity["mention_count"] = entity.get("mention_count", 0) + 1

    @staticmethod
    def _absorb_alias(entity: dict, surface: str) -> None:
        """仅并入别名（规范名来自知识库，不因首次出现字串改变）。"""
        if surface and surface != entity.get("canonical"):
            aliases = set(entity.get("aliases", []))
            aliases.add(surface)
            entity["aliases"] = sorted(aliases)

    def _adjust_count(self, old_id: Optional[str], new_id: str) -> None:
        with FileLock(self._lock_path):
            reg = self._read()
            if old_id:
                old_real = self.resolve_id(reg, old_id)
                if old_real in reg["entities"] and old_real != new_id:
                    reg["entities"][old_real]["mention_count"] = max(
                        0, reg["entities"][old_real].get("mention_count", 1) - 1)
            if new_id in reg["entities"]:
                reg["entities"][new_id]["mention_count"] = \
                    reg["entities"][new_id].get("mention_count", 0) + 1
            self._write(reg)

    def _update_mention(self, mention_id: str, patch: dict) -> None:
        """在单个排他临界区内扫描分片、替换一条提及并原子写回。"""
        with FileLock(lock_path_for(self.mentions.meta_path)):
            meta = self.mentions._read_meta()
            for index in range(meta.get("shard_count", 0)):
                path = self.mentions._shard_path(index)
                with FileLock(lock_path_for(path)):
                    records = self.mentions._read_shard(index)
                    for i, rec in enumerate(records):
                        if rec.get("id") == mention_id and \
                                not rec.get("_deleted"):
                            records[i].update(patch)
                            self.mentions._write_shard(index, records)
                            return
        raise KeyError(f"提及不存在: {mention_id}")


def _types_compatible(type_a: str, type_b: str) -> bool:
    """语义类型是否允许合并。

    完全相同当然可以；ORG/COMPANY/GOV/SCHOOL 同属机构大类，放宽为可匹配
    （后缀归类粒度不同），其余跨大类（食物 vs 公司、人名 vs 地名）一律拒绝。
    """
    if type_a == type_b:
        return True
    org_family = {"ORG", "COMPANY", "GOV", "SCHOOL"}
    loc_family = {"LOCATION", "NATURAL"}
    if type_a in org_family and type_b in org_family:
        return True
    if type_a in loc_family and type_b in loc_family:
        return True
    return False


def _context_from_evidence(evidence: list[str]) -> list[str]:
    """从证据串里还原共现实体（``共现:雷军/华为``）。"""
    out: list[str] = []
    for item in evidence:
        if item.startswith("共现:"):
            out.extend(item[len("共现:"):].split("/"))
    return out


def _industries_compatible(a: set[str], b: set[str]) -> bool:
    """行业相容性：惰性引用 linking 层的相容组，失败则退化为求交集。"""
    try:
        from nlp.linking import industries_compatible
        return industries_compatible(a, b)
    except Exception:  # noqa: BLE001
        return bool(a & b)


def _kb_name_map() -> dict[str, str]:
    """惰性构建 kb_id -> 知识库规范名（避免存储层强依赖算法层）。"""
    try:
        from nlp.linking import ENTITY_KB
    except Exception:  # noqa: BLE001
        return {}
    mapping: dict[str, str] = {}
    for candidates in ENTITY_KB.values():
        for cand in candidates:
            mapping[cand["kb_id"]] = cand["name"]
    return mapping


_KB_NAMES = _kb_name_map()
