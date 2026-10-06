"""实体消歧、实体链接与跨文档实体对齐。

在 NER 之后回答三个问题：

1. **消歧**：一个实体提及（mention）在当前语境下到底属于哪类、指向
   哪个真实对象——「苹果」是水果还是公司，「小米」是粮食还是品牌；
2. **链接**：把它挂到知识库中唯一对象的名下（entity_id）；
3. **跨文档对齐**：语料分片存放，同一对象散落在多篇文档里，要聚到
   同一实体名下；碰巧同名的不同对象（两个「苹果」）必须分开，不能
   混成一个。

判断依据（不只看字面）：
- **搭配词**：提及所在句子的共现词（吃 / 斤 / 果园 → 水果；
  发布 / 股价 / 财报 → 公司）；
- **行业词**：句子的领域分布（科技 / 农业 / 食品 …）与候选实体的
  领域是否一致，冲突时扣分；
- **上下文实体**：同句出现的其它实体是否在候选实体的关联表中
  （「苹果」与「华为」同框 → 公司）。

决策策略（拿不准就待定，不乱猜）：
- 最高分 >= accept_score 且与次高分拉开 margin → 链接；
- 知识库里查不到这个表面形式（unknown）→ 进入**隔离观察**：
  首次出现只记录语境证据、标待定；同一表面形式再次出现且累计
  证据足够时才新建动态实体——避免把 NER 噪声（如被误判为人名的
  普通词）一次就固化进实体库；
- 候选存在但分差不足 → 待定（pending）；待定提及在知识库扩充后
  可经 ``reresolve`` 重新评估——只升级，不改判已链接的结果。

增量稳定性（新文档不断入库也不越积越乱）：
- 已链接的提及永不自动改判，只有人工合并能改动；
- 实体画像（profile）只增不减、按频次封顶截断，避免单篇文档带偏；
- 实体合并是显式操作，通过 ``merged_into`` 保留重定向链。
"""

from __future__ import annotations

import math
import threading
import uuid
from typing import Optional

from .lexicon import STOPWORDS
from .ner import NERExtractor
from .segmenter import Segmenter


# ---------------------------------------------------------------------------
# 实体语义类别与领域词表
# ---------------------------------------------------------------------------

SENSE_TYPE_NAMES = {
    "COMPANY": "公司", "ORG": "机构", "BRAND": "品牌", "PRODUCT": "产品",
    "PERSON": "人物", "PLACE": "地点", "FRUIT": "水果", "GRAIN": "粮食",
    "UNKNOWN": "未知",
}

# 行业词表：领域 -> 指示词集合（用于句/文档的领域判定）
DOMAIN_LEXICON = {
    "科技": {"手机", "电脑", "芯片", "软件", "系统", "互联网", "发布会",
             "发布", "智能", "算法", "研发", "供应链", "生态", "用户"},
    "财经": {"股价", "市值", "财报", "融资", "上市", "股票", "营收",
             "利润", "大涨", "下跌"},
    "农业": {"种植", "果园", "农田", "亩产", "农户", "丰收", "庄稼",
             "大棚", "施肥", "果农", "产量", "采摘"},
    "食品": {"吃", "煮", "粥", "甜", "酸", "斤", "超市", "营养",
             "口感", "水果", "批发", "新鲜", "菜品"},
    "体育": {"比赛", "赛季", "球队", "冠军", "篮球", "足球", "联赛"},
    "自然": {"河流", "雨林", "流域", "森林", "气候", "生态", "位于"},
}

# NER 类型 -> 相容的实体语义类别（类型先验）
NER_SENSE_COMPAT = {
    "PERSON": {"PERSON"},
    "ORGANIZATION": {"COMPANY", "ORG", "BRAND"},
    "LOCATION": {"PLACE"},
}
LINKABLE_NER_TYPES = set(NER_SENSE_COMPAT)
# 动态实体的默认语义类别（无更细证据时）
NER_TO_SENSE = {"PERSON": "PERSON", "ORGANIZATION": "ORG", "LOCATION": "PLACE"}

# 决策阈值（可在 resolve_document 调用时覆盖）
DEFAULT_ACCEPT_SCORE = 3.5    # 最高分达到该值才可链接
DEFAULT_ACCEPT_MARGIN = 1.5   # 与次高分的最小差距
DEFAULT_REJECT_SCORE = 1.5    # 全部低于该值才考虑新建实体
MIN_INFORMATIVE_WORDS = 2     # 新建实体所需的最少有效语境词
MIN_SIGHTINGS = 2             # 新建实体所需的最少出现次数（隔离观察）
EVIDENCE_CAP = 5000           # 隔离证据表上限（超出按先入先出淘汰）
PROFILE_CAP = 120             # 实体画像词表上限（按频次截断）


# ---------------------------------------------------------------------------
# 内置实体知识库（种子）：覆盖几组经典歧义
# ---------------------------------------------------------------------------

def _seed(entity_id, canonical, aliases, sense_type, domains, cues, related):
    return {
        "entity_id": entity_id, "canonical": canonical, "aliases": aliases,
        "sense_type": sense_type, "domains": domains, "cues": cues,
        "related": related, "profile": {}, "mention_count": 0,
        "source": "builtin", "merged_into": None,
    }


BUILTIN_KB = [
    _seed("kb_apple_inc", "苹果公司", ["苹果", "Apple", "苹果电脑"], "COMPANY",
          ["科技", "财经"],
          {"手机": 3, "电脑": 3, "发布": 3, "发布会": 3, "股价": 3, "市值": 3,
           "财报": 3, "芯片": 3, "系统": 2, "供应链": 2, "库克": 3},
          ["华为", "小米", "微软", "谷歌", "乔布斯", "库克"]),
    _seed("kb_apple_fruit", "苹果(水果)", ["苹果"], "FRUIT",
          ["农业", "食品"],
          {"吃": 3, "斤": 3, "果园": 3, "种植": 3, "甜": 3, "水果": 3,
           "香蕉": 2, "葡萄": 2, "丰收": 2, "果农": 3, "超市": 2, "采摘": 2},
          ["香蕉", "葡萄", "梨"]),
    _seed("kb_xiaomi_inc", "小米科技", ["小米", "小米公司"], "COMPANY",
          ["科技", "财经"],
          {"手机": 3, "发布": 3, "发布会": 3, "雷军": 3, "生态链": 3,
           "智能家居": 2, "股价": 2, "财报": 2},
          ["华为", "苹果", "雷军"]),
    _seed("kb_xiaomi_grain", "小米(粮食)", ["小米", "谷子"], "GRAIN",
          ["农业", "食品"],
          {"粥": 3, "熬": 3, "种植": 3, "谷子": 3, "亩产": 3, "农户": 2,
           "红枣": 2, "营养": 2, "粗粮": 2},
          ["红枣", "大米", "谷子"]),
    _seed("kb_amazon_inc", "亚马逊(公司)", ["亚马逊", "Amazon"], "COMPANY",
          ["科技", "财经"],
          {"电商": 3, "云": 2, "股价": 3, "财报": 3, "贝索斯": 3, "市值": 2},
          ["苹果", "微软", "谷歌", "贝索斯"]),
    _seed("kb_amazon_river", "亚马逊(河流)", ["亚马逊", "亚马逊河"], "PLACE",
          ["自然"],
          {"河流": 3, "雨林": 3, "流域": 3, "位于": 2, "森林": 2},
          ["雨林", "巴西"]),
    _seed("kb_tesla_inc", "特斯拉(公司)", ["特斯拉", "Tesla"], "COMPANY",
          ["科技", "财经"],
          {"电动车": 3, "汽车": 3, "马斯克": 3, "股价": 3, "交付": 3,
           "工厂": 2},
          ["马斯克", "比亚迪"]),
    _seed("kb_tesla_person", "特斯拉(科学家)", ["特斯拉"], "PERSON", [],
          {"发明": 3, "交流电": 3, "科学家": 3, "物理学家": 2},
          ["爱迪生"]),
    # 无歧义实体（同时充当上下文实体的关联目标）
    _seed("kb_huawei", "华为", ["华为", "华为公司"], "COMPANY", ["科技"],
          {"手机": 3, "芯片": 3, "5G": 3, "任正非": 3, "发布": 2},
          ["小米", "苹果", "任正非"]),
    _seed("kb_tencent", "腾讯", ["腾讯", "腾讯公司"], "COMPANY", ["科技"],
          {"微信": 3, "游戏": 3, "股价": 2}, ["阿里巴巴", "百度"]),
    _seed("kb_alibaba", "阿里巴巴", ["阿里巴巴", "阿里"], "COMPANY",
          ["科技", "财经"],
          {"电商": 3, "马云": 3, "股价": 2}, ["腾讯", "马云"]),
    _seed("kb_leijun", "雷军", ["雷军"], "PERSON", ["科技"],
          {"小米": 3, "创始人": 3, "董事长": 2}, ["小米"]),
    _seed("kb_mayun", "马云", ["马云"], "PERSON", ["科技"],
          {"阿里巴巴": 3, "创始人": 3}, ["阿里巴巴"]),
    _seed("kb_banana", "香蕉", ["香蕉"], "FRUIT", ["食品", "农业"],
          {"吃": 3, "水果": 3, "种植": 2}, ["苹果", "葡萄"]),
]

# 机构别名回退后缀：「小米公司」未命中时尝试「小米」
_ALIAS_FALLBACK_SUFFIXES = ("公司", "集团", "大学", "医院", "银行")

_SENT_ENDINGS = "。！？!?；;\n"
_PUNCT_STRIP = "，。、；：""''（）《》【】,.!?;:()\" "


def _norm_entry(entry: dict) -> dict:
    """补齐实体记录缺省字段（持久化记录与内存对象共用同一结构）。"""
    e = dict(entry)
    e.setdefault("canonical", e.get("entity_id", ""))
    e.setdefault("aliases", [])
    e.setdefault("sense_type", "UNKNOWN")
    e.setdefault("domains", [])
    e.setdefault("cues", {})
    e.setdefault("related", [])
    e.setdefault("profile", {})
    e.setdefault("mention_count", 0)
    e.setdefault("source", "dynamic")
    e.setdefault("merged_into", None)
    return e


# ---------------------------------------------------------------------------
# 语境特征抽取
# ---------------------------------------------------------------------------

def extract_context(text: str, mention: dict, all_mentions: list,
                    segmenter: Segmenter) -> dict:
    """抽取提及的语境特征：句子搭配词、领域分布、同句共现实体。"""
    start, end = mention["start"], mention["end"]

    # 所在句子边界
    sent_start = 0
    for p in _SENT_ENDINGS:
        idx = text.rfind(p, 0, start)
        if idx >= 0:
            sent_start = max(sent_start, idx + 1)
    sent_end = len(text)
    for p in _SENT_ENDINGS:
        idx = text.find(p, end)
        if idx >= 0:
            sent_end = min(sent_end, idx + 1)
    sentence = text[sent_start:sent_end]

    # 分词并记录偏移：落在提及区间内的 token（如「星辰公司」里的「公司」）
    # 属于提及本身而非语境，必须剔除，不能只按字面剔除完整词
    surface = mention["text"]
    span = (start - sent_start, end - sent_start)
    words = []
    pos = 0
    for w in segmenter.cut(sentence):
        idx = sentence.find(w, pos)
        if idx < 0:
            idx = pos
        tok_start, tok_end = idx, idx + len(w)
        pos = tok_end
        if tok_start < span[1] and span[0] < tok_end:
            continue  # 与提及区间重叠
        w = w.strip(_PUNCT_STRIP)
        if not w or w in STOPWORDS or w == surface:
            continue
        if not any("一" <= c <= "鿿" or c.isalnum() for c in w):
            continue
        words.append(w)

    # 领域分布：句子为主，句子没有行业词时退回整篇文档
    domains = _detect_domains(words)
    if not domains:
        doc_words = [w.strip(_PUNCT_STRIP) for w in segmenter.cut(text)]
        domains = _detect_domains(
            [w for w in doc_words if w and w not in STOPWORDS])

    co_entities = [
        m["text"] for m in all_mentions
        if m is not mention and sent_start <= m["start"] and m["end"] <= sent_end
    ]
    return {"words": words, "domains": domains, "co_entities": co_entities}


def _detect_domains(words: list) -> list:
    word_set = set(words)
    hits = []
    for domain, lex in DOMAIN_LEXICON.items():
        n = len(word_set & lex)
        if n:
            hits.append((domain, n))
    hits.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in hits]


# ---------------------------------------------------------------------------
# 候选打分
# ---------------------------------------------------------------------------

def score_candidate(context: dict, entry: dict, surface: str,
                    ner_type: str) -> tuple[float, dict]:
    """对一个候选实体打分，返回 (总分, 明细)。分数可解释、可复算。"""
    words = set(context.get("words", []))

    # 1. 搭配词：种子特征词（权重高）+ 画像中学到的词（权重随频次）
    cue_score, matched = 0.0, []
    for w in words:
        if w in entry["cues"]:
            cue_score += entry["cues"][w]
            matched.append(w)
        elif w in entry["profile"]:
            cue_score += min(2.0, 0.5 * entry["profile"][w])
            matched.append(w)
    cue_score = min(cue_score, 6.0)

    # 2. 行业词：领域一致加分，明确冲突扣分。
    #    冲突扣分只适用于内置实体（领域是人工维护的，可靠）；
    #    动态实体的领域来自单篇文档的推断，覆盖不全，缺席不构成反证。
    ctx_domains = context.get("domains", [])
    domain_hits = [d for d in ctx_domains if d in entry["domains"]]
    domain_score = min(3.0, float(len(domain_hits)))
    mismatch = (entry["source"] == "builtin"
                and bool(ctx_domains) and bool(entry["domains"])
                and not domain_hits)
    mismatch_penalty = -1.0 if mismatch else 0.0

    # 3. 上下文实体：同句实体命中关联表
    co_hits = [c for c in context.get("co_entities", []) if c in entry["related"]]
    co_score = min(4.0, 2.0 * len(co_hits))

    # 4. NER 类型先验（弱证据：NER 词典本身可能带偏）
    compat = NER_SENSE_COMPAT.get(ner_type, set())
    if entry["sense_type"] in compat:
        type_prior = 1.0
    elif compat and entry["sense_type"] != "UNKNOWN":
        type_prior = -0.5
    else:
        type_prior = 0.0

    # 5. 名称连续性：动态实体允许更低门槛的「同名即同对象」，
    #    但有竞争候选或领域冲突时会被 margin / 扣分抵消
    name_bonus = 0.0
    if surface in entry["aliases"]:
        name_bonus += 1.0
    if surface == entry["canonical"]:
        name_bonus += 0.5
    if entry["source"] == "dynamic" and name_bonus:
        name_bonus += 1.5

    total = (cue_score + domain_score + co_score + type_prior
             + name_bonus + mismatch_penalty)
    detail = {
        "cue": round(cue_score, 2), "cue_words": matched,
        "domain": domain_score, "domain_hits": domain_hits,
        "domain_mismatch": bool(mismatch),
        "co_entity": co_score, "co_hits": co_hits,
        "type_prior": type_prior, "name": name_bonus,
    }
    return round(total, 3), detail


# ---------------------------------------------------------------------------
# 实体对齐器
# ---------------------------------------------------------------------------

class EntityResolver:
    """实体消歧 + 链接 + 跨文档对齐（纯内存计算，持久化由上层负责）。

    线程安全；所有改变状态的接口同时返回「变更集」，上层（Web 层）
    据此把新实体 / 画像更新写入分片存储，重启后可从存储重建。
    """

    def __init__(self, entries: Optional[list] = None,
                 segmenter: Optional[Segmenter] = None,
                 ner: Optional[NERExtractor] = None):
        self._lock = threading.RLock()
        self.segmenter = segmenter or Segmenter()
        self.ner = ner or NERExtractor(self.segmenter)
        self.entries: dict[str, dict] = {}
        self.alias_index: dict[str, set] = {}
        # 隔离观察：未知识别的表面形式 -> {证据键: 语境}
        # 证据键 = (doc_id, start, 语境指纹)，同一提及重复评估不会重复计数
        self._surface_evidence: dict[str, dict] = {}
        for e in (entries if entries is not None else BUILTIN_KB):
            self._register(_norm_entry(e))

    # -- 状态管理 ---------------------------------------------------------
    def _register(self, entry: dict) -> None:
        self.entries[entry["entity_id"]] = entry
        for alias in {entry["canonical"], *entry["aliases"]}:
            self.alias_index.setdefault(alias, set()).add(entry["entity_id"])

    def load_entries(self, entries: list) -> None:
        """用外部（持久层）的实体列表整体替换当前状态。"""
        with self._lock:
            self.entries = {}
            self.alias_index = {}
            self._surface_evidence = {}
            for e in entries:
                self._register(_norm_entry(e))

    def get_entry(self, entity_id: str) -> Optional[dict]:
        with self._lock:
            entry = self.entries.get(entity_id)
            if entry and entry.get("merged_into"):
                # 沿重定向链找到最终实体
                seen = set()
                while entry and entry.get("merged_into"):
                    if entry["entity_id"] in seen:
                        break
                    seen.add(entry["entity_id"])
                    entry = self.entries.get(entry["merged_into"])
            return entry

    def _lookup(self, surface: str) -> list:
        ids = set(self.alias_index.get(surface, ()))
        if not ids and len(surface) > 2:
            for suf in _ALIAS_FALLBACK_SUFFIXES:
                if surface.endswith(suf):
                    ids |= self.alias_index.get(surface[:-len(suf)], set())
        return [self.entries[i] for i in ids
                if i in self.entries and not self.entries[i].get("merged_into")]

    # -- 决策 -------------------------------------------------------------
    def _decide(self, context: dict, surface: str, ner_type: str,
                accept_score: float, accept_margin: float,
                reject_score: float) -> dict:
        candidates = self._lookup(surface)
        scored = []
        for entry in candidates:
            s, detail = score_candidate(context, entry, surface, ner_type)
            scored.append((s, entry, detail))
        scored.sort(key=lambda x: x[0], reverse=True)

        top = scored[0] if scored else None
        second = scored[1][0] if len(scored) > 1 else 0.0
        margin = round(top[0] - second, 3) if top else 0.0
        candidate_view = [
            {"entity_id": e["entity_id"], "canonical": e["canonical"],
             "sense_type": e["sense_type"], "score": s, "detail": d}
            for s, e, d in scored[:3]
        ]

        if top and top[0] >= accept_score and (
                len(scored) == 1 or margin >= accept_margin):
            return {"status": "linked", "entry": top[1], "score": top[0],
                    "margin": margin, "candidates": candidate_view,
                    "detail": top[2]}

        if (not top or top[0] < reject_score) \
                and ner_type in LINKABLE_NER_TYPES:
            # 知识库查无此名：可能是新对象，也可能是 NER 噪声 → 隔离观察
            return {"status": "unknown", "candidates": candidate_view,
                    "score": top[0] if top else 0.0, "margin": margin}

        return {"status": "pending", "candidates": candidate_view,
                "score": top[0] if top else 0.0, "margin": margin}

    @staticmethod
    def _confidence(score: float, accept_score: float) -> float:
        return round(1.0 / (1.0 + math.exp(-(score - accept_score))), 3)

    # -- 主入口 -----------------------------------------------------------
    def resolve_document(self, text: str, doc_id=None,
                         mentions: Optional[list] = None,
                         accept_score: float = DEFAULT_ACCEPT_SCORE,
                         accept_margin: float = DEFAULT_ACCEPT_MARGIN,
                         reject_score: float = DEFAULT_REJECT_SCORE) -> dict:
        """处理一篇文档，返回提及的链接结果与实体变更集。

        新文档入库即调用本方法：已链接结果稳定不变，只有本次新产生的
        提及会得到决策；实体画像随之增量更新。
        """
        if mentions is None:
            mentions = self.ner.recognize(text)

        results = []
        mutations: dict = {"new_entities": [], "updated_entities": {}}
        with self._lock:
            for m in mentions:
                if m.get("type") not in LINKABLE_NER_TYPES:
                    continue
                if len(m.get("text", "")) < 2:
                    continue
                context = extract_context(text, m, mentions, self.segmenter)
                decision = self._decide(context, m["text"], m["type"],
                                        accept_score, accept_margin,
                                        reject_score)
                rec = {
                    "text": m["text"], "start": m["start"], "end": m["end"],
                    "ner_type": m["type"], "doc_id": doc_id,
                    "status": "pending", "entity_id": None,
                    "canonical": None, "sense_type": None,
                    "score": decision["score"], "margin": decision["margin"],
                    "confidence": 0.0,
                    "candidates": decision["candidates"],
                    "context": context["words"], "domains": context["domains"],
                    "co_entities": context["co_entities"],
                }

                if decision["status"] == "unknown":
                    entry = self._handle_unknown(
                        m["text"], m["type"], context,
                        evidence_key=(doc_id, m["start"],
                                      tuple(context["words"])))
                    if entry is not None:
                        mutations["new_entities"].append(dict(entry))
                        decision = {"status": "linked", "entry": entry,
                                    "score": decision["score"],
                                    "margin": decision["margin"],
                                    "candidates": decision["candidates"],
                                    "detail": {}}
                    else:
                        decision = {"status": "pending",
                                    "score": decision["score"],
                                    "margin": decision["margin"],
                                    "candidates": decision["candidates"]}

                if decision["status"] == "linked":
                    entry = decision["entry"]
                    self._update_profile(entry, context)
                    mutations["updated_entities"][entry["entity_id"]] = dict(entry)
                    rec.update({
                        "status": "linked",
                        "entity_id": entry["entity_id"],
                        "canonical": entry["canonical"],
                        "sense_type": entry["sense_type"],
                        "confidence": self._confidence(decision["score"],
                                                       accept_score),
                    })
                results.append(rec)
        return {"mentions": results, "mutations": mutations}

    # -- 动态实体 ---------------------------------------------------------
    def _handle_unknown(self, surface: str, ner_type: str,
                        context: dict, evidence_key=None) -> Optional[dict]:
        """隔离观察：记录未知识别的表面形式，证据足够才新建实体。

        同一表面形式累计出现 >= MIN_SIGHTINGS 次（按证据键去重，同一
        提及的重复评估只算一次）、且累计有效语境词足够时，才把它立为
        动态实体（返回该实体）；否则返回 None（本次提及保持待定）。
        这样单次出现的 NER 噪声永远不会固化进实体库。
        """
        ev = self._surface_evidence.setdefault(surface, {})
        if evidence_key is None:
            evidence_key = ("adhoc", len(ev))
        ev[evidence_key] = {"ner_type": ner_type, "context": context}
        while len(self._surface_evidence) > EVIDENCE_CAP:
            self._surface_evidence.pop(next(iter(self._surface_evidence)))

        if len(ev) < MIN_SIGHTINGS:
            return None
        words, domains, co = [], [], []
        for e in ev.values():
            words += e["context"].get("words", [])
            domains += e["context"].get("domains", [])
            co += e["context"].get("co_entities", [])
        merged = {"words": list(dict.fromkeys(words)),
                  "domains": list(dict.fromkeys(domains)),
                  "co_entities": list(dict.fromkeys(co))}
        if len(merged["words"]) < MIN_INFORMATIVE_WORDS:
            return None
        entry = self._create_entity(surface, ner_type, merged)
        self._surface_evidence.pop(surface, None)
        return entry

    def _create_entity(self, surface: str, ner_type: str, context: dict) -> dict:
        """从（可能跨多次出现聚合的）语境新建动态实体。"""
        entity_id = f"ent_{uuid.uuid4().hex[:10]}"
        cues = {}
        for w in context.get("words", [])[:12]:
            cues[w] = 2.0
        entry = _norm_entry({
            "entity_id": entity_id, "canonical": surface, "aliases": [surface],
            "sense_type": NER_TO_SENSE.get(ner_type, "UNKNOWN"),
            "domains": context.get("domains", []), "cues": cues,
            "related": list(context.get("co_entities", [])),
            "source": "dynamic",
        })
        self._register(entry)
        return entry

    def _update_profile(self, entry: dict, context: dict,
                        boost: float = 1.0) -> None:
        """增量更新实体画像：只增不减，按频次封顶截断（防带偏）。"""
        profile = entry["profile"]
        for w in context.get("words", []):
            profile[w] = profile.get(w, 0) + boost
        if len(profile) > PROFILE_CAP:
            keep = sorted(profile.items(), key=lambda x: x[1],
                          reverse=True)[:PROFILE_CAP]
            entry["profile"] = dict(keep)
        entry["mention_count"] = entry.get("mention_count", 0) + 1

    # -- 待定重估 ---------------------------------------------------------
    def reresolve(self, mention_record: dict,
                  accept_score: float = DEFAULT_ACCEPT_SCORE,
                  accept_margin: float = DEFAULT_ACCEPT_MARGIN,
                  reject_score: float = DEFAULT_REJECT_SCORE) -> Optional[dict]:
        """用当前知识库重新评估一条待定提及。

        只对 status == pending 的提及有意义；能链接则返回新决策与变更集，
        否则返回 None。已链接的提及不应走这里——它们保持稳定。
        """
        if mention_record.get("status") != "pending":
            return None
        context = {
            "words": mention_record.get("context", []),
            "domains": mention_record.get("domains", []),
            "co_entities": mention_record.get("co_entities", []),
        }
        with self._lock:
            decision = self._decide(context, mention_record["text"],
                                    mention_record.get("ner_type", ""),
                                    accept_score, accept_margin, reject_score)
            new_entities = []
            if decision["status"] == "unknown":
                entry = self._handle_unknown(
                    mention_record["text"],
                    mention_record.get("ner_type", ""), context,
                    evidence_key=(mention_record.get("doc_id"),
                                  mention_record.get("start"),
                                  tuple(context.get("words", []))))
                if entry is None:
                    return None
                new_entities = [dict(entry)]
                decision = {"status": "linked", "entry": entry,
                            "score": decision["score"],
                            "margin": decision["margin"],
                            "candidates": decision["candidates"]}
            if decision["status"] != "linked":
                return None
            entry = decision["entry"]
            self._update_profile(entry, context)
            return {
                "entity_id": entry["entity_id"],
                "canonical": entry["canonical"],
                "sense_type": entry["sense_type"],
                "score": decision["score"], "margin": decision["margin"],
                "confidence": self._confidence(decision["score"], accept_score),
                "candidates": decision["candidates"],
                "mutations": {"new_entities": new_entities,
                              "updated_entities": {entry["entity_id"]: dict(entry)}},
            }

    # -- 人工操作 ---------------------------------------------------------
    def manual_link(self, mention_record: dict, entity_id: str,
                    boost: float = 2.0) -> dict:
        """人工把一条提及指派给实体：画像加倍吸收该语境。"""
        with self._lock:
            entry = self.entries.get(entity_id)
            if entry is None:
                raise KeyError(f"实体不存在: {entity_id}")
            context = {
                "words": mention_record.get("context", []),
                "domains": mention_record.get("domains", []),
                "co_entities": mention_record.get("co_entities", []),
            }
            self._update_profile(entry, context, boost=boost)
            return {"new_entities": [],
                    "updated_entities": {entity_id: dict(entry)}}

    def add_alias(self, entity_id: str, alias: str) -> dict:
        """给实体登记别名（同一对象的另一种叫法对上号）。"""
        with self._lock:
            entry = self.entries.get(entity_id)
            if entry is None:
                raise KeyError(f"实体不存在: {entity_id}")
            if alias and alias not in entry["aliases"]:
                entry["aliases"].append(alias)
                self.alias_index.setdefault(alias, set()).add(entity_id)
            return {"new_entities": [],
                    "updated_entities": {entity_id: dict(entry)}}

    def merge_entities(self, src_id: str, dst_id: str) -> dict:
        """把 src 并入 dst：别名/特征词/画像合并，src 留重定向。

        提及记录的改挂由上层持久化（本方法返回受影响的实体新状态）。
        """
        with self._lock:
            src, dst = self.entries.get(src_id), self.entries.get(dst_id)
            if src is None or dst is None:
                raise KeyError("源或目标实体不存在")
            if src_id == dst_id:
                raise ValueError("不能合并到自身")
            if dst.get("merged_into"):
                raise ValueError("目标实体已被合并，请选择最终实体")

            for alias in {src["canonical"], *src["aliases"]}:
                if alias not in dst["aliases"]:
                    dst["aliases"].append(alias)
                self.alias_index.setdefault(alias, set()).discard(src_id)
                self.alias_index.setdefault(alias, set()).add(dst_id)
            for w, weight in src["cues"].items():
                dst["cues"][w] = max(dst["cues"].get(w, 0.0), weight)
            for w, cnt in src["profile"].items():
                dst["profile"][w] = dst["profile"].get(w, 0) + cnt
            dst["mention_count"] = (dst.get("mention_count", 0)
                                    + src.get("mention_count", 0))
            src["merged_into"] = dst_id
            return {"new_entities": [],
                    "updated_entities": {dst_id: dict(dst), src_id: dict(src)}}

    # -- 统计 -------------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            live = [e for e in self.entries.values() if not e.get("merged_into")]
            by_sense: dict[str, int] = {}
            for e in live:
                by_sense[e["sense_type"]] = by_sense.get(e["sense_type"], 0) + 1
            return {
                "total": len(live),
                "merged": len(self.entries) - len(live),
                "dynamic": sum(1 for e in live if e["source"] == "dynamic"),
                "builtin": sum(1 for e in live if e["source"] == "builtin"),
                "by_sense": by_sense,
            }
