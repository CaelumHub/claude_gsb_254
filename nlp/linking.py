"""实体链接与语义消歧（Entity Linking & Disambiguation）。

NER 只回答「这段字串是不是实体、粗类是什么」，本模块进一步回答两件事：

1. **当前语境下是哪一类**：同一个名字在不同语境可能指向完全不同的对象
   （「苹果」可以是公司也可以是水果，「小米」可以是品牌也可以是粮食）。
   结合三类线索综合判断，而不是只看字面：

   - 搭配线索（动词/量词/后缀，如「一斤～」「～发布了新机」）；
   - 行业词（如「芯片/财报」指向科技，「热带/雨林」指向地理自然）；
   - 上下文实体（同句共现的其它实体，如「雷军/华为」指向公司）。

2. **指向知识库中哪个真实对象**：命中内置知识库（``ENTITY_KB``）时，
   对该名字下的所有候选打分；证据不足 / 候选分差不明显时，宁可标
   ``PENDING``（待定）也不乱猜。

未命中知识库的实体由 :meth:`infer_semantic_type` 基于后缀、行业词做
粗粒度归类，供跨文档对齐层（见 :class:`storage.registry_store.EntityRegistry`）
积累画像、增量聚合同一对象。

判定结果一律附带 ``confidence``（0~1）与 ``evidence``（命中的线索），
便于审计与人工复核。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from .lexicon import (ORG_SUFFIXES, PERSONS, SURNAMES,
                      GIVEN_NAME_CHARS)
from .ner import NERExtractor


# ---------------------------------------------------------------------------
# 语义类型（比 NER 的 PERSON/ORGANIZATION/... 更细）
# ---------------------------------------------------------------------------

SEMANTIC_TYPE_NAMES = {
    "PERSON": "人物",
    "ORG": "组织机构",
    "COMPANY": "企业/公司",
    "GOV": "政府机构",
    "SCHOOL": "学校/科研",
    "LOCATION": "地点/地区",
    "NATURAL": "山川河岳/自然地理",
    "PRODUCT": "产品/品牌型号",
    "FOOD": "食物/农产品",
    "PLANT": "植物",
    "ANIMAL": "动物",
    "WORK": "作品/影视歌书",
    "OTHER": "其他",
    "PENDING": "待定",
}

# NER 粗类 -> 允许的语义类型（用于「类型相容」打分）
NER_TYPE_COMPAT = {
    "PERSON": {"PERSON"},
    "ORGANIZATION": {"ORG", "COMPANY", "GOV", "SCHOOL"},
    "LOCATION": {"LOCATION", "NATURAL"},
}

# 机构后缀 -> 语义类型（未知机构归类用）
_ORG_SUFFIX_TYPE = {
    "公司": "COMPANY", "集团": "COMPANY", "银行": "COMPANY",
    "大学": "SCHOOL", "学院": "SCHOOL", "研究所": "SCHOOL",
    "研究院": "SCHOOL",
    "委员会": "GOV", "政府": "GOV", "总局": "GOV", "部门": "GOV",
    "法院": "GOV", "检察院": "GOV",
}

# 待定语义类型（证据不足，不强行归类）
PENDING = "PENDING"

# ---------------------------------------------------------------------------
# 行业词表（行业线索 / 未知实体画像的行业归属）
# ---------------------------------------------------------------------------

INDUSTRY_LEXICON: dict[str, set[str]] = {
    "科技": {
        "手机", "芯片", "发布", "新机", "财报", "营收", "市值", "上市",
        "互联网", "软件", "硬件", "系统", "智能", "数码", "电商", "股价",
        "融资", "创始人", "发布会", "处理器", "电脑", "平板", "续航",
        "科技", "技术", "用户量", "营收额", "专利", "研发",
    },
    "农业": {
        "种植", "亩产", "丰收", "粮食", "农作物", "农药", "化肥", "田间",
        "稻田", "果园", "农民", "农业", "谷物", "庄稼", "收割", "播种",
        "一斤", "一筐", "含糖量", "糯", "煮粥", "杂粮",
    },
    "饮食": {
        "好吃", "口感", "营养", "水果", "吃", "甜", "脆", "榨汁", "食用",
        "菜谱", "食材", "味道", "酸甜", "果香",
    },
    "地理自然": {
        "热带雨林", "雨林", "河流", "流域", "丛林", "探险", "热带",
        "高原", "山脉", "森林", "自然", "风光", "植被", "生态",
    },
    "文娱": {
        "电影", "导演", "主演", "上映", "票房", "电视剧", "专辑", "歌曲",
        "演唱", "演员", "歌星", "歌手", "节目", "综艺", "小说", "饰演",
        "歌坛", "演艺圈",
    },
}

# 自然/动植物强提示（用于把「水果/花/鸟」字面线索纳入行业判断）
_DOMAIN_HINTS = {
    "饮食": ("水果", "好吃", "甜", "脆", "口感", "榨汁", "吃"),
    "农业": ("粮食", "谷物", "煮粥", "糯", "杂粮", "亩产", "丰收"),
    "地理自然": ("河流", "流域", "雨林", "丛林", "热带", "探险"),
    "文娱": ("演员", "歌星", "歌手", "导演", "电影", "专辑", "演唱"),
}

# 行业相容组：细类不同但实指同一大领域（农产品既会出现「农业」词，
# 也会出现「水果/甜」等饮食词），匹配时按大领域相容、不互斥。
INDUSTRY_GROUPS = [
    {"农业", "饮食"},
]


def industries_compatible(a: set[str], b: set[str]) -> bool:
    """两个行业集合是否相容（直接交集，或落在同一相容组）。"""
    if a & b:
        return True
    for group in INDUSTRY_GROUPS:
        if (a & group) and (b & group):
            return True
    return False

# ---------------------------------------------------------------------------
# 内置迷你知识库：歧义名字 -> 候选实体
#
# 每个候选的字段：
#   kb_id       知识库稳定 id
#   name        规范名
#   sem_type    语义类型
#   industry    行业（粗粒度，用于画像互斥/相容）
#   aliases     别名集合（命中也算同一对象）
#   colloc      搭配/行业线索词，命中加权
#   context_ent 上下文共现实体线索，命中强加权
# ---------------------------------------------------------------------------

ENTITY_KB: dict[str, list[dict]] = {
    "苹果": [
        {
            "kb_id": "kb_apple_company", "name": "苹果公司", "sem_type": "COMPANY",
            "industry": "科技",
            "aliases": {"Apple", "苹果公司", "美国苹果"},
            "colloc": {"手机", "发布", "新机", "芯片", "财报", "营收", "市值",
                       "iPhone", "电脑", "平板", "系统", "股价", "上市",
                       "科技", "创始人"},
            "context_ent": {"乔布斯", "库克", "华为", "小米", "谷歌", "微软",
                            "腾讯", "富士康"},
        },
        {
            "kb_id": "kb_apple_fruit", "name": "苹果（水果）", "sem_type": "FOOD",
            "industry": "饮食",
            "aliases": {"红富士", "红富士苹果"},
            "colloc": {"一斤", "一筐", "好吃", "甜", "脆", "口感", "营养",
                       "水果", "榨汁", "果园", "吃", "糖分", "新鲜"},
            "context_ent": {"香蕉", "梨", "葡萄", "果农"},
        },
    ],
    "小米": [
        {
            "kb_id": "kb_xiaomi_company", "name": "小米集团", "sem_type": "COMPANY",
            "industry": "科技",
            "aliases": {"小米集团", "小米公司", "Xiaomi", "红米", "Redmi"},
            "colloc": {"手机", "发布", "新机", "芯片", "财报", "营收", "市值",
                       "性价比", "智能家居", "生态链", "系统", "股价", "创始人",
                       "MIUI", "发布会"},
            "context_ent": {"雷军", "华为", "苹果", "OPPO", "vivo", "魅族",
                            "京东", "高通"},
        },
        {
            "kb_id": "kb_millet_grain", "name": "小米（粮食）", "sem_type": "FOOD",
            "industry": "农业",
            "aliases": {"粟", "谷子", "黄米"},
            "colloc": {"一斤", "粮食", "谷物", "煮粥", "杂粮", "糯", "养胃",
                       "亩产", "丰收", "庄稼", "农民", "收割", "熬粥"},
            "context_ent": {"大米", "玉米", "高粱", "水稻", "麦子"},
        },
    ],
    "华为": [
        {
            "kb_id": "kb_huawei", "name": "华为技术有限公司", "sem_type": "COMPANY",
            "industry": "科技",
            "aliases": {"华为公司", "HUAWEI"},
            "colloc": {"手机", "芯片", "通信", "5G", "基站", "发布", "新机",
                       "鸿蒙", "财报", "研发", "专利", "运营商"},
            "context_ent": {"任正非", "苹果", "小米", "中兴", "腾讯"},
        },
    ],
    "亚马逊": [
        {
            "kb_id": "kb_amazon_company", "name": "亚马逊公司", "sem_type": "COMPANY",
            "industry": "科技",
            "aliases": {"Amazon", "亚马逊公司"},
            "colloc": {"电商", "云计算", "AWS", "财报", "营收", "市值",
                       "购物", "快递", "平台", "上市", "股价", "Kindle"},
            "context_ent": {"贝索斯", "谷歌", "微软", "苹果", "阿里巴巴"},
        },
        {
            "kb_id": "kb_amazon_river", "name": "亚马孙河", "sem_type": "NATURAL",
            "industry": "地理自然",
            "aliases": {"亚马孙河", "亚马逊河", "亚马孙"},
            "colloc": {"河流", "流域", "热带雨林", "雨林", "丛林", "探险",
                       "热带", "生态", "植被", "流量"},
            "context_ent": {"巴西", "安第斯山", "大西洋"},
        },
    ],
    "杜鹃": [
        {
            "kb_id": "kb_cuckoo_bird", "name": "杜鹃（鸟）", "sem_type": "ANIMAL",
            "industry": "地理自然",
            "aliases": {"布谷鸟", "子规"},
            "colloc": {"鸟", "鸟类", "鸣叫", "巢寄生", "羽毛", "栖息", "山林",
                       "孵卵", "飞", "观赏鸟"},
            "context_ent": {"黄鹂", "喜鹊"},
        },
        {
            "kb_id": "kb_azalea_plant", "name": "杜鹃花", "sem_type": "PLANT",
            "industry": "地理自然",
            "aliases": {"杜鹃花", "映山红"},
            "colloc": {"花", "开花", "花瓣", "种植", "盆栽", "盛开", "花园",
                       "花卉", "枝叶", "园艺", "赏花"},
            "context_ent": {"茶花", "兰花"},
        },
        {
            "kb_id": "kb_dujuan_person", "name": "杜鹃（人物）", "sem_type": "PERSON",
            "industry": "文娱",
            "aliases": set(),
            "colloc": {"演员", "歌星", "歌手", "饰演", "主演", "专辑", "节目",
                       "小姐", "女士", "老师"},
            "context_ent": {"导演"},
        },
    ],
    "浪潮": [
        {
            "kb_id": "kb_inspur_company", "name": "浪潮集团", "sem_type": "COMPANY",
            "industry": "科技",
            "aliases": {"浪潮集团", "浪潮信息", "Inspur"},
            "colloc": {"服务器", "云计算", "软件", "信息技术", "数据中心",
                       "财报", "营收", "数字经济", "AI", "算力"},
            "context_ent": {"华为", "联想"},
        },
        {
            "kb_id": "kb_wave_meaning", "name": "浪潮（比喻/自然现象）",
            "sem_type": "OTHER",
            "industry": "",
            "aliases": set(),
            "colloc": {"掀起", "改革", "一股", "时代", "涌来", "浪潮般",
                       "海水", "海浪", "波涛"},
            "context_ent": set(),
        },
    ],
    "阿里巴巴": [
        {
            "kb_id": "kb_alibaba", "name": "阿里巴巴集团", "sem_type": "COMPANY",
            "industry": "科技",
            "aliases": {"阿里巴巴集团", "阿里", "Alibaba"},
            "colloc": {"电商", "淘宝", "天猫", "支付宝", "云计算", "财报",
                       "营收", "平台", "上市", "新零售"},
            "context_ent": {"马云", "腾讯", "京东", "亚马逊"},
        },
    ],
    "腾讯": [
        {
            "kb_id": "kb_tencent", "name": "腾讯控股", "sem_type": "COMPANY",
            "industry": "科技",
            "aliases": {"腾讯公司", "Tencent"},
            "colloc": {"微信", "QQ", "游戏", "社交", "财报", "营收", "互联网",
                       "平台", "视频号", "王者荣耀"},
            "context_ent": {"马化腾", "阿里巴巴", "百度", "字节跳动"},
        },
    ],
}

# 名字 -> kb_id 的别名反查表（惰性构建）
_ALIAS_INDEX: Optional[dict[str, str]] = None

# 打分权重
W_COLLOC = 2.5       # 每个搭配/行业线索
W_INDUSTRY = 3.0     # 行业词命中（比普通搭配更强）
W_CONTEXT = 4.0      # 上下文共现实体
W_TYPE_COMPAT = 1.5  # NER 粗类与语义类型相容

# 判定阈值
ACCEPT_MIN_SCORE = 5.0     # 最高分至少达到该值才接受
ACCEPT_MARGIN = 2.0        # 冠亚军分差小于该值则证据冲突 -> 待定
SINGLE_ACCEPT = 3.0        # 唯一候选时，达到该值即可链接

# 两个名字本身都有歧义时（如「苹果」与「小米」共现），互为共现属于
# 循环证据，不能当成强线索，只给一个较小的权重。
AMBIGUOUS_COOCCUR_WEIGHT = 0.5

# 机构名前常见的动词/虚词（用于把后缀扫描截出的专名去前导）
_ORG_LEAD_CUTS = set("了在和与及对把被让给从到向为对是有的于赴往考察访问"
                     "参观调研表示认为称将已正也都又很最不没")


def _org_literals() -> set[str]:
    """知识库中属于机构大类的别名/名字面量集合。"""
    org_types = {"ORG", "COMPANY", "GOV", "SCHOOL"}
    names: set[str] = set()
    for candidates in ENTITY_KB.values():
        for cand in candidates:
            if cand["sem_type"] in org_types:
                names.add(cand["name"])
                names.update(cand.get("aliases", ()))
    return names


_ORG_LITERALS = _org_literals()


# ---------------------------------------------------------------------------
# 实体链接器
# ---------------------------------------------------------------------------

class EntityLinker:
    """结合 NER 结果与上下文，对实体做语义消歧并链接到知识库。"""

    def __init__(self, ner: Optional[NERExtractor] = None,
                 context_window: int = 40):
        self.ner = ner or NERExtractor()
        # 线索只取实体前后 ``context_window`` 个字符，避免跨句串味
        self.context_window = context_window

    # -- 对外接口 ---------------------------------------------------------
    def link(self, text: str, entities: Optional[list[dict]] = None,
             collect_extra: bool = True) -> list[dict]:
        """对文本中的实体逐个消歧。

        :param text: 原始文本
        :param entities: NER 结果（``start/end/text/type``）；
            为空时自动跑一次 NER。
        :param collect_extra: 是否额外召回 NER 漏掉、但知识库明确收录的
            实体（如 NER 漏识别的「杜鹃/亚马逊」）。
        :return: 每个实体的链接结果，字段：

            - ``text/start/end/ner_type``：原 NER 信息
            - ``sem_type``：语义类型；判不了为 ``PENDING``
            - ``kb_id/name``：链接到的知识库对象；未命中为 ``None``
            - ``status``：``LINKED``（已链接）/ ``KB_PENDING``（知识库有
              多个候选但证据不足）/ ``UNKNOWN``（知识库未收录）
            - ``confidence``：0~1
            - ``industry``：推断行业
            - ``evidence``：命中的线索（搭配/行业/共现/类型相容）
            - ``candidates``：各候选打分，便于复核
        """
        if entities is None:
            entities = self.ner.recognize(text)

        # NER 只给出粗类；同名字面（如「小米」）可能被标成 ORGANIZATION，
        # 需以知识库为准重新消歧，故这里要把所有提到都纳入。
        mentions = self._prepare_mentions(text, entities, collect_extra)

        results = []
        for mention in mentions:
            surface = mention["text"]
            # 别名召回（如 Apple）时窗口中心按实际字串展开
            form = mention.get("surface_form", surface)
            window = self._context_window(text, mention["start"], mention["end"])
            if surface in ENTITY_KB:
                result = self._link_to_kb(surface, window, mention)
            else:
                result = self._link_unknown(surface, window, mention)
            # 结果文本回填为实际出现的字串
            result["text"] = form
            if form != surface:
                result["surface_form"] = form
            results.append(result)
        results.sort(key=lambda r: r["start"])
        return results

    def candidates(self, surface: str, context: str,
                   ner_type: Optional[str] = None) -> list[dict]:
        """返回某名字在给定上下文下各候选的打分（调试/人工选择用）。"""
        if surface not in ENTITY_KB:
            return []
        mention = {"text": surface, "ner_type": ner_type}
        return self._score_candidates(surface, context, mention)

    # -- 提及准备 ---------------------------------------------------------
    def _prepare_mentions(self, text: str, entities: list[dict],
                          collect_extra: bool) -> list[dict]:
        mentions = []
        for e in entities:
            # 规则 NER 的 2 字「姓+常用字」人名误报较多（丰收/水果/宣布…），
            # 链接层只保留高精度人名；但知识库收录的多义名字（如「杜鹃」
            # 可能是人/鸟/花）即使被 NER 误标人名也要保留，交给知识库消歧。
            if e.get("type") == "PERSON" and \
                    e["text"] not in ENTITY_KB and \
                    self._is_unreliable_person(e["text"]):
                continue
            mentions.append({
                "text": e["text"], "start": e.get("start", -1),
                "end": e.get("end", -1), "ner_type": e.get("type"),
            })

        if collect_extra:
            occupied = [(m["start"], m["end"]) for m in mentions
                        if m["start"] >= 0]
            # 既扫歧义名字本身，也扫各候选的别名（Apple/红富士/Amazon…）。
            targets: dict[str, str] = {}
            for surface, candidates in ENTITY_KB.items():
                targets[surface] = surface
                for cand in candidates:
                    for alias in cand.get("aliases", ()):
                        if len(alias) >= 2:
                            targets.setdefault(alias, surface)
            for alias, surface in targets.items():
                self._scan_surface(text, alias, surface, mentions, occupied)
            # 规则 NER 依赖分词整词，未知机构名常被切碎而漏识别；
            # 直接在原文上按机构后缀补召回。
            self._scan_org_suffixes(text, mentions, occupied)
        return mentions

    @staticmethod
    def _scan_surface(text: str, literal: str, surface: str,
                      mentions: list, occupied: list) -> None:
        start = 0
        while True:
            pos = text.find(literal, start)
            if pos < 0:
                break
            end = pos + len(literal)
            if not any(pos < b and a < end for a, b in occupied):
                mentions.append({
                    "text": surface,
                    "surface_form": literal if literal != surface else None,
                    "start": pos, "end": end,
                    "ner_type": "ORGANIZATION" if literal in _ORG_LITERALS else None,
                })
                occupied.append((pos, end))
            start = end

    @staticmethod
    def _scan_org_suffixes(text: str, mentions: list,
                           occupied: list) -> None:
        """在原文上扫描「专名 + 机构后缀」，补回被分词切碎的机构。

        取后缀前最多 10 个汉字、并从最后一个动词/虚词之后开始，尽量
        截出干净的专名（如在「…考察红星科技公司」里得到「红星科技公司」）。
        """
        for m in re.finditer(r"[一-鿿]{2,12}?(?:公司|集团|大学|学院|银行|"
                             r"研究所|研究院|基金会)", text):
            span, pos, end = m.group(), m.start(), m.end()
            if any(pos < b and a < end for a, b in occupied):
                continue
            cut = 0
            for i, ch in enumerate(span[:-2]):
                if ch in _ORG_LEAD_CUTS:
                    cut = i + 1
            name = span[cut:]
            name_start = end - len(name)
            if len(name) < 3:
                continue
            mentions.append({
                "text": name, "start": name_start, "end": end,
                "ner_type": "ORGANIZATION",
            })
            occupied.append((name_start, end))

    @staticmethod
    def _is_unreliable_person(surface: str) -> bool:
        """规则 NER 给的人名是否不可信（常见词误报）。"""
        if surface in PERSONS:
            return False
        if len(surface) == 3 and surface[0] in SURNAMES and \
                surface[1] in GIVEN_NAME_CHARS:
            return False
        return True

    def _context_window(self, text: str, start: int, end: int) -> str:
        if start < 0:
            return text
        left = max(0, start - self.context_window)
        right = min(len(text), end + self.context_window)
        return text[left:right]

    # -- 知识库链接 -------------------------------------------------------
    def _link_to_kb(self, surface: str, context: str,
                    mention: dict) -> dict:
        scored = self._score_candidates(surface, context, mention)
        top = scored[0]
        runner = scored[1] if len(scored) > 1 else None
        margin = (top["score"] - runner["score"]) if runner else top["score"]

        if len(scored) == 1:
            accepted = top["score"] >= SINGLE_ACCEPT
        else:
            accepted = (top["score"] >= ACCEPT_MIN_SCORE
                        and margin >= ACCEPT_MARGIN)

        confidence = self._confidence(top, runner)
        result = self._base_result(mention)
        if accepted:
            result.update({
                "sem_type": top["sem_type"], "kb_id": top["kb_id"],
                "name": top["name"], "status": "LINKED",
                "confidence": round(confidence, 3),
                "industry": top.get("industry", ""),
                "evidence": top["evidence"],
            })
        else:
            result.update({
                "sem_type": PENDING, "kb_id": None, "name": None,
                "status": "KB_PENDING", "confidence": round(confidence, 3),
                "industry": self._vote_industry(context),
                "evidence": top["evidence"],
            })
        result["candidates"] = scored
        return result

    def _score_candidates(self, surface: str, context: str,
                          mention: dict) -> list[dict]:
        ner_type = mention.get("ner_type")
        compatible = NER_TYPE_COMPAT.get(ner_type or "", set())
        # 上下文实体：当前窗口里出现的其它已知实体名
        present_ents = self._present_entities(context, exclude=surface)
        # 行业票（用于行业级加权）
        industries = self._vote_industries(context)

        scored = []
        for cand in ENTITY_KB[surface]:
            score = 0.0
            evidence: list[str] = []

            colloc_hits = {w for w in cand.get("colloc", ()) if w in context}
            if colloc_hits:
                score += W_COLLOC * len(colloc_hits)
                evidence.append("搭配:" + "/".join(sorted(colloc_hits)[:6]))

            if cand.get("industry") and cand["industry"] in industries:
                score += W_INDUSTRY
                evidence.append(f"行业:{cand['industry']}")

            ent_hits = {e for e in cand.get("context_ent", ())
                        if e in present_ents}
            if ent_hits:
                # 共现词自身也是多义名字（苹果×小米）时只给弱权重，
                # 避免两个歧义词互相「抬轿」。
                strong = {e for e in ent_hits if e not in ENTITY_KB
                          or len(ENTITY_KB[e]) == 1}
                weak = ent_hits - strong
                weight = W_CONTEXT * len(strong) + \
                    W_CONTEXT * AMBIGUOUS_COOCCUR_WEIGHT * len(weak)
                score += weight
                evidence.append("共现:" + "/".join(sorted(ent_hits)))

            if cand["sem_type"] in compatible:
                score += W_TYPE_COMPAT
                evidence.append("类型相容")

            scored.append({
                "kb_id": cand["kb_id"], "name": cand["name"],
                "sem_type": cand["sem_type"],
                "industry": cand.get("industry", ""),
                "score": round(score, 2),
                "evidence": evidence,
            })

        scored.sort(key=lambda c: c["score"], reverse=True)
        return scored

    @staticmethod
    def _confidence(top: dict, runner: Optional[dict]) -> float:
        """把「最高分 + 分差」折算成 0~1 的置信度。"""
        if top["score"] <= 0:
            return 0.0
        spread = (top["score"] - runner["score"]) if runner else top["score"]
        # 最高分部分饱和到 0.7，分差部分再补 0.3
        import math
        score_part = 1.0 - math.exp(-top["score"] / 6.0)      # 0~0.7 附近
        margin_part = 1.0 - math.exp(-max(spread, 0) / 4.0)
        return min(1.0, 0.7 * score_part + 0.3 * margin_part)

    # -- 未收录实体 -------------------------------------------------------
    def _link_unknown(self, surface: str, context: str,
                      mention: dict) -> dict:
        sem_type, type_conf, type_evidence = self.infer_semantic_type(
            surface, context, mention.get("ner_type"))
        result = self._base_result(mention)
        result.update({
            "sem_type": sem_type, "kb_id": None, "name": None,
            "status": "UNKNOWN",
            "confidence": round(type_conf, 3),
            "industry": self._vote_industry(context),
            "evidence": type_evidence,
            "candidates": [],
        })
        return result

    def infer_semantic_type(self, surface: str, context: str,
                            ner_type: Optional[str] = None
                            ) -> tuple[str, float, list[str]]:
        """知识库未收录时的粗归类：后缀规则 + 行业词。"""
        evidence: list[str] = []

        # 1) 机构后缀细分（公司/学校/政府）
        if ner_type == "ORGANIZATION":
            for suffix, sem in _ORG_SUFFIX_TYPE.items():
                if surface.endswith(suffix):
                    return sem, 0.9, [f"后缀:{suffix}"]
            return "ORG", 0.6, ["机构（无明确后缀）"]

        if ner_type == "PERSON":
            return "PERSON", 0.85, ["NER:人名"]
        if ner_type == "LOCATION":
            # 自然地理后缀 vs 行政地区
            for suffix in ("河", "江", "山", "湖", "海", "洋", "岭", "峰",
                           "岛", "森林", "雨林"):
                if surface.endswith(suffix):
                    return "NATURAL", 0.85, [f"后缀:{suffix}"]
            return "LOCATION", 0.7, ["NER:地名"]

        # 2) 无 NER 粗类：靠后缀 / 行业线索兜底
        for suffix, sem in _ORG_SUFFIX_TYPE.items():
            if surface.endswith(suffix) and len(surface) > len(suffix):
                return sem, 0.8, [f"后缀:{suffix}"]
        for suffix in ("公司", "集团", "银行"):
            if surface.endswith(suffix):
                return "COMPANY", 0.8, [f"后缀:{suffix}"]

        # 作品号/号化提示
        if any(w in context for w in ("电影", "小说", "专辑", "电视剧")) and \
                surface in context:
            if any(w in context for w in ("《", "》")):
                return "WORK", 0.55, ["书名号/作品语境"]

        # 3) 实在判不了
        return PENDING, 0.2, evidence

    # -- 特征（供跨文档对齐层使用） ---------------------------------------
    def extract_features(self, surface: str, context: str,
                         ner_type: Optional[str] = None) -> dict:
        """提取一个提及的对齐画像特征。

        返回 ``sem_type / industries / context_entities / aliases_hint``，
        供 :class:`storage.registry_store.EntityRegistry` 做跨文档聚合。
        """
        sem_type, _, _ = self.infer_semantic_type(surface, context, ner_type)
        industries = self._vote_industries(context)
        return {
            "sem_type": sem_type,
            "industries": sorted(industries),
            "context_entities": sorted(
                self._present_entities(context, exclude=surface)),
        }

    # -- 上下文工具 -------------------------------------------------------
    def _present_entities(self, context: str, exclude: str = "") -> set[str]:
        """窗口中出现的、知识库收录的实体名（作为共现线索）。"""
        names = set()
        for surface, candidates in ENTITY_KB.items():
            if surface == exclude:
                continue
            if surface in context:
                names.add(surface)
            for cand in candidates:
                for alias in cand.get("aliases", ()):
                    if alias and alias in context:
                        names.add(alias)
        return names

    @staticmethod
    def _vote_industries(context: str) -> Counter:
        votes: Counter = Counter()
        for industry, words in INDUSTRY_LEXICON.items():
            hits = sum(1 for w in words if w in context)
            if hits:
                votes[industry] += hits
        for industry, hints in _DOMAIN_HINTS.items():
            if any(h in context for h in hints):
                votes[industry] += 1
        return votes

    def _vote_industry(self, context: str) -> str:
        votes = self._vote_industries(context)
        return votes.most_common(1)[0][0] if votes else ""

    @staticmethod
    def _base_result(mention: dict) -> dict:
        return {
            "text": mention["text"],
            "start": mention.get("start", -1),
            "end": mention.get("end", -1),
            "ner_type": mention.get("ner_type"),
            "sem_type": PENDING, "kb_id": None, "name": None,
            "status": "PENDING", "confidence": 0.0, "industry": "",
            "evidence": [], "candidates": [],
        }


def link_entities(text: str, entities: Optional[list[dict]] = None) -> list[dict]:
    """便捷函数：对一段文本做实体链接。"""
    return EntityLinker().link(text, entities)
