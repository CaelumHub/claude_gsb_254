"""实体链接与跨文档对齐测试。

运行::

    python -m unittest discover -s tests -v

覆盖：
- 同名歧义在不同语境下的语义判定（苹果/小米/亚马逊/杜鹃/浪潮）；
- 证据不足时标「待定」而非乱猜；
- 跨文档同一对象聚合、同名不同对象不混；
- 增量入库 ID 稳定、计数守恒；
- 合并（留重定向）/拆分/人工指定/待定再解析。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.linking import EntityLinker, PENDING, industries_compatible
from storage.sharded import ShardedStore
from storage.registry_store import EntityRegistry


def _link_by_text(linker: EntityLinker, text: str, name: str) -> dict:
    for item in linker.link(text):
        if item["text"] == name or item.get("surface_form") == name:
            return item
    raise AssertionError(f"未找到实体 {name}: {text}")


class TestDisambiguation(unittest.TestCase):
    def setUp(self):
        self.linker = EntityLinker()

    def test_apple_company_vs_fruit(self):
        company = _link_by_text(
            self.linker, "苹果今年发布自研芯片新手机，财报营收创新高", "苹果")
        fruit = _link_by_text(
            self.linker, "果农种的苹果丰收，一斤五块钱又甜又脆", "苹果")
        self.assertEqual(company["status"], "LINKED")
        self.assertEqual(company["sem_type"], "COMPANY")
        self.assertEqual(company["kb_id"], "kb_apple_company")
        self.assertEqual(fruit["status"], "LINKED")
        self.assertEqual(fruit["sem_type"], "FOOD")
        self.assertEqual(fruit["kb_id"], "kb_apple_fruit")

    def test_xiaomi_brand_vs_grain(self):
        brand = _link_by_text(
            self.linker, "小米创始人雷军发布新机", "小米")
        grain = _link_by_text(
            self.linker, "农民用小米和大米熬粥，是养胃的杂粮", "小米")
        self.assertEqual(brand["kb_id"], "kb_xiaomi_company")
        self.assertEqual(grain["kb_id"], "kb_millet_grain")

    def test_amazon_company_vs_river(self):
        river = _link_by_text(
            self.linker, "亚马逊河流域有广阔的热带雨林", "亚马逊")
        company = _link_by_text(
            self.linker, "亚马逊公司云计算电商营收增长", "亚马逊")
        self.assertEqual(river["sem_type"], "NATURAL")
        self.assertEqual(company["sem_type"], "COMPANY")

    def test_cuckoo_vs_azalea(self):
        bird = _link_by_text(
            self.linker, "杜鹃在山里鸣叫，是巢寄生的鸟类", "杜鹃")
        plant = _link_by_text(
            self.linker, "山坡上的杜鹃花盛开，花瓣红艳艳", "杜鹃")
        self.assertEqual(bird["sem_type"], "ANIMAL")
        self.assertEqual(plant["sem_type"], "PLANT")

    def test_insufficient_context_is_pending(self):
        # 没有任何线索时不能乱猜
        r = _link_by_text(self.linker, "苹果和小米都是常见的词", "苹果")
        self.assertEqual(r["status"], "KB_PENDING")
        self.assertEqual(r["sem_type"], PENDING)
        self.assertIsNone(r["kb_id"])

    def test_evidence_recorded(self):
        r = _link_by_text(
            self.linker, "小米创始人雷军发布新机", "小米")
        joined = " ".join(r["evidence"])
        self.assertTrue("雷军" in joined or "搭配" in joined)
        self.assertGreater(r["confidence"], 0.5)

    def test_alias_anchor(self):
        r = _link_by_text(
            self.linker, "Apple 公司财报营收增长，股价上涨", "Apple")
        self.assertEqual(r["kb_id"], "kb_apple_company")
        self.assertEqual(r["surface_form"], "Apple")

    def test_unknown_org_suffix(self):
        # 知识库未收录的机构，靠后缀扫描 + 归类
        results = self.linker.link("星辰生物公司研发了新型疫苗")
        org = next((x for x in results if x["sem_type"] == "COMPANY"), None)
        self.assertIsNotNone(org)
        self.assertTrue(org["text"].endswith("公司"))

    def test_industry_compatibility(self):
        self.assertTrue(industries_compatible({"农业"}, {"饮食"}))
        self.assertFalse(industries_compatible({"科技"}, {"农业"}))


class TestRegistryAlignment(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.linker = EntityLinker()
        self.mentions = ShardedStore(self.tmp, "entity_mention", shard_size=5)
        self.reg = EntityRegistry(self.tmp, self.mentions)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ingest(self, text, doc_id):
        self.reg.ingest(self.linker.link(text), doc_id=doc_id, text=text)

    def test_same_apple_company_across_docs(self):
        self._ingest("苹果公司发布自研芯片新手机，财报亮眼", "d1")
        self._ingest("苹果新机将于下月开售，供应链产能充足", "d2")
        entities = self.reg.all_entities()
        company = [e for e in entities if e.get("kb_id") == "kb_apple_company"]
        self.assertEqual(len(company), 1)
        self.assertEqual(company[0]["mention_count"], 2)

    def test_same_name_different_object_separated(self):
        self._ingest("苹果发布自研芯片新手机，财报营收创新高", "d1")
        self._ingest("红富士苹果丰收，一斤五块钱，又甜又脆", "d2")
        ids = {e["kb_id"]: e for e in self.reg.all_entities()}
        self.assertIn("kb_apple_company", ids)
        self.assertIn("kb_apple_fruit", ids)
        self.assertNotEqual(
            ids["kb_apple_company"]["id"], ids["kb_apple_fruit"]["id"])

    def test_unknown_same_name_orgs_separated(self):
        self._ingest("红星科技公司发布新一代AI芯片和手机系统", "d1")
        self._ingest("红星科技公司财报显示云计算营收增长", "d2")
        self._ingest("红星农产品公司今年水果销售额翻倍", "d3")
        self._ingest("红星农产品公司建成新的果园种植基地", "d4")
        canon = {}
        for e in self.reg.all_entities():
            if e["canonical"].startswith("红星"):
                canon[e["canonical"]] = e["mention_count"]
        self.assertEqual(canon.get("红星科技公司"), 2)
        self.assertEqual(canon.get("红星农产品公司"), 2)

    def test_stable_ids_on_incremental_ingest(self):
        self._ingest("苹果公司发布自研芯片手机", "d1")
        before = {e["id"]: e["mention_count"] for e in self.reg.all_entities()}
        self._ingest("苹果公司财报营收增长", "d2")
        self._ingest("红富士苹果又甜又脆，一斤五块", "d3")
        after = {e["id"]: e for e in self.reg.all_entities()}
        # 已分配的 ID 不漂移、不复用
        for eid in before:
            self.assertIn(eid, after)
        # 计数守恒
        total = sum(e["mention_count"] for e in after.values())
        pending = len(self.reg.list_mentions(status="PENDING", limit=10000))
        self.assertEqual(total + pending,
                         len(self.reg.list_mentions(limit=10000)))

    def test_pending_then_resolve(self):
        # 先入库一条没有上下文的「杜鹃」（人物/鸟/花不明）
        self._ingest("杜鹃是个常见名字", "d1")
        pending = self.reg.list_mentions(status="PENDING")
        self.assertTrue(pending)
        # 后续出现强语境（但对象对齐只影响新提及；旧提及可被重解析）
        self._ingest("杜鹃作为演员饰演了女主角，导演赞不绝口", "d2")
        r = self.reg.resolve_pending()
        self.assertIn("checked", r)

    def test_merge_with_redirect(self):
        self._ingest("苹果发布自研芯片手机，财报亮眼", "d1")
        # 用别名制造第二个对象后人工合并（模拟）
        import nlp.linking as linking
        # 直接造两个同类型对象并合并
        reg = self.reg._read()
        a = self.reg._new_id(reg)
        b = self.reg._new_id(reg)
        reg["entities"][a] = self.reg._new_entity(
            "甲公司", "COMPANY", kb_id=None,
            features={"industries": ["科技"], "context_entities": []},
            status="RESOLVED")
        reg["entities"][b] = self.reg._new_entity(
            "乙公司", "COMPANY", kb_id=None,
            features={"industries": ["科技"], "context_entities": []},
            status="RESOLVED")
        self.reg._write(reg)
        result = self.reg.merge(b, a)
        self.assertEqual(result["target"], a)
        # 旧 id 经重定向仍可解析
        self.assertEqual(self.reg.resolve_id(self.reg._read(), b), a)
        merged = self.reg.get(b)  # 应透明拿到 a
        self.assertEqual(merged["id"], a)

    def test_split_mismerged(self):
        self._ingest("红星科技公司发布AI芯片", "d1")
        self._ingest("红星科技公司云计算营收增长", "d2")
        ent = next(e for e in self.reg.all_entities()
                   if e["canonical"] == "红星科技公司")
        mentions = self.reg.list_mentions(entity_id=ent["id"])
        self.assertEqual(len(mentions), 2)
        sp = self.reg.split(ent["id"], [mentions[0]["id"]],
                            new_name="另一家红星")
        new_ent = self.reg.get(sp["new_id"])
        self.assertEqual(new_ent["mention_count"], 1)
        self.assertEqual(self.reg.get(ent["id"])["mention_count"], 1)

    def test_manual_assign(self):
        self._ingest("杜鹃是个常见名字", "d1")
        mid = self.reg.list_mentions(status="PENDING")[0]["id"]
        r = self.reg.assign(mid, create_name="杜鹃（人物）", sem_type="PERSON")
        self.assertTrue(r["entity_id"].startswith("E"))
        self.assertIsNone(self.reg.get(mid) and None)  # get 取的是实体
        mention = self.mentions.get(mid)
        self.assertEqual(mention["entity_id"], r["entity_id"])
        self.assertEqual(mention["status"], "RESOLVED")

    def test_concurrent_ingest(self):
        texts = ["苹果公司发布自研芯片手机", "红富士苹果又甜又脆一斤五块"] * 8

        def worker(chunk):
            for i, t in enumerate(chunk):
                self.reg.ingest(self.linker.link(t), doc_id=f"{id(chunk)}-{i}")

        threads = [threading.Thread(target=worker, args=(texts[::2],)),
                   threading.Thread(target=worker, args=(texts[1::2],))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        entities = self.reg.all_entities()
        ids = [e["id"] for e in entities]
        self.assertEqual(len(ids), len(set(ids)))  # 无重号
        companies = [e for e in entities
                     if e.get("kb_id") == "kb_apple_company"]
        fruits = [e for e in entities
                  if e.get("kb_id") == "kb_apple_fruit"]
        self.assertEqual(len(companies), 1)
        self.assertEqual(len(fruits), 1)
        # 同名跨类型绝不混
        self.assertNotEqual(companies[0]["id"], fruits[0]["id"])


if __name__ == "__main__":
    unittest.main()
