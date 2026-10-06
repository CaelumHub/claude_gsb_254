"""实体消歧 / 链接 / 跨文档对齐 与 存储 update 的单元测试。

运行：``python -m unittest discover -s tests -v``
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.entitylink import EntityResolver, extract_context, BUILTIN_KB
from nlp import get_segmenter, get_ner
from pipeline import PipelineEngine
from storage import ShardedStore


def _mention(text, surface, ner_type="ORGANIZATION"):
    start = text.index(surface)
    return {"start": start, "end": start + len(surface),
            "text": surface, "type": ner_type}


class TestDisambiguation(unittest.TestCase):
    """同一名字在不同语境下应链到不同对象。"""

    @classmethod
    def setUpClass(cls):
        cls.resolver = EntityResolver()

    def _link(self, text, surface="苹果"):
        result = self.resolver.resolve_document(text, doc_id="t")
        hits = [m for m in result["mentions"] if m["text"] == surface]
        self.assertTrue(hits, f"未识别到提及: {surface}")
        return hits[0]

    def test_apple_company(self):
        m = self._link("苹果公司发布了新款手机，股价大涨。")
        self.assertEqual(m["status"], "linked")
        self.assertEqual(m["entity_id"], "kb_apple_inc")

    def test_apple_fruit(self):
        m = self._link("我今天吃了一个苹果，很甜。")
        self.assertEqual(m["status"], "linked")
        self.assertEqual(m["entity_id"], "kb_apple_fruit")

    def test_apple_cooccurrence(self):
        # 上下文实体线索：与华为同框 → 公司
        m = self._link("苹果和华为在手机市场激烈竞争。")
        self.assertEqual(m["entity_id"], "kb_apple_inc")

    def test_xiaomi_company_vs_grain(self):
        company = self._link("小米发布了新款手机。", surface="小米")
        grain = self._link("小米粥很有营养，农户今年丰收。", surface="小米")
        self.assertEqual(company["entity_id"], "kb_xiaomi_inc")
        self.assertEqual(grain["entity_id"], "kb_xiaomi_grain")
        self.assertNotEqual(company["entity_id"], grain["entity_id"])

    def test_amazon_company_vs_river(self):
        company = self._link("亚马逊的财报超出预期。", surface="亚马逊")
        river = self._link("亚马逊雨林覆盖了广阔的流域。", surface="亚马逊")
        self.assertEqual(company["entity_id"], "kb_amazon_inc")
        self.assertEqual(river["entity_id"], "kb_amazon_river")

    def test_thin_context_is_pending_not_guessed(self):
        # 语境不足：标待定而不是乱猜
        m = self._link("苹果不错。")
        self.assertEqual(m["status"], "pending")
        self.assertIsNone(m["entity_id"])
        self.assertTrue(m["candidates"], "待定也应给出候选供人工参考")

    def test_candidates_sorted(self):
        m = self._link("苹果和香蕉都是我爱吃的水果。")
        self.assertEqual(m["entity_id"], "kb_apple_fruit")
        scores = [c["score"] for c in m["candidates"]]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestCrossDocumentResolution(unittest.TestCase):
    """跨文档对齐：同一对象聚到同一名下，同名不同对象分开。"""

    def test_same_object_across_docs(self):
        resolver = EntityResolver()
        m1 = resolver.resolve_document(
            "苹果公司发布了新款手机。", doc_id="d1")["mentions"][0]
        m2 = resolver.resolve_document(
            "苹果的财报显示营收增长。", doc_id="d2")["mentions"][0]
        self.assertEqual(m1["entity_id"], "kb_apple_inc")
        self.assertEqual(m2["entity_id"], "kb_apple_inc")

    def test_same_name_different_objects_not_merged(self):
        resolver = EntityResolver()
        fruit = resolver.resolve_document(
            "今年苹果丰收，果农忙着采摘。", doc_id="d1")["mentions"][0]
        company = resolver.resolve_document(
            "苹果股价创下新高。", doc_id="d2")["mentions"][0]
        self.assertEqual(fruit["entity_id"], "kb_apple_fruit")
        self.assertEqual(company["entity_id"], "kb_apple_inc")

    def test_dynamic_entity_created_and_reattached(self):
        resolver = EntityResolver()
        # 首次出现：隔离观察，只积累证据、标待定，不急着建实体
        text1 = "星辰公司发布了年度财报，营收持续增长。"
        r1 = resolver.resolve_document(
            text1, doc_id="d1", mentions=[_mention(text1, "星辰公司")])
        m1 = r1["mentions"][0]
        self.assertEqual(m1["status"], "pending")
        self.assertEqual(resolver.stats()["dynamic"], 0)

        # 再次出现：证据足够，新建动态实体并链接
        text2 = "星辰公司股价继续上涨。"
        r2 = resolver.resolve_document(
            text2, doc_id="d2", mentions=[_mention(text2, "星辰公司")])
        m2 = r2["mentions"][0]
        self.assertEqual(m2["status"], "linked")
        self.assertTrue(m2["entity_id"].startswith("ent_"))
        self.assertEqual(len(r2["mutations"]["new_entities"]), 1)

        # 首次的待定提及经重估升级到同一个实体（跨文档对上号）
        out = resolver.reresolve(m1)
        self.assertIsNotNone(out)
        self.assertEqual(out["entity_id"], m2["entity_id"])

        # 第三篇文档：直接挂上已有实体，不再新建
        text3 = "星辰公司公布了分红方案。"
        m3 = resolver.resolve_document(
            text3, doc_id="d3", mentions=[_mention(text3, "星辰公司")],
            )["mentions"][0]
        self.assertEqual(m3["entity_id"], m2["entity_id"])
        same_name = [e for e in resolver.entries.values()
                     if e["canonical"] == "星辰公司" and not e.get("merged_into")]
        self.assertEqual(len(same_name), 1)

    def test_ner_noise_does_not_pollute_kb(self):
        # 被 NER 误判的普通词只出现一次：永远待定，不进实体库
        resolver = EntityResolver()
        text = "今年的收成不错。"
        mentions = [{"start": 2, "end": 4, "text": "收成", "type": "PERSON"}]
        m = resolver.resolve_document(text, doc_id="d1",
                                      mentions=mentions)["mentions"][0]
        self.assertEqual(m["status"], "pending")
        self.assertEqual(resolver.stats()["dynamic"], 0)

    def test_pending_does_not_create_entity(self):
        resolver = EntityResolver()
        before = resolver.stats()["total"]
        m = resolver.resolve_document("苹果不错。", doc_id="d1")["mentions"][0]
        self.assertEqual(m["status"], "pending")
        self.assertEqual(resolver.stats()["total"], before)


class TestIncrementalStability(unittest.TestCase):
    """新文档入库后：已对齐的关系稳定，待定可升级，不越积越乱。"""

    def test_reresolve_upgrades_pending_after_kb_growth(self):
        resolver = EntityResolver()
        # 语境太薄 → 待定，且不新建实体
        text1 = "星辰公司如何？"
        r1 = resolver.resolve_document(
            text1, doc_id="d1", mentions=[_mention(text1, "星辰公司")])
        pending = r1["mentions"][0]
        self.assertEqual(pending["status"], "pending")

        # 后续文档带来证据（第二次出现），动态实体建立
        text2 = "星辰公司发布了年度财报。"
        resolver.resolve_document(
            text2, doc_id="d2", mentions=[_mention(text2, "星辰公司")])

        # 重估：待定提及升级为链接，且指向新建的那个实体
        out = resolver.reresolve(pending)
        self.assertIsNotNone(out)
        self.assertEqual(out["canonical"], "星辰公司")
        self.assertEqual(out["entity_id"],
                         resolver._lookup("星辰公司")[0]["entity_id"])

    def test_linked_mentions_never_rejudged(self):
        resolver = EntityResolver()
        linked = resolver.resolve_document(
            "苹果公司发布了新款手机。", doc_id="d1")["mentions"][0]
        self.assertEqual(linked["status"], "linked")
        # reresolve 只处理待定，已链接结果原样保留
        self.assertIsNone(resolver.reresolve(linked))

    def test_profile_grows_but_capped(self):
        resolver = EntityResolver()
        for i in range(5):
            resolver.resolve_document(
                f"苹果公司发布了第{i}款手机，芯片和系统都升级了。",
                doc_id=f"d{i}")
        entry = resolver.entries["kb_apple_inc"]
        self.assertGreaterEqual(entry["mention_count"], 5)
        self.assertLessEqual(len(entry["profile"]), 120)
        self.assertIn("手机", entry["profile"])

    def test_merge_entities_redirects(self):
        resolver = EntityResolver()

        def make_entity(surface, text1, text2):
            resolver.resolve_document(
                text1, doc_id=f"{surface}_1",
                mentions=[_mention(text1, surface)])
            r = resolver.resolve_document(
                text2, doc_id=f"{surface}_2",
                mentions=[_mention(text2, surface)])
            return r["mentions"][0]["entity_id"]

        e1 = make_entity("星辰公司", "星辰公司发布了年度财报。",
                         "星辰公司股价继续上涨。")
        e2 = make_entity("明月公司", "明月公司召开了股东大会。",
                         "明月公司宣布了分红方案。")
        out = resolver.merge_entities(e1, e2)
        self.assertIn(e1, out["updated_entities"])
        self.assertEqual(resolver.entries[e1]["merged_into"], e2)
        # 旧名字路由到合并后的实体
        self.assertEqual(resolver.get_entry(e1)["entity_id"], e2)
        self.assertIn("星辰公司", resolver.entries[e2]["aliases"])

    def test_manual_link_strengthens_profile(self):
        resolver = EntityResolver()
        m = resolver.resolve_document("苹果不错。", doc_id="d1")["mentions"][0]
        self.assertEqual(m["status"], "pending")
        self.assertTrue(m["context"], "待定提及也应保留语境快照")
        mutations = resolver.manual_link(m, "kb_apple_fruit")
        entry = mutations["updated_entities"]["kb_apple_fruit"]
        # 人工指派的语境以加倍权重进入画像
        for w in m["context"]:
            self.assertGreaterEqual(entry["profile"].get(w, 0), 2)


class TestNERJoinRule(unittest.TestCase):
    def test_unknown_company_joined(self):
        # 未登录机构名：专名 + 后缀词被分词拆开时应拼接识别
        ents = get_ner().recognize("星辰公司发布了年度财报。")
        texts = {e["text"]: e["type"] for e in ents}
        self.assertEqual(texts.get("星辰公司"), "ORGANIZATION")

    def test_function_word_not_joined(self):
        # 「的公司」这类功能词 + 后缀不应拼成实体
        ents = get_ner().recognize("他的公司很大。")
        self.assertNotIn("的公司", [e["text"] for e in ents])


class TestPipelineStage(unittest.TestCase):
    def test_entity_link_stage(self):
        engine = PipelineEngine().register_builtin()
        pipe = engine.build({"name": "t", "stages": [
            {"name": "ner"}, {"name": "entity_link"}]})
        self.assertIn("entity_link", pipe.order)
        self.assertIn("ner", pipe.deps.get("entity_link", []))
        ctx = pipe.run({"text": "苹果公司发布了新款手机，股价大涨。"})
        links = ctx["entity_links"]
        apple = [l for l in links if l["text"] == "苹果"]
        self.assertTrue(apple)
        self.assertEqual(apple[0]["entity_id"], "kb_apple_inc")


class TestStoreUpdate(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="store_update_")
        self.store = ShardedStore(self.dir, "t", shard_size=3)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_update_merges_patch(self):
        rid = self.store.insert({"name": "苹果", "status": "pending"})
        ok = self.store.update(rid, {"status": "linked", "score": 4.5})
        self.assertTrue(ok)
        rec = self.store.get(rid)
        self.assertEqual(rec["status"], "linked")
        self.assertEqual(rec["score"], 4.5)
        self.assertEqual(rec["name"], "苹果")  # 未触及字段保留
        self.assertEqual(rec["id"], rid)       # id 不可改

    def test_update_missing_and_deleted(self):
        self.assertFalse(self.store.update("nope", {"a": 1}))
        rid = self.store.insert({"a": 1})
        self.store.delete(rid)
        self.assertFalse(self.store.update(rid, {"a": 2}))

    def test_update_across_shards(self):
        ids = [self.store.insert({"n": i}) for i in range(7)]
        target = ids[5]  # 落在第二个分片
        self.assertTrue(self.store.update(target, {"n": 99}))
        self.assertEqual(self.store.get(target)["n"], 99)


if __name__ == "__main__":
    unittest.main()
