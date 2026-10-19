# 別call予算と静的draw上界

sample512、primary construction16、G7 make_generator14、F3.prepare2、AuditParity8、Canonical snapshot6。
B sampling audit最大256、B angle/requery各256、Canonical angle192、Full angle64、driver correctness event_witness512。
Full sample packet128、Full angle packet64、additional sampling0。F3 prepare内部constructor2はprimary16の内数。
event_witness512はdriverのcorrectness境界であり、B sampling audit内の同名helper呼出し（最大256）を別に観測する。同名関数の総call上限512とは呼ばない。

固定m5ではraw degree≤4。Canonicalのgroup/labels/childが最大6draws、Fullのorder/labels/acceptance/childが最大7draws。
Canonical384×6＋Full128×7=3200 requests、各H160。非reducedはdegree+1、coin rejectはdegree+2、acceptedはdegree+3。
audit/requeryは追加drawなし、retry/追加samplingなしという静的条件に依存する。人工fixtureの固定draw数を実lawへ採択しない。
実装は各pairの分配予算と全plan/task bindingを制限し、offlineが最終counter合計とglobal sample512を照合する。
root/law/predicate/packet内部の総関数呼出し数、実variable draws/IIDはNOT_ESTABLISHED。境界予算を内部総call認可とみなさない。
詳細はcall_budget_manifest.json。

**LIVE_RELEASE_CANDIDATE_UNAUTHORIZED。実科学未実行。mandatory STOP。次の担当：ユーザー・GPT。**
