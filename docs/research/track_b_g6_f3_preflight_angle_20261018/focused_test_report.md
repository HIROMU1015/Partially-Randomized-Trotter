# F3限定人工テスト報告

**最終63件PASS、FAIL/ERROR/SKIP各0。** 登録分子入力、保存科学データ、backend、回路/量子行列を使わず、F3の新規testファイル1本だけを選択した。管理/公開チェックとは別の実際の人工correctness実行である。

## 1. 実行scopeと再現

標準unittestのexact file loaderを専用runnerから使用した。全repo pytest、既存tests、conftest/plugins、科学runnerは起動しない。新3SOURCEと旧g7_generator/return_aggregationのpure rational関数だけを小規模な人工入力で呼ぶ。providerはhash読取だけでimportしない。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src python -B docs/research/track_b_g6_f3_preflight_angle_20261018/focused_test_runner.py /tmp/f3-independent-check
```

上記は同じ限定人工検証を将来再現するためのコマンド例。本タスク完了後の新たな実行を開始するものではない。runnerは既存log/JSONを上書きしない。対象test、source bytes/SHA256、全test ID/成否、import済みproject modulesを保存する。

## 2. 群別結果

| 群 | 件数 | 確認した内容 |
|---|---:|---|
| T1 Entrance | 13 | exact Fraction/int/整数対、normalization、zero/negative/iterator拒否、bool/float/string/denominator、degree/arm、η/ρ/σ/τ/H/K、word/child、missing/unknown precision |
| T2 Identity | 4 | x=0、constructor/angle禁止spy、bit取得0、Iとzeroの区別、σ/identity、stochastic helper拒否 |
| T3 Preflight | 13 | P0〜P5とdimension/exact evidence、P5 pending、H/K不足、R/P3/P4境界、root端点、全parent列挙なし、選択parent、L1/m1/x1、P5 law改変、helper domain |
| T4 Failures | 14 | raw/coin zero、law/source/input/evidence改変、fault-injected guard、constructor/runtime/bit failure、不正trial_id、angle不足、terminal STOP、parent/context/位置保存 |
| T5 Angle | 10 | zero、exact positive/reference enclosure、r=1およびr=2の独立照合、σ±、τ、非法型/範囲、固定root1回、巨大exact証拠roundtrip、context拒否 |
| T6 Event semantics | 9 | 旧eventとの同値、固定accepted bitstream、phase/word/child/σ、packet、snapshot、input/policy取り違え、bound angle、再hashされたevent改変 |

全test IDは[test_catalog.json](test_catalog.json)。r=1/2はr=2用の独立referenceにも用いる。

主人工fixtureはp=(2/5,3/5),x=1/2,m=3、η=1/1000、ρ=1/10^6、K_root=128、H=96、τ=2^-32、K_phi=80、J=48。m=5ではp=(1/6,1/3,1/2),x=2/5と少数の指定parentだけ。境界としてp=(1,),x=1,m=1を使う。全語列挙や乱択acceptance統計はない。

## 3. 実行履歴

| 記録 | 結果 | SOURCEとの対応 |
|---|---|---|
| run01 | 56 PASS、FAIL/ERROR/SKIP0 | 最初のF3実装。実行前AST確認で新testのwith構文を修正。実行失敗はなし |
| run02 | 62 PASS、FAIL/ERROR/SKIP0 | 人工監査で見つけた不正trial_idのserialization失敗リスクを修正し、missing argument・P5改変・context等の6testを追加 |
| run03 | 63 PASS、FAIL/ERROR/SKIP0 | dimensions、bit-draw内訳、失敗parent/child/context保存を追加。最終SOURCEとtestのhashが一致 |
| run04 | 63 PASS、FAIL/ERROR/SKIP0 | 公開用portable runnerのroot解決を静的修正後、そのrunner自身を確認。同じ最終SOURCE/testのhashと63件PASSを記録 |

run01〜04の[raw JSON/log](test_runs/run04.json)を全て保全する。これは科学one-shotのretryではなく、ユーザーが今回許可した**新F3コードの修正・限定人工テスト**の反復である。旧科学claimは作成/消費/再利用していない。故意のfault-injectionによるSTOPがassertどおりに動くことをPASSとし、実際にF1との矛盾が起きたと表現しない。

4回ともPython3.11.1、専用clone、-B/ユーザーsite無効、共有cache/venv設定変更なし。単一processのunittestで共有moduleへのpatchを扱い、不要な競合や起動費用を避けた。CPU/RAM/wall等の人工的capは設定していない。独立なhash/公開照合は並列化する。run04の自己process時間/RSSはJSONの**管理metadata**であり、性能実験やspeedup/科学費用の証拠ではない。

管理検査の初回では、公開用runner自身が`__main__`としてimport inventoryに含まれることを許可集合に入れていなかったため、import-scope検査1項目がFAILした。raw記録を改変せず、固定pathのrunner自身だけを明示許可する管理検査へ修正した。科学packageの混入や人工correctnessの失敗ではない。

## 4. 不足・結論

無改変旧SOURCEの許可人工例で、P条件PASSに反するlocal guardは見つからなかった。63件で全入力・IID・物理provider・一般数学定理を証明したとはしない。canonicalは明示非対応。H/K最適化、統計budget/全N accumulator、DF/PR/RTE、native/回路、分子correctnessは未実施・範囲外。今回新しい非自明な数学保証を必要とする問題は見つからなかった。

新科学計算/保存科学値再解析/分子・DF/量子回路・simulation/compile・synthesis/backend/新authority・claimは全て0。**F3_TECHNICALLY_VERIFIED、mandatory STOP。**
