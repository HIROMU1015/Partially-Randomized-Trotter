# 新A：N5-B3後のprecision・domain・word policy改善調査

2026-11-08 JST、SERVER 2。基準 **2886ac07c1ec706a1180d107e0111745e74aec1f**。Branch `codex/pr-reference-post-n5b3-policy-20261108`。

**承認範囲の静的調査・技術設計・最小実行準備を完了しmandatory STOP。** 一様gate capを全keyの必須precisionとする必要はなく、role配分・同じwordの鋭い保証・保存record再選択を先に調べる余地がある。Dyadic細分化、signed digit、部分的追加合成、複数のdomain構成も残る。新policy・保証方式を採択せず、総T改善率やm17成功を認定していない。

本指示のN5-B3は、基準commit内の`docs/research/pr_reference_n5b2_numeric_20261108/REPORT.md`と対応RESULTSを指す。旧path・成果・STOPを改名／改変していない。

固定背景はLinear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・準備費除外、Partial S2 LD3 h=.02 finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、compiler A/Qiskit1.3.0。既存B′／Reference構成とPF scheduleを保ち、m17 q131072／r32／88 bits、626／630 shots/axis、alpha1/720、failure.05、phase cap.4、energy1.5936e-4 Haを変更しない。

## 1. 基準結果の正確な範囲

[基準REPORT](../pr_reference_n5b2_numeric_20261108/REPORT.md)と[RESULTS](../../../artifacts/pr_reference_n5b2_numeric/2026-11-08/RESULTS.json)の採用判定は、**Q+Gの全8 axis比較がcap外、全4角でQ>G**である。QとGの値は保存されているが、各項単独のcap比較は旧scopeで実行していない。今回その新規比較も実行しておらず、「QとGをそれぞれ新たにcap判定した」という結果は追加しない。

この負結果は、現b36のadditive十分証明の不足を示す。実word誤差の下界、RPE失敗、新規合成の必然性ではない。正確なsaved gate capの導出をSOURCEで追い、固定Nを保持するjoint予算を総worst-case RZ数で一様に割った値だと確認した。

## 2. 今回具体化した候補

| 優先して調べるもの | 改善の因果経路 | まだ判定できないこと |
|---|---|---|
| P0：Role/key別の誤差予算 | 少数の中央／scalarへ一様capより広い割当を与え、他roleと合わせて保存Nを守る。Word不変ならTも不変 | 全role和と全supportの制約が閉じるか。期待多重度で誤差を緩めない |
| P1：保存delta/Gによる上下囲い、必要時だけdirect guard | 合成せず、現wordの誤差を逆三角不等式でも調べる。単なる上界の保守性と真の近似不足を分ける | 局所capへの適合、joint配分への適合は別。未数値評価 |
| P2：保存record再選択 | Literal一致のguard／実T候補を有限catalogで比較し、不要な新規合成を避ける | 最も精密なrecordが最も安いとは限らず、現C0優先を変えるなら新policyとして明示が必要 |
| P3/P4：細かいdyadic＋signed digit | Qを減らす一方、chunk数・guard和・Tの増加を抑える候補 | b・基本precision・carry域・missing wordと価格は未選択 |
| P5：必要keyだけのstrict合成 | Budgetを先に固定し、中央4角等の不足部分だけを直接近似 | 取得成功、実T、全domainへの供給は未取得。全面再合成を既定案にしない |
| D1/D2/D3：Domain構成 | 既存最大basic反復、有限指数拡張、strict補償付き周期表現で域外rejectの解消を設計 | 反復／追加wordのguard・T、周期差のrelative誤差、全support適合は未資格 |

[POLICY_OPTIONS](POLICY_OPTIONS.md)にSOURCE差分、数式、必要証拠、Tへの作用を比較した。多くは既知の角度分解・誤差配分・norm評価の組合せであり、未検証であることを新規性の根拠にしていない。

## 3. 設計上解消できた点と残る障害

**解消できたのは、改善を試す前の論点・方法・実行境界の曖昧さである。**

- 保存capが唯一のprecision条件ではなく、rolewise joint certificateという代替があるとSOURCEで対応させた。
- 同じwordの厳密誤差を調べる方法として、保存delta／Gだけによる上下囲いを導出した。新規合成や新guardを先に要求する必要はない。
- Domainを大きくする方法として、追加basicなしの有限反復構成も明示した。現providerのL定数を変えるだけでは足りず、multiplicity・guard・Tの証明が必要とした。
- 全18roundの比較式を、shot差と強baselineを保持して明示した。局所精度が良くなっても総Tが減らなければ採用根拠にならない。

**科学的障害は未解消。** b36のQ／保存G、m17の全roleへのlogical資格、B′固定scalarの実域外reject、Reference vacuumの包絡超過、event安全supersetと到達性、Gの本番terminal条件、完成policyの全域word供給、各roundのRPEと総T価格が残る。

Vacuum包絡が現Lを超えることは、到達角が必ず域外である証明ではない。B′scalarは保存実literal自体が域外であり、random非到達で除外できない。Eventの域外candidateを到達済みとも確率0とも扱わない。

## 4. 既存コードで対応できる範囲

- **誤差会計**：CT-0 geometry／fixed_shot_budget／worst_counts、N4-CのX/Y max・正確なinterval端点。旧SOURCE／compiler diskを同じraw単位で一度だけ足す規約を維持。
- **Word評価**：保存C0299、43 basic、85 certificate、旧strict inventory、exact_word_matrix／strict_guard、actual adjoint。新guardは今回未実行。
- **配分**：既存C2 strong-baselineのMILP候補とFraction検算は参考になる。2精度専用offer生成や旧Full Y4条件・alpha下界は転用不可。
- **Provider／consumer**：exact整数丸め、固定word順、negative whole inverse、named emission、inplace consumerは再利用候補。b／域／signed recodingを変える場合は必要な差分と保証を独立に資格する。

新しいPF、Hamiltonian、sampling law、状態準備、fusion、旧C2／C-S／SK研究は開始していない。候補Gでpure common scalarを分離しても、relative角・logical guard・量子化・旧diskを消去しない。

## 5. 次の最小実行はV0-A/B

[NEXT_VALIDATION_PLAN](NEXT_VALIDATION_PLAN.md)は、**保存4中央角の上下囲いと2 armの中央role予算probe**を、固定入力・数値精度・call数・判定・STOPまで具体化した。新規合成、guard、word生成、provider、PF build/compile、RPE／shot再計算、T集計は不要な設計である。ただし新しい科学数値再解析であり、明示承認までは実行しない。

[実行準備済みSOURCE](../../../scripts/diagnose_pr_reference_post_n5b3_bounds.py)を追加した。既存I／endpointsだけをASTで再利用する小wrapperで、**今回の確認はsyntaxと静的参照だけ**。本番runtimeや数値結果は未検証である。科学入力baseは2886ac07c…に固定し、negative/indexの過去検算は繰り返さない。

その先のV1–V3は、有限b診断・domain要求、保存recordと実離散Tによるstrong baseline、必要部分のguard／合成という条件分岐に分けた。未決の予算やpolicyを推測で正式採択せず、それぞれの固定scopeと承認が必要である。

## 6. 全18round総Tの評価条件

最終目的は同じエネルギー精度・成功確率・準備費除外のPR＋RPE総T削減である。M17はB′626／Reference630 shots/axisを維持し、共通frameの1-shot取消しを総費用差へ流用しない。全roundの各保存NとPF／r／bits／誤差余地を使う。

Strong B′にも同じrecord候補・domain・精度配分・価格取得機会・保証改善を与え、実行可能な費用同士を比較する。旧policy比だけの改善、LP下界、未取得価格の0埋めでは優越を認定しない。未資格roundを含む総Tや改善率は今回計算していない。

## 7. 実施と保全

新たな科学数値計算、保存値の新規再解析、strict guard、合成、回路build/compile、RPE再計算、T価格取得・集計は**0**。旧検算／160 fixture／科学runnerは再実行していない。独立したSOURCE読取は並行し、共有設定・他ユーザー・他Trackへ変更しなかった。CPU/RAMの独自capも設けていない。

[SOURCE_AND_STOP](../../../artifacts/pr_reference_post_n5b3_policy/2026-11-08/SOURCE_AND_STOP.json)に固定参照・保存判定の引用・未実行SOURCE hash・静的check・STOPをまとめた。新しい承認証明／grant／管理基盤は作っていない。正式共通指示2026-10-25、親／worktree AGENTS、PROJECT_MAPと既存研究資料を適用。著者Codexの静的技術調査であり、正式な新規性判定・独立科学レビュー・保証採択ではない。

旧SOURCE／raw／result／grant／ledger／marker／STOPを保全。**Mandatory STOP。次担当はGPT／ユーザー**：複数候補を残したまま、最初のV0-A/Bを実行承認するか、保留するか判断する。新precision／domain／policyや追加合成へ自動進行しない。
