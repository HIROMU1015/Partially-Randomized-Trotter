# PR-DF固定gridのdecoder-only bridgeと5方式readiness

2026-11-15 JST。既存B branch `codex/rpe-total-dp-acceleration-20261114`を継続した。基点999304918、共通運用指示11/14版、ロードマップ固定fa84c0d6bを参照。新しい実H4 MSE/DP/N探索、保存科学値の数値再解析、価格・G計算は0。確認は主agentが行い、独立科学レビューではない。

## 技術成果

保存PR-DF LD1/LD2の各13roundを、旧`A_FIXED_GRID_510FBACD_V1`を保った版付きtotal契約へSOURCE付きで接続した。generic ceil-gridやPauli辞書・κ・Rへ変換していない。

| 構成 | 元canonical ID | 新opt-in total canonical ID |
|---|---|---|
| PR-DF LD1 | `a4dda8f7e3fc07666ca2aabf634917e8bced14eb62151eb629ef23c0cce21d3d` | `9c3d392ec5c5ad373aaa1d96a8ba0828ddd268e6fef35da229029beec0229e8b` |
| PR-DF LD2 | `1fa33556c17303f51ba03c4eae11189d35e59b3f57faad0e69d9fd08bf9245ae` | `e6fe7e6845dd5eae8829f54b8bbae655ccf1c81c320c4d0ee1bfed0399bb5583` |

新しいviewの版は`A_FIXED_GRID_510FBACD_TOTAL_DECODER_VIEW_V1`。新旧のcanonical candidate IDは別で、元physical candidate IDは全round不変。[CANONICAL_MAPPING](CANONICAL_MAPPING.json)にSOURCE、元κ、T/q/R/実幅、signed-piece allocation、event数とCの各軸coverageを保存した。

## 既存SOURCEの再利用と変更箇所

- A714247619にあるPR-DF bundleを、元a9abf8deのconfiguration、fefd548eの数値実装、16210932のraw carrierへ既存`rpe_signal_statistics_caller.accept_saved/bind_bundle`で照合した。信号・normalizer・bias・allowanceを再計算していない。
- `rpe_total_binding`は元specを`decoder_origin_spec`に保全し、**無変更の**`pr_rpe_rmse_integration.generate_from_a_spec`で元round列を生成する。既存Bの`_assemble`で新identityだけを生成する。
- controllerには新版へのdispatchと再生成検査だけを追加した。不明なadapterや外部mapperは拒否する。元A adapterの厳密grid検査、ceil-grid、会計、選択器、identity算式は変更していない。
- `rpe_c_interfaces`にはAの既存`pr_rpe_signal_plan.plan`と`numerical_recipe`だけを固定fefd548eから読み込んだ。C c628e9e8のplanner・provider checker・price guardを再利用し、信号producerは読み込んでいない。
- total decoder/Q3 kernel710aa69b、旧partial risk・R3・旧A adapter・旧DP runnerはbyte不変。旧科学結果・STOPも不変。controller自体は新版dispatchが加わったため、以前の「controller全体が旧版とbyte一致」という技術検査を、旧A adapter不変と旧configuration再生成一致の検査へ更新した。

## 人工・回帰検証

新6＋既存Pauli8＋baseline7＋旧A/B/C統合12＝**33 test method PASS**。[TESTS](TESTS.txt)。人工13roundでphysical context・raw/normalization/target/lawを保ち、新decoderのzero保持/tie argmin、旧kernelへの明示契約委譲、科学N/Gへの非昇格を確認した。保存metadataのSOURCE取り違え、LD混在、不明grid、外部mapper、物理ID変更を拒否した。

価格のID bridgeは人工価格で、元physical ID・元receiptを保全することを確認した。実価格の新受入や平均価格資格は発行していない。旧partial DP結果をtotal MSEとして転用していない。

## 5方式と価格候補の対応

[READINESS.md](READINESS.md)／[機械可読14構成](READINESS.json)は、PR-DF2・Pauli2・Full PF基準1＋shortlist3・Truncated基準1＋shortlist3・Full-H κ2/R448を別の物理候補として扱う。

PR-DFは今回total metadata接続済みだがtotal MSE未評価。Pauliの限定Q3はcaf1417の2件だけが完了済みで、再計算していない。Full-H κ2/p7の既存metadata接続は保全し、R448も独立した構成としてmetadata接続を確認した。全方式で十分N/global RMSE/G_qualifiedは未資格/null。

C85557306のPR-DF52軸coverageについて、元physical ID・q/Rを各13roundへmetadata照合した。Pauli LD1 m0とFull-H κ2 m3は元の保存sidecarの物理contextがsignal側と一致した。一方、Truncated p7/S2/幅hの全13rawは、価格shortlistのS2/16h/7・Y4/16h/3・M5/16hとは別scheduleである。Full PFも全round rawを同定するまで価格shortlistをrisk適格にしない。価格値の平均・fit・Gを計算していない。

## ロードマップへの影響と引渡し

固定fa84c0d6bの§4.3、§5のPR-DF行、§7・§10の「旧gridのdecoder-only bridge不足」は**技術的に解消**した。§8の「信号→推定器接続」の入力準備が進んだだけで、MSE評価・必要N・Gの達成ではない。

統合担当向けの更新案は「PR-DF各13保存signalのZ1＋T1 metadata接続PASS、total MSEは未評価、旧partial DPは別結果」である。新証拠は本REPORT・実装・33技術test・SOURCE付き対応表。次の依存は**ユーザー**による推定量/target/loss/model閾値/law/数値scope/N資格水準の判断、続いて明示承認された有限科学評価である。C/Dの価格側を待つ必要はないが、正式Gには同一候補の適格NとE[C]が必要。

公開された07a6e0095のロードマップ更新規約に従い、個別担当のBはロードマップ本体や統合担当branchを直接編集しない。この節を統合担当への影響報告とする。A/CへのAPI・不足条件は[HANDOFF](HANDOFF.md)。新しい科学計算の依頼・起動はしていない。

## 公開coverageと残る判断

報告前のA公開ref/coverage確認範囲は[READINESS](READINESS.json)の`A_coverage_publication_check`を参照。新しい全round coverageを同定できない項目は未確認のままとし、「過去に未実施」と断定しない。Aが後で公開するSOURCE付きpacketを同じintakeへ結合する。保存値の生成し直しは不要。

新canonicalを作ったことはZ1＋T1の主推定量採択でも科学的精度保証でもない。ground/target、実sampler law移送、数値certificate、元Minkowskiとの統合は未資格。raw直接riskへAのδEを二重加算しない。`sufficient_N/global_RMSE_bound/G/G_qualified/winner=null`を維持する。

成果のcommit/push・remote照合後は**mandatory STOP**。新しいMSE/DP、shot探索、価格取得、Gへ進まない。
