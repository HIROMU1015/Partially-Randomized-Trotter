# H6 B0+I：現在profileへの準備hash結合と不一致診断の修正

2026-10-15。利用者の「修正を行って／サブエージェントを使わないで」に従い、担当Codex単独で調査・実装・検査した。**修正済み、82限定人工tests PASS。本計算の再起動・新grant発行は0。**

- Branch: `track-a-h6-b0i-blueprint-fix-20261015`
- 修正SOURCE: `6b4d3026bf6c30bea3d74e7ccaeb06bda0e9baaf`
- 起点STOP監査: `1453ceebe887fbc3040b19e00059e7a4d6766015`、失敗したSOURCE `7300b39df4c53d854d5bc8934430af19fbdb0085`
- [資料・未認可契約・検証入口](../../artifacts/resource_applicability/track_a_h6_b0i_blueprint_fix_v1/2026-10-15/README.md)
- [run01の停止事実と消費済みone-shot](track_a_h6_native_b0i_pilot6_run01_20261015.md)

## 原因と確認範囲

run01はPhase Aで `B0_BLUEPRINT_FINGERPRINT_MISMATCH`、native exit2で停止した。消費済みgrant/outputを保持し、新しいpilotを実行せず原因を限定再現した。

固定された旧blueprint自身のdigestと元prepared metadata原本SHAは正しかった。保存snapshotを一度読み、既存 `_prepare_discard` でp5q1/p10q2/p15q8の準備情報だけを構成する診断を行ったところ、**全3構成でblueprint内の `preparation_hash` だけが異なった**。constant、extracted tail、fragment/block順、mode、prefix/q/T/formula、H/DF/state/input/compiler bindingは一致した。

| 構成 | 旧準備hashの先頭 | 現profileの準備hash先頭 | 旧blueprint digest先頭 | 現blueprint digest先頭 |
|---|---|---|---|---|
| p5q1 | ddef30ec | 96fd0b87 | e955c9c6 | 22d1f68e |
| p10q2 | 6537970b | 8c063113 | c94ddeba | 614f00dd |
| p15q8 | caac6fed | 0f65152f | 1761bf62 | 99b01ca1 |

`preparation_hash` は数値basis operations・局所matrix fingerprint・対角値・partition/tail等を含む実装依存のhash。旧保存値が現在の固定environmentでbit同一に再構成されることは、従来のprofileでも未検証だった。それにもかかわらず旧値を現在runtimeの必須期待値として使った結合が停止の直接原因である。

旧signal workerはsource `54986a9f54515c8c2289e38fa16fd81e2b6008c2`・thread4で、現在profileはthread1。関連科学準備sourceはGit上同一だった。旧thread4条件で3構成だけを比較再現しても現在thread1と同じhashとなり、旧値は再現しなかった。**thread数が原因とは結論しない。** p15の内部partition hashも旧値と異なる。旧結果には完全なpreparation payloadや当時の全binary環境が保存されていないため、旧eigensolver/basisのどの数値が異なるかまでは特定できない。旧bit同一性・物理的同一性をこの診断で証明したとは扱わない。

診断はsnapshot load計2、B0 preparation計6。最初の診断専用import制限が既存packageのPySCF bootstrap importを止めた試行はsnapshot0/準備0で別保存し、その費用の未記録分をUNKNOWNとして保持した。通常package importへ合わせた後もSCFを呼んでいない。CuPy/GPU、新SCF/DF/state、signal指数作用、overlay、native evolution build、probe、compileは0。thread4は旧条件比較だけで、本番thread1は維持する。成功した診断2回のwall合計約8.39秒、RSS最大約375.7MiB。共有設定・venv・他job変更なし。

## 修正

[準備identity module](../../scripts/h6_b0i_preparation_binding_v1.py)を追加し、[adapter](../../scripts/h6_b0i_pilot_adapter_v1.py)と[runner](../../scripts/h6_b0i_pilot_runner_v1.py)へ接続した。

1. 現在の固定profile・input5件・科学provider closureへ、3件の準備blueprintを明示結合する。[固定metadata](../../artifacts/resource_applicability/track_a_h6_b0i_blueprint_fix_v1/2026-10-15/current_profile_preparation_identity_v1.json)自体を新SOURCEに含め、Git blob/bytes/SHAで固定する。2回の限定診断が同じ現在値となったことを証跡に残す。
2. 旧blueprint/hashを別欄に残す。旧との比較は `preparation_hash` 以外の全field一致を必須とする。入力/H/DF/state/compilerや候補等が変わったものへ流用しない。
3. 将来の実行前にmetadataのbytes/SHA・SOURCE closure、profile digest、input receipt digest、provider SOURCE/closure、全3caseと現在期待digestを確認する。実行時は完全blueprintを固定現在値へ厳密照合する。**予期しない実値で期待hashを自動更新する経路はない。**
4. 不一致時には期待digest・実digest・期待/実field差分を `preparation_mismatch_v1.json` へ保存し、overlay/build/compile前STOP。証拠write自体が失敗した場合もfirst STOPを保持し、write errorをterminalへ別記録する。
5. 合格caseのscalar/signal receiptにも現在/旧の準備identityを保存する。旧数学bridge/shot/u規則と科学providerは変更しない。

新契約案の `binding.b0_fingerprints` は現在profileの固定3件を参照する。この変更は旧bit一致の認定ではない。保存H/DF/state、prefix/fragment順、signed full-Fock scalar、full-H reference不回転、compiler options、empirical allowanceと誤差gateは保持する。将来pilotのPhase Bで実native/control/X-Yと保存corrected/独立forward signalの照合が必要で、全3caseが合格するまでcompile0。今回その実物理correctness・compiled費用は未検証。

## 検証・固定

単一test process/thread1・AS512MiB/CPU120秒/wall180秒の計画内で **新12＋影響する既存70＝82件PASS**。wall約1.35秒、peak RSS32MiB。サブエージェントは使用せず、既存の独立担当作成testsも今回の担当が実行した。今回の独立レビューを行ったとは表示しない。

新testsは現在値合格/旧digest保持、不明なpreparation hashの詳細拒否、元bridgeでの合格、旧期待値の誤使用拒否、profile/input/provider/closure・metadata tamper拒否、科学field変更/候補追加/path escape拒否、未認可loader拒否、diagnostic IO失敗とfirst STOP保持を検査した。人工testsでは科学import・実NPZ・実fork・compile0。旧70/63/75の結果と失敗履歴を保持した。

SOURCEにはruntime変更3ファイル、新tests/harnessと固定identity metadataの6ファイルを明示commit。新runtime code11＋identity metadata1をclosureへ固定する。既存profile34,273 filesを終了後read-only SHA照合し差分0、run01の全runtime file receipts、旧4主要worktreeのHEAD/clean状態も不変だった。

[修正契約案](../../artifacts/resource_applicability/track_a_h6_b0i_blueprint_fix_v1/2026-10-15/h6_pilot_repaired_contract_draft_v1.json)は未認可・未seal。最大6worker・thread1・合計6compile、per AS32GiB/RSS9GiB、stage wall/総CPU4h、admission240.5GiB等を保持。CPU[]、output/control未指定、command null、approved/runtime false、automatic_retry/next_stage false。run01のsnapshot1/overlay1予約・wall8.597542秒・compile0と、今回診断のsnapshot2/preparation6/失敗bootstrap UNKNOWNをcarryへ残し、返却/resetしない。新grantは発行しない。

## 次の実行条件

本計算は再起動していない。次回は修正SOURCEと現在profile identity参照を使う新しいone-shot認可、直前SOURCE/input/profile/carry・CPU/memory/PSI/OOM/fs/inode/quota・未使用output/control/marker結合が必要。旧消費済みgrant/outputを使い回さない。実native Phase Bが不合格ならcompile前STOP、旧hashへfallbackしない。実H6成功・速度・費用は未取得のまま。軽量資料をcommit/non-force push試行後に終了する。
