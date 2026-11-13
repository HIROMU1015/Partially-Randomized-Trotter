# A → B：残10roundの証拠補完基盤

2026-11-13 JST。技術準備と人工試験のみ。**実H4の新mean/reference/normalization/N、保存再解析、decoder simulation、C/Gは0。** [REPORT](REPORT.md)／[PLAN](PLAN.json)／[有限科学scope](SCIENCE_PLAN.md)。

## 現在の成果は再生成しない

B固定`3216d27a69ac8532d2a22e78f5f94491d202d238`のm1受入関数を再利用。A最新`3e035c0d61386010f59ad7ea792bc7a5cefd18ac`の元packetをSOURCEで参照し、m0/m1/m6 nominal64/60/38を保全する。今回新N packetは作っていない。

正本configuration ID：`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。M12/exactκ/H4 DF10/元H-state-snapshot/sector36/S2/prefix1/Δ.1/K2/h.1/prior/decoder/law/profile/P0/minkowski/finite_Bを変更しない。PLANの残10rowはfull canonical identity/law、physical ID、T/qPF/R/event、保存mean/ref/SOURCE、nominalと十分Nのnullを含む。

## 新しい実行経路（未実行）

- `h4_pr_rmse_completion.plan()`：metadataだけ。default CLI `scripts/run_h4_pr_rmse_completion.py --out <fresh-PLAN.json>`。科学leafなし。
- **S1 m3/m4**：一致保存mean/対照/logBを使い、新reference pair2時点/4 actionで補完する。観測norm欠測はnullのまま、元SOURCE gateを条件付きempirical診断へ使う提案。旧block backendとR-aware式の違いを記録し、過去u/marginを下げない。
- **S2 m2/m5/m7–12**：新spectral-power mean/対照8候補、保存8reference pair再利用、新reference0。m2だけdirect actionでも1候補照合。新nominalは10候補×X/Yまで。m0/m1/m6はcopyのみ。
- 構成は元`h46_long_time_signal.evolve_batch`/finite-K law/polynomial、既存loader/reference/action/scalar/geometry/A evaluatorを再利用。旧Hoeffding N solver、価格、B controller/plannerを実装し直していない。

実H4では未計算。人工非可換/負時間・大event形状で経路を確認しただけで、長時間精度・速度・global RMSEを認定していない。S1のSOURCE guard根拠を使えなければSTOPし、同候補の追加mean取得は別scope。失敗時のfallback/retry・R/K/M/閾値変更はない。

## 将来の新packet契約

今回のbranchの科学SOURCEが固定・承認された後だけ、`--execute --stage S1_S2 --source <commit> --out <fresh-run-directory>`でまとめて実行できる。途中のmean/対照/ref、局所diagnostic、numerical/source norm証拠、allowance、nominal recordを保存する。

既存B `n_from_a`へ渡してcanonical identity/lawを保持する新scope packetを作る。元m1 packetを再作成・旧ID再ラベルする方式ではない。

- `previous_packet`：A3e035c0dの元N_HANDOFF locator。
- `diagnostic_scope`：`newly_evaluated_rounds`、`saved_mean_rounds`（m3/m4）、`new_mean_rounds`（8候補）、`preserved_rounds=[0,1,6]`、`whole_schedule_RMSE_qualified=false`。
- 各row：canonical identity/law/source_request_id/physical source_candidate_id、nominal/model_statusと`diagnostic_source`。旧3roundは元SOURCE、新roundは今回SOURCE、未評価はnull。
- 元result/packet/model recordは不変。十分N_X/Y・global RMSE boundはnull、status/model/global/finite-KはUNQUALIFIEDのまま。

B3216d27aのm1 validatorは新10round scope専用ではない。**B担当がこの新scopeの受入条件を拡張すること**。Aは旧validatorへ偽装しない。共通C recordやG会計は変更しない。科学call候補数とX/Y軸数を重複jobへ展開しない。

## S3への不足と次の担当

全13raw mean/nominalが揃っても、energy RMSE/十分Nは未資格。未解決zero/tie trialの会計、prior/alias、保存state/元DF-H target、finite-K/PF biasとMinkowski分解、ideal/actual sampling law、MSE精度と検証信頼度・反復数が必要。S3の算出式と未決定事項はSCIENCE_PLAN参照。旧α/β/Nを新保証へ流用しない。

**ユーザー**：S1＋S2の有限scopeとSOURCE norm条件を一括承認するか判断。**A**：承認後にだけ実行し、新scope packet/全局所証拠を公開。**B**：新scope受入の技術接続。S3の科学判断・統計実行承認は別段階。今回commit/push・remote照合後mandatory STOP。
