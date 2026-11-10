# Codex B/CへのHANDOFF：詳細N候補計画

2026-11-11 JST。[REPORT](REPORT.md)／[EXECUTION_PLAN](EXECUTION_PLAN.json)。今回は全てmetadataと人工/保存fixture検査で、新signal・保存数値再解析・N/C/Gは未実行。科学許可は全packageでfalse、実行SOURCEはnull。A0の承認を詳細Nへ拡張しない。

## Bが受け取る範囲

A0全24 recordを `rows[].A0_record` に無変更で保持。18件は有限科学案、6件は保留で、FAIL/支配関係ではない。元86候補や上側未探索範囲を削除せず、この計画をglobal最適Rの採択に使わない。

`N_request`は固定B 7bca8f964のbuild_menu/request_contextと、統合B ac3e1ce87のLAWから作ったもの。完全identityは元A0 requestと等しい。**A0元law宣言は原recordへ残し、N用lawへaliasしていない。** 将来のproducerが指定SOURCEとfinite mean lawの一致を確認して初めてN用law_bindingを付ける。sampler実装の資格を宣言だけで追加しない。

受渡しは既存 `identity/plan/actual/diagnostics/source/origin` 共通N record、さらにB用 `law_binding` と `N_comparison_policy=R_AWARE_EMPIRICAL_ACTION_COUNT_V1`。既存RMenuBatchのexportを再利用し、B optimizer/C schemaを変更しない。

- identity：candidate IDだけでなく、H/state/snapshot/rank/sector、m/qRPE/qPF/T/δ、formula/prefix/K/R、11piece順/allocation/event/scalar規則、RPE configuration、compiler/control/full measurement wrapper/P0まで照合する。
- plan：条件付きstatus、N_X/N_YのPython int、axesのdisk/radius/beta/epsilon/alpha。巨大Nの十進文字列もdiagnosticsへ保持し、binary64へ変換しない。
- diagnostics：reference2経路とSOURCE、independent corrected/raw/exact-tail signal、bias、B/logB、normalization/closure/norm/allowance、整数確認、検査status。phase/radius未資格はN=nullと理由を残す。
- actual：未知implementation/law/厳密ground誤差はnull。ideal conditional PASSから全RPE成功やactual Nを作らない。
- source：将来の固定実行SHA、使用保存reference/signalのpin、科学input hash。承認証明書/追加grantの発行は不要。

H4旧巨大Nの正本は89c91c692の `docs/research/h4_df10_rpe_batch_results_20261110/N_HANDOFF.json.shots`。PLANへ在庫をcopyしたが、その旧policy recordは新R比較用のNではない。同policy再解析recordを別SOURCEで保存し、旧handoff・rawを上書きしない。

## 保存再利用と小さい結線の残作業

H4 m0/6/12、H6 m8/12は指定保存結果からbaseline signalと2経路referenceを読める。PLANのfield locatorとidentityを使い、科学実行時に数値gateを再度検査する。今回行ったのはschema/identityの照合で、pair差や新Nの再解析ではない。

H6 m4は確認したD1証拠に独立pair/exact normが不足するため、P1には新2経路referenceとbaseline R16の新mean/内部対照1を明記した。callback未指定による暗黙新referenceに頼らない。その他のTで保存pairが使えなければSTOPし、無断fallbackしない。

既存N cacheに旧baselineがあるとrun_missingはその値を再利用するため、同policy比較用baselineだけは明示的なscalar再解析queueへ分けて新recordを供給する。saved FAILを単にcacheから除いて再実行しない。baseline不足normを0補完せず、PLANに明記したH6 m4だけを新伝播する。必要adapterは既存 `pr_r_dependent_n.analyze` とrecord envelopeを使う数十行程度の結線とし、新RPE solverを作らない。

新RはB RMenuBatch→A RBatchで実行できる。Bはcontext/referenceを共有するが、RBatchがgroupごとにhalf-stepを再作成する点に注意する。cacheの小改善や大R power backendは提案段階で、今回コードに導入していない。V1の中間norm/演算順の根拠を保持できない高速化はそのまま採用しない。

計画順は各packageのbaseline同policy処理を先にし、FAILなら後続を未実行として止める。P1/P2/P3間は自動遷移なし。価格/G providerは今回のN案では呼ばない。

## C domainの受渡し

現pinでA0に一致するC Phase1セルはH6 m4/R16（保存calibration）とR32（新calibration予定）だけ。他22件に価格を補完しない。C元8セルとroles/cohortをPLANにcopyしてあり、phase1_cellはexact candidate一致のときだけ付く。小qモデルを大q/Rの適格価格としない。

Cが将来返すrecordは既存full candidate/domain、binding/compiler/axis、sampling-law context、cohort/expectation分類、SOURCE、shared calibration/paired XYの不確かさを保持する。N候補のqはqPF、Rはwhole-outer、H6 event数はqPF Rで、各pieceへRを重複させない。整数tieで非対称になるallocationも原本のまま。

C Phase1はH4 qPF8/16×R4/8、H6 qPF1/2×R16/32の有限点。H4 m6/m12のR128〜8192やH6 m8/m12のR128〜2048には新しいR/q検証が必要になり得る。全law期待価格、rare寄与、長q systematic、cross-R/round covariance、total CI/選択後保証は未知のまま残す。A0のshot係数やBをCへ再乗算しない。

## 次の担当

**ユーザー**：先行P1（12候補、mean/内部対照9、保存baseline scalar3、新reference2 action、全24軸のN）の実行可否を判断。P2/P3は別の未承認案。

**Codex A**：許可後、既存SOURCEを使う小さい結線を固定してP1だけを実行・公開。**Codex B**：指定公開SHAのNを同policyで取り込み、C不足・FAIL/未資格を保持。**Codex C**：自身の許可範囲で価格検証を進める。現時点で実G/R選択は要求しない。

今回の公開・remote SHA/tree/成果物照合後mandatory STOP。新科学実行や詳細科学レビューへ進まない。
