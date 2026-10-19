# 共通実験プロトコルv2：H2完了・結果受入の追記案

2026-10-19（Asia/Tokyo）。**DRAFT_H2_STATUS_AMENDMENT_NOT_ADOPTED_BY_THIS_TASK**。

本案は[固定v2草案 c0808b25](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c0808b254aa02a94c7482f5c6e353b54532a54f8/docs/research/track_a_pr_resource_evaluation_protocol_v2_DRAFT.md)への追加案であり、後に別branchで[文書採択されたv2 1f001cfc](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1f001cfc4e8e0531279a26613be29a52c7c0640c/docs/research/track_a_pr_resource_evaluation_protocol_v2.md)のH2 snapshot更新にも使用できる。両固定版の内容・過去の承認状態を上書きしない。v1 `9036d8a87ad1c38e83b9ff654b412dde7041e9d3` は変更しない。**H2は旧v1対象外の追加系列**である。

既存v2はH2をプロトコル対象へ含めているが、H2の情報はS_STATS `829d697a` 時点でP未seal・未実行。本案は最新P `f2d1a86c431f835db700f348ce618d8604e7ee23` とユーザーの有限標本結果の正式受入を追加する。既存採択版の撤回・自動再採択・遡及的な科学契約変更は行わない。本案の正式採択は別途判断する。

## §2 入力条件：H2追記

linear H2（座標 `(0,0,0)` と `(0,0,1)` Å）、STO-3G、charge0・multiplicity1、RHF。SCF conv_tol1e-9/max_cycle50、空間symmetryなし。OpenFermion DF threshold1e-8、final_rank指定なし、signed weights・生成順・cutoff0を保持し、実rank3。4 qubits、Nα=Nβ=1のspin sector次元4。

F SOURCE `37b1cf72990885962d8449d8bd5ec7e303329350` のdense `scipy.linalg.eigh` を使用し、最大sector振幅を実正に位相固定。norm gate1e-12、残差gate1e-10、最低gap gate1e-12、保存後の再正規化なし。保存残差は `2.3625993563384045e-16`。工学的検査のPASSはground-state certificateではない。

Hamiltonian `d0bc260db6fc0a485fe518050075c18ccc0dff532253dcc719a52a61f41afa80`、state `080b7ad2a46f38ed51e2989eb399ceda4cae95fca688722959ee65463cf174bf`、snapshot `62c6481bd1f8b05ec3b399b3a5293873796737866409049d808e8fe432dcacaa`。旧H2の設定選択rank3/sector6と同一視しない。

H4 tol-only rank10のdense sector36、H6 rank19/H8 rank28の記録上eigsh sector400/4900とはsolver・正規化・残差gateの例外が残る。共通DF政策から全入力生成手順の完全同一性を推論しない。既存v2のH4/H6/H8条件・限定はそのまま継承する。

## §3–4 科学条件とstage到達点：現在欄の更新案

T=.8/P=0、D/Aはε=.05/.01/.005/.001/.0005、通常signed/allowance1、full cosine/sine Hadamard wrapper、symmetric_directional、original scalar＋tail identity一回、reference不回転。Qiskit1.3.0、rz/sx/x/cx、opt1、seed_transpiler17、backend/coupling/layout/routing未指定。compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。

F/FP/D/A/S_STATS/Pは全て固定scopeで完了。科学SOURCE・結果commit・入力・実行契約・終了監査の対応は[stage_evidence.json](../../artifacts/resource_applicability/track_a_h2_results_integration/2026-10-19/stage_evidence.json)。FP診断各n1、A66各新n8、独立P n80/12は分離する。DはD70、B0/B0I p0/1/2・full B1 p3、S2/S4・q1/2/4/8/16。Aは元のA66非直積scheduleを変更しない。候補の退化・重複扱いも元契約を保持し、全域最適とは呼ばない。

旧D_run01の未確認STOPと部分compileは保全し、受入結果はD_run02 `7b202a1a` に限定する。S_STATSは保存summaryに対する既存24 scenario・12感度・2方向の設計であり、今回再計算しない。

## §5–6 独立確認P：H2追加系列

|ε|PR / PF|L_D / δ|n|保存G_PR / paired SE|保存G_PF|点方向|
|---|---|---|---|---|---|---|
|.05|B2_p1_q1_2nd_r2_K4 / B1_p3_q1_2nd|1 / .8|80|384135011/80 / 15992.018778605378|5533984|PR低費用、REPRODUCED|
|.0005|B2_p1_q4_2nd_r2_K4 / B1_p3_q4_2nd|1 / .2|12|998482376201/4 / 276760628.92856276|305695076994|PR低費用、REPRODUCED|

n80/12はS_STATS提案後にユーザーが採用し、最終seal `6eacd7f82afb1c6d3fa6b66987df8dcfa1ed8bcb`、科学SOURCE `ce1f511c4f7429190adcee58cde86be8a87a2fb1` に結合したPを別途承認・one-shot完了した。固定予算92 trajectory/184 compiled軸/256 RTE event、追加sampling・自動retry・候補再選択なし。PFは保存D対照を再利用し、新PF compile・shot再計算なし。

seed plan digest `e6a19d26f531d4a16ec8e5dcab0374fcc306d609f5399046dbca7c80a444ad77`、spec digest `43991ca79559bd1957fd8588e63c0b78bcf6b8f8faa9ef02519c6187279bc168`。保存seed receiptの既知旧seed除外・衝突検査を引用し、本案でseed列の生成・再抽選・新解析はしない。seed非衝突だけで統計的独立性を証明したとは扱わない。

native exit0、同一親wait4、92 worker全回収、保存終了監査COMPLETE_MANDATORY_STOP。正式受入は今回のユーザー指示による**有限標本2条件**に限定する。母平均の点順位保証・実power認定へ昇格させない。

## §7–8 限定と論文scope

q1はS0_ONLY79/S2_SINGLE1、q4はS0_ONLY12。K4の未観測高次・多重eventの費用・分散、law確率はunknownを維持し、0で置換しない。保存SEはtrajectory費用のSEで、量子測定shotのSEではない。経験的bias allowanceを厳密bias保証としない。

H4/H6/H8主表6条件と原図34ファイルは変更しない。H2追加2条件は別の比較資料とし、H2の本文掲載を先取りしない。H4非再現、H6粗精度PF低費用、H8の公開詳細PR監査不足は継承する。H8主表とFigure2B/Mechanismの同一cohortを二重に数えず、H2へ既存Mechanism因子を移送しない。

paper manifest・CEM・台帳は[加算型更新案](../../artifacts/resource_applicability/track_a_h2_results_integration/2026-10-19/paper_manifest_update_DRAFT.json)へ結合する。既存MethodsのH2対象外という文章は元の論文scopeの記述として保全する。本文追加を採用する場合のMethods文言案は「H2の2条件は旧v1対象外の追加系列として、別入力来歴・固定n80/12の有限標本点結果を報告する。H4/H6/H8主表と図は従来scopeを保持する。」とし、未適用である。

## 承認・停止

今回の科学call・新統計解析・sampling・compileは0。既存SOURCE・raw・grant・ledger・marker・STOPを保持する。この文書で新実行を承認せず、v2追記案の正式採択、詳細科学レビュー、最終evidence freezeは実施しない。

**ユーザー**が本追記案の採択とH2論文掲載範囲を別途判断し、**Codex**は指示された文書反映のみを行う。今回の準備完了後はmandatory STOP。
