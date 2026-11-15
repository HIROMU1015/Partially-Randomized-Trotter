# C→B/D1/統合担当：Full-H κ2 m0の参照単価案

**同条件の保存m0価格は確認範囲で0軸。現在C_0X=C_0Y=null。推奨は、事前固定したnormal path1本のX/Y実測値を、それぞれ経験的な1 shot参照単価にする規則。未採択・未取得。**

入口：[REPORT](REPORT.md)／[PROPOSAL.json](PROPOSAL.json)。規則名 `FULL_H_KAPPA2_M0_NORMAL_PATH_REFERENCE_V1`。

| 項目 | 内容 |
|---|---|
| physical ID | `FULL_H_2a9a68a11d08af234a8c` |
| 物理点 | m0、T=.1、q grouping1、R9、Full-H finite K2・κ2、P0 |
| X単価案 | `C_0X = compiled_RZ(ω0,X)`、値null |
| Y単価案 | `C_0Y = compiled_RZ(ω0,Y)`、値null |
| 費用への使用 | `N_0X*C_0X + N_0Y*C_0Y`。96を単価へ含めない |
| 価格の意味 | 単一normal pathの実測による経験的参照。取得後のkind候補 `DIRECT_COMPILED_FINITE_SAMPLE / FINITE_COHORT_ONLY` |
| 同条件の保存元receipt | 未確認/欠測。存在するようにlocatorを作らない |
| 固定正本 | `61f249e6e…:docs/research/full_h_d7_holdout_admission_plan_20261115/ADMISSION.json#rows[0],rows[1]`。完全SHAと各canonical/B_joinはPROPOSALへ保存 |

D7最小R566、P1 q8/R71、旧Full-H K4/q16/R16/T.8、旧T2 T=2.0、m0 B2/DF12/LD3/seed7を現m0へ流用しない。旧preparation `0e3989…` とD7 `923253…` の差が価格へ影響するかは未確認。旧資料のhash差を勝手に技術名差へ縮約しない。

## 最小取得・D1への仕様

新normal path **1本**、index0、X/Y各1回、**最大2新規full-wrapper compile**。二軸は同じpathを共有し、独立path数を二重に数えない。cohort `D_FULL_H_KAPPA2_M0_REFERENCE_N1_20261115_V1`、元trial_seedによるseed **15612141344395026302** を提案metadataとして固定した。samplingは未実行。K2のorder0/2を元normal samplerで扱い、forced/選別/resamplingなし。

元D1 `9cbabc…:scripts/run_full_h_kappa2_p2.py:acquire_path` と既存normal sampler・wrapper・compile/cache・phase/identity/numeric gateを再利用する。新しい1cell参照jobを固定し、旧D7 planを拡張しない。expected prep hash923253…、scalar total −0.33147781341681615、一回適用、new ZZ、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17/full measured wrapper/P0を照合。不一致・欠測はnull/STOP、自動retryや推測修正なし。

n1なのでX/Y各receiptの整数値を直接保存する。等しい値でも軸ごとに元SOURCE・measurement・fingerprint・receiptを保持。数値は未取得なのでPROPOSALの各axis `per_shot_RZ` はnull。CI・母期待値保証・全support診断を採用条件に追加しない。

## 採択後の比較範囲とSTOP

物理構成/law/実装を固定したNだけの変更に、この単価規則を使える。新配分のQ3適合は別の評価。実際の全shotを同じω0へ固定する規則ではなく、元signal/lawは変更しない。

採択・取得後に、m0経験的参照2軸とF1のm1〜m12保存24軸を結合した**新しい比較シナリオ**を作る。元F1の負値・全額nullとD7資格は保全。今回そのシナリオも全額も作っていない。

ロードマップ差分：「m6下界探索終了。m0保存同条件価格は確認範囲で0、1path/2wrapperの経験的参照単価案を提示。採択・有限取得の明示許可待ち」。ロードマップ本体は統合担当へ。今回新規sampling/build/compile/集計/予測/fit/Q3/全額再計算0、公開照合後mandatory STOP。
