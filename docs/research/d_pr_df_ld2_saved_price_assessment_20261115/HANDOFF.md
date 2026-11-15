# 統合担当・B/Cへ：PR-DF LD2保存価格の照合と優先順位

**結論**：現Q3のPR-DF LD2に対応する直接compiled価格・S/E[S]・残差・評価済みモデル・非自明期待価格boundは、確認範囲では全26軸で未同定。B0I p7 S2の593,555,352 RZ、Full PF S2の838,693,580 RZとの順位はどちらも未判定。0価格で補完しない。

**次の一作業はLD1片側下界の確認を先行**する案を推奨する。対象はLD1 m9-Xの1軸だけ。SOURCE下界論証を試し、成立した場合だけ期待価格下界・shot加重部分下界を1件ずつ評価する。新path/build/compile0。成功は未確定で、LD1対PFが判定できてもLD2やPR-DF方式全体の順位は決まらない。

[REPORT](REPORT.md)／[26軸正本](ASSESSMENT.json)／[CSV](AXIS_MAP.csv)／[旧raw照合](RAW_CATALOG_CHECK.json)／[有限案](NEXT_ONE_TASK.json)／[静的照合](CHECKS.json)。全固定commit/path/selectorはASSESSMENT.source_refsとrowsに保存した。LD2の新しい数値入力・科学計算はない。

## LD2の固定候補と証拠

Q3 `a82b0f0ec7f0f11d9981ddb2343fcd5378a87503:docs/research/pr_df_fullh_q3_fixed_results_20261115/C_RESULT_JOIN.json#candidates[1]`、allocation **PR_DF_LD2_TOTAL_Q3_SAVED_CONTROL_01**。NはX/Y各軸で **[64,60,56,52,47,43,38,34,30,25,21,17,12]**。保存ideal finite-RTE law、元H4直線1Å/STO-3G/tol-only DF10、LD2/S2/Δ=.1/K2/元κ、P=0、scalar総T一回、新ZZ・測定付きfull controlled X/Y wrapper、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17を保持する。

original configurationは `1fa33556c17303f51ba03c4eae11189d35e59b3f57faad0e69d9fd08bf9245ae`、total viewは `e6fe7e6845dd5eae8829f54b8bbae655ccf1c81c320c4d0ee1bfed0399bb5583`。ALIGNMENT0b4b2d94のrows338〜363、C85557306 coverage26〜51、D748 requestへ同physical/m/axisで接続した。旧requestのscalar/keyを流用せず、将来はe8273676のcorrected total bindingを使う。

- **今使える**：上記候補・配分・law/profile、元入力出典、PF比較先の固定2総額。期待価格の値は未接続。
- **有限作業を加えれば使える**：既存のLD2 prefix別Runtime・normal sampler・同path S/C取得部品、Cの4セル案（較正m0 q1/R1、m3 q8/R5、m6 q64/R40、別holdout m4 q16/R10）。LD2の構造平均又はfull Cモデルを独自に評価する必要があり、n・compile数は今回新採択していない。これだけで全law/長roundの資格は得ない。
- **根拠不足**：全26軸の同候補直接価格、S/E[S]、期待残差、評価済みモデル、正当な片側bound。C1はprefix1・旧R/旧lowering、LD1 ES/モデル/rangeも別prefix。Pauli LD2は辞書・lowering・configuration d7dd4f52…が別で使わない。

旧carrier155529e2の公開raw5台帳（481/1/39/203/22 record）をIDと物理tupleで照合し、一致は0。旧H4/PR/prefix2価格はあるが、例えばT.8/q4/R4と現m3のT.8/q8/R5、T1.6/q16/R4と現m4のT1.6/q16/R10は不一致だった。公開manifestと元trial locatorを保持する。初回8dc41b35では自動承認レビューの拒否により本文を複製しなかった。後続の明示許可で旧候補134088ad0d3333745e3a/trial0の原本1件をbyte一致で公開した。[receipt](excluded_receipts/134088ad0d3333745e3a/trial0/result.json)／[出所・照合](RECEIPT_PUBLICATION.json)、raw公開SHA `1d8c5110c83fc289460a077a99cf2dbef725a447`。元実行SOURCEはreceiptに記録がなくnull、公開SHAとは区別する。現m3とのq/R/Δ不一致により価格入力には不採用で、26軸・PF判断・次案に変更はない。新科学計算0。

m12は752b9747のREQUEST/RESULT/WORKER_STOPで同physical f73c8ae1…を確認した。SOURCE8291f16c…はpreparation_contextのscalar照合でsampling前STOP、compile_logは空、価格はnull。e8273676の修正後runの取得実績はなく、旧T2を再実行しない。

「未同定」は確認範囲に限る。未列挙cache/archive/他serverの不存在を証明していない。価格recordの不一致と、所在未確認の領域を分け、未同定を新規投入許可としない。

## PF 2基準と部分下界の経路

| 基準 | 現LD2の判断 | 一つ得られれば前進する量 |
|---|---|---|
| B0I p7 S2：593,555,352 RZ | 未判定 | 同lawの正当な期待費用部分下界がこの額を超えればPF側を判定可能。LD2側には全体上界等が必要 |
| Full PF S2：838,693,580 RZ | 未判定 | 同様にこの額を超える部分下界ならFull PF側を判定可能。LD2側には全体の上側根拠が必要 |

非負価格では、正当なell<=E_I[C]を持つ一部round/軸だけでも `Σ_subset N ell <= T` を使える。有限path最小、旧価格、S/E[S]、モデル参照値をellとしない。全26軸を埋めること自体は片側判定の必要条件ではないが、現LD2には有用なellが未同定である。PFの閉包・Q3固定配分という条件も維持する。

## 二つの次案が答える問いと推奨scope

LD2の価格評価は、**LD1を代表にしたことがPR-DF対PFの結論を左右するか**を直接調べる作業である。LD2固有の同law期待価格又は有効boundが得られる範囲で初めて分かり、LD1を除外するだけでは代わりにならない。

LD1片側下界は、**固定LD1対2基準をより少ない追加作業で決着できるか**を調べる。LD1はS/kernelとcompiler/word調査が揃うため先行を推奨する。参照総額の大きさを下界と見なす判断ではない。

提案対象：LD1 m9-X、physical `06a78c7f9856f373336f`、T51.2/q512/R1020、N25、保存I。85dab58dの保存参照加重項2,828,507,555.268906326 RZは対象選択の動機だけに使う。LD2 m9/R320への数値移送は0。m12より短いq/R・1軸に限定する。

再利用入力は同保存RESULT.rows[18]のidentity/structural locator、9d95d3eaのCOVERAGE.values[18]、dcf905fcのSOURCE inventory、元Accounting/compose/new-ZZ、標準Qiskit1.3.0 SOURCE。まず全supportの正当な下側observable又はC−S下限を論証し、成立時だけ期待下界ell_9Xを評価する。新規算出上限は1軸下界1件・`25*ell_9X`の部分下界1件・比較2件、新path/H4 build/compile/再fit/Q3は0。

部分下界が838,693,580を超えれば固定LD1対両PF、593,555,352だけを超えれば少なくともB0I対LD1の片側判断が進む。不十分な下界からLD1が安いとは言わない。論証不能ならUNKNOWNでSTOPし、Y/m12/LD2/新pathへ自動拡張しない。実行は別指示である。

## ロードマップ・終了

LD2の価格未同定を、26軸のQ3・physical・公開raw/台帳・除外理由付きで具体化した。旧prefix2価格の存在と現候補への再利用可能性を分離。LD1先行の片側確認は固定比較に限り、LD2によって代表選択の結論が変わる余地は残す。ロードマップ本体は統合担当が更新する。

今回の新価格予測・総額・下界・LD1モデル移送・Q3・sampling/build/compileは0。元SOURCE/raw・旧STOP、LD1/Pauli/PFと他担当worktree/jobは保全。専用branchへcommit/push、remote一致確認後 **mandatory STOP**。
