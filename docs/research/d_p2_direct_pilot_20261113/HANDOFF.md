# P2直接paired pilot：完了引継ぎ

[REPORT](REPORT.md)／[PLAN](PLAN.json)／[RESULT](run01/RESULT.json)／[PATHS.csv](PATHS.csv)／
[CHECKS](CHECKS.json)／[人工16 PASS](TESTS.txt)。

## 完了範囲

- 実行SOURCE：`ea0706ab05313423f336614912758a93e9e8285c`、基点5a67cea7。
- replay：P1 run02のm0/index0・m6/index0、保存eventを無乱数復元。C/S/fingerprint/6指標/seed/digestが全一致。
- gate通過後、fresh `D_P2_DIRECT_PILOT_V1`、m0/m6各index0..3をnormal samplerで1回取得。
- 科学compile20/20、cache hit0、retry0、wall29分34秒。全10 pathのX/Yを個別に取得。
- 元C1/context/sampler/compose、compiled_price_fast、新ZZ、paired残差adapterを再利用。高速化本体は基点とbyte一致。

固定物理条件はLinear H4/1.00 Å/STO-3G・保存tol-only DF10、S2/L_D1/K2/h=.1/M12/P0、
元exactκ/H/state/snapshot、symmetric-directional full Hadamard X/Y。
新ZZ=`CONTROLLED_ZZ_CX_CRZ_CX_V1`、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17。

## 実H4性能とfresh統計

m6 replay：P1履歴phase-B3,139秒に対し新pathは復元込み881秒（約3.56倍）、price call857秒。
compile各242秒はP1とほぼ同じ。Python側の改善を実H4で確認した。
単回・履歴比較であり計時境界/並列負荷の差を含む。m6 peak RSSは旧約2.999→新2.794 GB
（decimal換算、process high-water）。fresh m6は846〜890秒/path、peak2.75〜2.82 GB。

| round | fresh n | D平均（X=Y） | D標本分散＝X/Y共分散 |
|---|---:|---:|---:|
| m0 | 4 | 187.5 | 6,529 |
| m6 | 4 | 74,388.75 | 134,716,355/12 ≈ 11,226,362.917 |

P1 development/replayは上の統計に加えない。C/Sの平均・標本分散・共分散もRESULT.summaryに保存。
全fresh pathでorder2=0。rare寄与0やX/Y一般同値とは認定しない。
全law E[C]・law距離・rare/tail・total CI・長round systematicは未資格/null。
2%/95%達成、十分N/RMSE/Gを供給した結果ではない。

## 次の有限案（未承認・未実行）

既存P2の候補精度（MC1.2%×E[S]、26軸Bonferroni z≈3.102）へpilot分散を代入すると、
m0はn144、m6はn1。m6の独立確認には少なくとも4 pathという設計floorを提案している。
候補は合計148 path/296 compile、今回の平均path時間による仕事量和は約63.7分。
4 long workerと小pathを重ねればwall約15分＋overheadの目安だが、保証ではない。
これはn=4の不確かな分散を使う正規近似設計で、law/rare未解決のまま精度認定しない。

全13roundを目標にする場合、未測定11round（m1〜5、m7〜12）へこのn/平均を移植しない。
長pathの実行可能性の次の最小案はm7(q128/R255) fresh2 path・X/Y最大4 compileの負荷診断。
32,640 event位置という入力規模以外のwall/RSSは未測定で、平均価格認定用nではない。
どちらの案も、新SOURCE・seed・有限予算と明示実行許可を別途固定してから実行する。

## 実装と保全

- runner `scripts/run_p2_direct_pilot.py`、helper `src/trottertracks/p2_direct_pilot.py`。
- `make_plan`：pinned P1/B metadataからreplay原本hash・全fresh seed・予算を固定。
- `restore_requests`：元configとevent生成関数を使い、保存event内容が違えばcompile前に失敗。
- `compare_replay`：保存C/S/両wrapper identity/6指標等が一致した場合だけfreshへ進む。
- `fresh_summary`：既存paired_residuals/cohort_summaryを再利用し、replayを除外。ddof1のexact rational。
- `planning`：将来nの設計値のみ。confidenceの達成フラグはfalse。
- private：`/home/AbeHiromu/projects/d-p2-direct-pilot-20261113-private/run01/`。raw eventは公開せず、70証拠fileのSHAをCHECKSへ保存。
- fingerprintはfull-wrapper pre-transpileのcanonical SHA。compiled回路自体のhashと混同しない。
- 旧SOURCE/input/raw/結果/KEEP/grant/ledger/marker/STOP、P1のPLAN・失敗/成功は不変。

## 担当とSTOP

- **ユーザー**：必要価格domain・精度・target law/PRNG・rare/tailの扱いを判断し、必要なら独立確認かm7負荷診断の有限範囲を別途明示承認する。
- **Codex D**：結果をcommit/pushし、remote SHA/tree/主要成果物を照合してmandatory STOP。次の本格平均価格評価には進まない。
- **Codex C**：現時点の追加実行なし。law診断F0が必要なら、その範囲を別途指定する。
- **Codex B**：作業不要。今回はfinite pilotとして参照し、E[S]+D平均を適格E[C]へ自動昇格しない。旧loweringと新loweringのidentityを分ける。
