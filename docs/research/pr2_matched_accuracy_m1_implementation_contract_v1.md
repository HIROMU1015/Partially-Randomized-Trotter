# PR-2 matched-accuracy M1実装契約 v1

作成日：2026-09-29
基準commit：`61bbaadfa852f569726b7a392f231606790b2aac`
status：`M1_IMPLEMENTATION_CONTRACT_FROZEN_SCIENCE_NOT_AUTHORIZED`

## 1. 目的と停止点

本書は、[M1前resource-map契約](pr2_matched_accuracy_resource_contract_v1.md)を機械実行可能な
候補identity、選抜関数、seed規則、schema、test、zero-compute dry-runへ変換する実装契約である。
候補の科学値を生成するM1本実行ではない。

今回許可するのは、標準libraryだけを使う候補列挙、fingerprint、synthetic selector fixture、既存S2
JSONのidentity確認、source hash確認、test、dry-run artifact生成だけである。development/held-out NPZ、
signal、trajectory、circuit build、transpile、量子shotを実行しない。

このcommit後は独立reviewで停止する。M1科学実行には、source identity、計算budget、process数、output、
test gateを固定した別authorizationが必要である。

## 2. 固定ファイル

| role | path |
|---|---|
| pure contract source | `src/trotterlib/pr2_matched_accuracy_m1_contract.py` |
| zero-compute runner | `scripts/run_pr2_matched_accuracy_m1_contract.py` |
| dedicated tests | `tests/test_pr2_matched_accuracy_m1_contract.py` |
| dry-run schema | `artifacts/pr2_matched_accuracy_m1_contract/2026-09-29/pr2_matched_accuracy_m1_dry_run_schema_v1.json` |
| reserved M1 result schema | `artifacts/pr2_matched_accuracy_m1_contract/2026-09-29/pr2_matched_accuracy_m1_result_schema_v1.json` |
| machine authorization | `artifacts/pr2_matched_accuracy_m1_contract/2026-09-29/pr2_matched_accuracy_m1_implementation_authorization_v1.json` |
| dry-run artifact | `artifacts/pr2_matched_accuracy_m1_contract/2026-09-29/pr2_matched_accuracy_m1_dry_run_v1.json` |

`pr2_matched_accuracy_m1_contract.py`はNumPy、SciPy、Qiskitおよび既存科学計算moduleをimportしない。
dry-run runnerもNPZ pathを開かず、既知S2 JSONとauthorization/sourceだけを読む。

## 3. M0 read-only台帳

dry-runが読む既存科学artifactは次の一件だけである。

- `artifacts/pr2_v4_s2_development/2026-09-29/pr2_s2_development_resource_result_parallel_v1.json`
- SHA-256：`bbe665724af438d242569b020ab6148dacce84716a75681949bea22c632dd27c`
- result fingerprint：`51fb92fdbcedb67299964eddd25e81c1faeaa55d7f7966765600a05e71d41a49`
- status：`S2_TRANSFER_CANDIDATE_AWAITING_REVIEW`
- held-out NPZ load：false

これは既知result identityの照合であり、候補再評価ではない。development snapshotもheld-outも開かない。

## 4. 候補台帳

base候補を次で固定する。

| method | rank | q | r | K | 件数 |
|---|---:|---|---|---|---:|
| B0 discard | 3,6,9 | 1,2,4,8 | 0 | 0 | 12 |
| B1 deterministic | 12 | 1,2,4,8 | 0 | 0 | 4 |
| B2 partial | 3,6,9 | 1,2,4,8 | 1,2,4,8,16,32 | 2,4 | 144 |
| B3 random-dominant | 0 | 1,2,4,8 | 1,2,4,8,16,32 | 2,4 | 48 |

base合計は208、random baseは192である。r32からの一段境界確認r64はsplitごとに最大1件、全体最大4件
なので、signal candidate上限は212である。r64は親r32 fingerprintをidentityに含め、r128は作らない。

各candidate fingerprintは少なくとも次をcanonical JSONへ含める。

- series、development snapshot SHA、Hamiltonian hash、state hash、state-vector hash。
- method、mode、rank、`T`、`q`、`delta`、`r`、`K`。
- `T`と`delta`のhex representation、`q*delta=T`。
- identity policy、coefficient threshold、outer formula、wrapper semantics。
- compiler version/basis/optimization/seed/coupling map。
- occurrence seed policy。r64だけは親r32 fingerprintも含む。

base candidate IDとfingerprintは全件uniqueでなければならない。

## 5. seed規則

future M1のrandom compile seedは次の全座標をSHA-256へ入れて導出する。

```text
master_seed
candidate_fingerprint
axis
trajectory
outer_step
tail_occurrence
rte_step
```

axisはcosine/sineで区別する。別trajectory、outer step、tail occurrence、RTE stepでseedを再利用しない。
candidate fingerprintは具体seed列を含めず、per-occurrence recordがseedを追加する。

## 6. compile選抜API

selector入力はaccuracy-eligibleなrandom候補の次の整数値だけである。

- `total_shots`
- `n_det`
- `n_rand`
- `n_fixed`
- `W_action=total_shots*(n_det+n_rand+n_fixed)`
- `W_tail=total_shots*n_rand`

compiled結果はselectorへ渡さない。tieは`method,rank,q,r,K,fingerprint`まで含む固定順序で解く。
選抜順はresource契約どおり、split anchor、q anchor、r64 boundary、tail challenger、proxy frontierの
split round-robin、W_action fillとする。random direct compileは最大16 cellである。

未選択の`(total_shots,n_det,n_rand)`非支配候補、未収容boundary、未収容W_tail challengerがあれば
`selection_limited=true`とし、future resultはwinnerまたはheld-out candidateを確定しない。compile後にも
未compile候補を安全に除外できなければ正式statusを`SELECTION_LIMITED`とする。

## 7. synthetic dry-runの意味

dry-runは192 random base候補へmetadataだけから作る合成整数を割り当て、selectorの全段階、r64最大4件、
16-cell cap、`SELECTION_LIMITED`経路を機械検査する。この値はsignal、shot見積り、gate count、科学的proxy、
M1候補選択ではない。future M1は実測前のsignal/bias/normalizationとaction proxyから新たにselector入力を
作らなければならず、dry-runのselected candidateを流用しない。

## 8. reserved M1 result schema

`pr2_matched_accuracy_m1_result_v1`は次を必須とする。

1. execution authorizationと全source hash。
2. development snapshot/Hamiltonian/state identity。
3. 208 base候補と最大4 boundary候補のledger。
4. raw/corrected signal、bias分解、normalization、axis shots、eligibility。
5. 16-cell選抜理由、未選択候補、`SELECTION_LIMITED`理由。
6. compile record、32/96/128 trajectory provenance、1-shot cost。
7. fixed-q8とmatched-accuracy比較、P>=0 lower envelopeと全crossing。
8. 可変q、outer repetition、fresh seed、B_total、fingerprint、Re/Im wrapper gate。
9. 全counter、held-out未開封、S3 false、automatic next stage null、mandatory stop。

正式科学statusは次のいずれかである。

- `CONTINUE_TO_FROZEN_TRANSFER_REVIEW`
- `NARROW_TO_TECHNICAL_NOTE`
- `STOP_DUPLICATIVE`
- `SELECTION_LIMITED`
- correctness不成立時だけ`IMPLEMENTATION_GATE_FAILED`

## 9. resource上限

future authorizationが許可できる最大値を次で固定する。

- base signal候補208、r64追加最大4、合計最大212。
- deterministic/discard direct compile 16 cell。
- random direct compile最大16 cell。
- random初回32 trajectory/cell、固定条件成立時だけ96追加、最大128/cell。
- random trajectory最大2048、random full wrapper最大4096。
- deterministic/discardの2軸32 wrapperを加え、full wrapper最大4128。
- quantum backend shot 0、held-out NPZ load 0、S3 0。

実時間、peak RSS、worker数、cache、再開規則はfuture execution authorizationでこの上限内に固定する。

## 10. zero-compute gate

dry-run artifactは次を全て0として保存し、validatorが拒否条件にする。

- development/held-out NPZ load。
- molecule計算、signal評価、random trajectory sampling。
- circuit build、compile、full-wrapper compile、量子shot。

held-outについてpath stat、hash read、NPZ loadを全て0とする。authorizationは
`m1_scientific_execution_authorized=false`、`held_out_access_authorized=false`、
`s3_authorized=false`、`automatic_next_stage=null`を維持する。

## 11. review後の分岐

本契約、source、test、schema、dry-runをcommit・pushした後に停止する。独立reviewが変更を要求すれば、
M1計算前にv2として再freezeする。通過した場合だけ、同じsource identityとresource上限を参照する
M1 execution authorizationを別commitで作る。authorizationなしにrunnerを科学計算対応へ拡張しない。

held-out、S3、追加geometry、別PF、H12、長RPE、最終総costは引き続き未承認である。
