# 次の有限compile：対象・件数・STOP（すべて未承認・未実行）

固定SOURCEは `8b28dce02cc5b5da1a8f4f1718f0b9f855759978`。各callerは、自分自身・接続module・native・providerのbyteを、実行時にHEADと固定blobで照合する。段階はそれぞれ別に承認でき、自動では次へ進まない。前回のS-1、P-1、T3（固定SOURCE `04bead11`）は [前回のNEXT_EXECUTION](../d_kappa_s1_price_builders_20261114/NEXT_EXECUTION.md) のとおり。

## Pauli B0：LD1/m0 pilot

| 項目 | 内容 |
|---|---|
| 実行 | `python -B scripts/run_pauli_ld1_m0_pilot.py --execute-approved-pauli-b0`（親）→ 子 `--child`（元 `install()` が最初） |
| 対象 | A LD1 canonical `d905d219…`、m0、T=0.1、qPF=1、R=3、K2、cohort `D_PAULI_LD1_M0_WIRING_V1`、seed `14788159701244282917` |
| 件数 | native_context 1（prefix1）、P0 array読込 1、sample 1 path（3 event）、compose 1、wrapper 2、**compile最大2**、statevector 7以下 |
| gate | P0のbit順・label・head/tail・s_R、DF head（2 block、順序）、s_Rとnative tail identityのbit一致、scalarは最後の一回だけ、index0（low、leakage、full-2^n oracle 1e−9、X/Y Hadamard）、compiled vector 1e−7、seedが計画と一致 |
| 出力 | `docs/research/d_pauli_ld1_m0_pilot_results_20261114/run01/`、private `/home/AbeHiromu/projects/d-pauli-ld1-m0-20261114-private/run01/` |
| 位置付け | 配線と実回路の技術証拠。E[C]、law平均、平均単価ではない |

## Full PF定常幅（Cの計画A1）

| 項目 | 内容 |
|---|---|
| 実行 | `python -B scripts/run_full_pf_shortlist_prices.py --execute-approved-full-pf` |
| 新規cell | S2 m4/m5/m6（q7/14/28）、Y4 m4/m5/m6（q3/6/12）、M5 m5/m6（q2/4）の8 cell、**16 wrapper**。samplingは0 |
| 再compileしない | M5 m4（保存EXACT、X/Y）。値の受入はC |
| gate | 全11 block、tail identity=0、scalar＝c、既存index0 gate、physical IDがC COVERAGEと一致 |
| 出力 | `docs/research/d_full_pf_shortlist_prices_20261114/run01/`。PRICES、SIDECARS、IMPLEMENTATION_EVIDENCE、C_FIT_INPUTS、SAVED_RECEIPTS_FOR_C、RESULT |

Truncated p7のT3（18 wrapper）とFull PF（16 wrapper＋保存2件）は、同じround・q・実幅で対になる。合わせて実行するか、どちらかだけにするかはユーザーが決める。較正・holdout判定とfitはCの担当である。

## 共通STOP

- SOURCE・input・ID・q/R・scalar・block構成の不一致、gate失敗、OOM、継続不能な資源不足。
- 部分保存してSTOPする。retry、別seed・round・幅・lowering、閾値緩和、片軸への切替、0価格の補完はしない。
- 一時的なPSI・CPU競合だけでは止めない。他担当・他ユーザーのprocess・設定には触れない。CPU・RAMに人工上限は置かない。
- 実行後もmandatory STOP。fit、N、G、方式の採択には自動で進まない。
