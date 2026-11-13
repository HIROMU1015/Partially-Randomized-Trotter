# 次に実行可能な有限compile（すべて未承認・未実行）

価格callerの固定SOURCEは `62ae1bd56718287de5d958cf0ed6411f5b7395c9`（branch `codex/d-price-records-c-bridge-20261114`）。各callerは、自身・接続module・native・providerのbyteを、実行時にHEADと固定blobで照合する。段階はそれぞれ別々に承認でき、自動では次へ進まない。

| 段階 | 実行 | SOURCE | 件数 | 価格recordの分類 |
|---|---|---|---|---|
| Pauli B0（LD1/m0） | `python -B scripts/run_pauli_ld1_m0_pilot.py --execute-approved-pauli-b0` | `62ae1bd5` | sample 1 path（3 event）、**compile最大2** | `DIRECT_COMPILED_FINITE_SAMPLE`／`FINITE_COHORT_ONLY`（n=1、Gでは `FINITE_COHORT_SCENARIO`） |
| Full PF A1 | `python -B scripts/run_full_pf_shortlist_prices.py --execute-approved-full-pf` | `62ae1bd5` | 8 cell、**16 wrapper**。保存M5 m4は除外 | `DIRECT_COMPILED_DETERMINISTIC`／`DETERMINISTIC` |
| Truncated p7 T3 | `python -B scripts/run_truncated_p7_prices.py --execute-approved-t3` | `62ae1bd5`（v2のrecord） | 9 cell、**18 wrapper** | `DIRECT_COMPILED_DETERMINISTIC`／`DETERMINISTIC` |
| Full-H κ2 S-1 | 旧branchの `run_full_h_kappa2_signals.py --execute-approved-s1` | `04bead11` | mean 13、normalizer 13 | 価格なし（signal） |
| Full-H κ2 P-1 | 旧branchの `run_full_h_kappa2_p1.py --execute-approved-p1` | `04bead11` | **compile最大2** | 旧名のまま（配線用。Gに使うなら新版が必要） |

## STOP条件

- **Pauli**：P0のbit順・辞書・head/tail、s_R≠nativeのtail identity、canonical候補と固定requestの不一致、seed≠`14788159701244282917`、scalarが一回でない、index0（full-2^n oracle）、compiled vector。
- **Full PF・Truncated**：保持block数とtail identity（Full PFは0、Truncatedは保存scalarとbit一致）、scalar、index0、physical IDの不一致。
- **共通**
  - SOURCEとbyteの不一致、OOM、継続不能な資源不足。
  - 部分保存してSTOPする。retry、別seed・round・lowering、閾値緩和、片軸への切替、0での補完はしない。
  - 一時的な競合だけでは止めない。他担当・他ユーザーのprocessと設定には触れない。CPU・RAMに人工上限は置かない。
  - 実行後もmandatory STOP。fit、N、G、比較policyの採択へは自動で進まない。
