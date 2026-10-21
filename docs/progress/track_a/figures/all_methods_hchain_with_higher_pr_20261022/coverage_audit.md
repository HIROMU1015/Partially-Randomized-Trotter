# 表示・比較・独立確認の監査

600枠の完全な情報はcoverage_matrix.csv / coverage_audit.json。以下はstatusの一覧。
OK=plotted、I=saved-but-ineligible、M=missing、NC=not comparable。
AのNC高次点は別outer32パネルでplottedであり、旧PFとの直接比較・旧PF基準の比を作っていない。
独立確認は別columnで管理し、新outer32はH2/H4/H6ともnot yet independently confirmed。
H8は未保存。旧Aの探索点を示すことは、旧Pの有無や判定を新たに変更するものではない。

## A / USUAL_SIGNED

| H-chain / ε | PR S2 (A) | PR high best | full PF S2 | full PF S4 | B0 S2 | B0 S4 | B0+I S2 | B0+I S4 |
|---|---|---|---|---|---|---|---|---|
| H2 / 0.05 | OK | NC | OK | OK | OK | OK | OK | OK |
| H4 / 0.05 | OK | NC | OK | OK | OK | OK | OK | OK |
| H6 / 0.05 | OK | NC | OK | OK | OK | OK | OK | OK |
| H8 / 0.05 | OK | M | OK | OK | OK | OK | OK | OK |
| H2 / 0.01 | OK | NC | OK | OK | I | I | I | I |
| H4 / 0.01 | OK | NC | OK | OK | OK | OK | OK | OK |
| H6 / 0.01 | OK | NC | OK | OK | OK | OK | OK | OK |
| H8 / 0.01 | OK | M | OK | OK | OK | OK | OK | OK |
| H2 / 0.005 | OK | NC | OK | OK | I | I | I | I |
| H4 / 0.005 | OK | NC | OK | OK | OK | OK | OK | OK |
| H6 / 0.005 | OK | NC | OK | OK | OK | OK | OK | OK |
| H8 / 0.005 | OK | M | OK | OK | OK | OK | OK | OK |
| H2 / 0.001 | OK | NC | OK | OK | I | I | I | I |
| H4 / 0.001 | OK | NC | OK | OK | OK | OK | OK | OK |
| H6 / 0.001 | OK | NC | OK | OK | OK | OK | OK | OK |
| H8 / 0.001 | OK | M | OK | OK | OK | OK | OK | OK |
| H2 / 0.0005 | OK | NC | OK | OK | I | I | I | I |
| H4 / 0.0005 | OK | NC | OK | OK | OK | OK | OK | OK |
| H6 / 0.0005 | OK | NC | OK | OK | OK | OK | OK | OK |
| H8 / 0.0005 | OK | M | OK | OK | OK | OK | OK | OK |

## B / USUAL_SIGNED

| H-chain / ε | PR S2 | PR Yoshida4 | PR m5_best |
|---|---|---|---|
| H2 / 0.05 | OK | OK | OK |
| H4 / 0.05 | OK | OK | OK |
| H6 / 0.05 | OK | OK | OK |
| H8 / 0.05 | M | M | M |
| H2 / 0.01 | OK | OK | OK |
| H4 / 0.01 | OK | OK | OK |
| H6 / 0.01 | OK | OK | OK |
| H8 / 0.01 | M | M | M |
| H2 / 0.005 | OK | OK | OK |
| H4 / 0.005 | OK | OK | OK |
| H6 / 0.005 | OK | OK | OK |
| H8 / 0.005 | M | M | M |
| H2 / 0.001 | OK | OK | OK |
| H4 / 0.001 | OK | OK | OK |
| H6 / 0.001 | OK | OK | OK |
| H8 / 0.001 | M | M | M |
| H2 / 0.0005 | OK | OK | OK |
| H4 / 0.0005 | OK | OK | OK |
| H6 / 0.0005 | OK | OK | OK |
| H8 / 0.0005 | M | M | M |

## B / NONCANCELLATION

| H-chain / ε | PR S2 | PR Yoshida4 | PR m5_best |
|---|---|---|---|
| H2 / 0.05 | OK | OK | OK |
| H4 / 0.05 | OK | OK | OK |
| H6 / 0.05 | OK | OK | OK |
| H8 / 0.05 | M | M | M |
| H2 / 0.01 | OK | OK | OK |
| H4 / 0.01 | OK | OK | OK |
| H6 / 0.01 | OK | OK | OK |
| H8 / 0.01 | M | M | M |
| H2 / 0.005 | OK | OK | OK |
| H4 / 0.005 | OK | OK | OK |
| H6 / 0.005 | OK | OK | OK |
| H8 / 0.005 | M | M | M |
| H2 / 0.001 | OK | OK | OK |
| H4 / 0.001 | OK | OK | OK |
| H6 / 0.001 | OK | OK | OK |
| H8 / 0.001 | M | M | M |
| H2 / 0.0005 | OK | OK | OK |
| H4 / 0.0005 | OK | OK | OK |
| H6 / 0.0005 | OK | OK | OK |
| H8 / 0.0005 | M | M | M |

## C / USUAL_SIGNED

| H-chain / ε | PR S2 | PR high best | full PF m5 | B0 m5 | B0+I m5 | full PF S2 | B0 S2 | B0+I S2 |
|---|---|---|---|---|---|---|---|---|
| H2 / 0.05 | NC | NC | M | M | M | NC | NC | NC |
| H4 / 0.05 | OK | OK | OK | OK | OK | NC | NC | NC |
| H6 / 0.05 | OK | OK | OK | OK | OK | NC | NC | NC |
| H8 / 0.05 | M | M | M | M | M | M | M | M |
| H2 / 0.01 | NC | NC | M | M | M | NC | NC | NC |
| H4 / 0.01 | OK | OK | OK | OK | OK | NC | NC | NC |
| H6 / 0.01 | OK | OK | OK | OK | OK | NC | NC | NC |
| H8 / 0.01 | M | M | M | M | M | M | M | M |
| H2 / 0.005 | NC | NC | M | M | M | NC | NC | NC |
| H4 / 0.005 | OK | OK | OK | OK | OK | NC | NC | NC |
| H6 / 0.005 | OK | OK | OK | OK | OK | NC | NC | NC |
| H8 / 0.005 | M | M | M | M | M | M | M | M |
| H2 / 0.001 | NC | NC | M | M | M | NC | NC | NC |
| H4 / 0.001 | OK | OK | OK | OK | OK | NC | NC | NC |
| H6 / 0.001 | OK | OK | OK | OK | OK | NC | NC | NC |
| H8 / 0.001 | M | M | M | M | M | M | M | M |
| H2 / 0.0005 | NC | NC | M | M | M | NC | NC | NC |
| H4 / 0.0005 | OK | OK | OK | OK | OK | NC | NC | NC |
| H6 / 0.0005 | OK | OK | OK | OK | OK | NC | NC | NC |
| H8 / 0.0005 | M | M | M | M | M | M | M | M |

## C / NONCANCELLATION

| H-chain / ε | PR S2 | PR high best | full PF m5 | B0 m5 | B0+I m5 | full PF S2 | B0 S2 | B0+I S2 |
|---|---|---|---|---|---|---|---|---|
| H2 / 0.05 | NC | NC | M | M | M | NC | NC | NC |
| H4 / 0.05 | OK | OK | OK | M | M | NC | NC | NC |
| H6 / 0.05 | OK | OK | OK | M | M | NC | NC | NC |
| H8 / 0.05 | M | M | M | M | M | M | M | M |
| H2 / 0.01 | NC | NC | M | M | M | NC | NC | NC |
| H4 / 0.01 | OK | OK | OK | M | M | NC | NC | NC |
| H6 / 0.01 | OK | OK | OK | M | M | NC | NC | NC |
| H8 / 0.01 | M | M | M | M | M | M | M | M |
| H2 / 0.005 | NC | NC | M | M | M | NC | NC | NC |
| H4 / 0.005 | OK | OK | OK | M | M | NC | NC | NC |
| H6 / 0.005 | OK | OK | OK | M | M | NC | NC | NC |
| H8 / 0.005 | M | M | M | M | M | M | M | M |
| H2 / 0.001 | NC | NC | M | M | M | NC | NC | NC |
| H4 / 0.001 | OK | OK | OK | M | M | NC | NC | NC |
| H6 / 0.001 | OK | OK | OK | M | M | NC | NC | NC |
| H8 / 0.001 | M | M | M | M | M | M | M | M |
| H2 / 0.0005 | NC | NC | M | M | M | NC | NC | NC |
| H4 / 0.0005 | OK | OK | OK | M | M | NC | NC | NC |
| H6 / 0.0005 | OK | OK | OK | M | M | NC | NC | NC |
| H8 / 0.0005 | M | M | M | M | M | M | M | M |

## 未完了の独立確認

固定c7c1d822は実行準備の記録。H2/H4/H6 ε=.0005の主要3比較はn_C提案と固定shotがあるが科学実行前。
H4 ε=.005は補助設計で主要grantから除外。その他精度も独立確認完了を示す保存結果はこの固定資料にない。
提案n_C、pilot、旧n8、旧Pを新outer32の独立確認完了として数えていない。
