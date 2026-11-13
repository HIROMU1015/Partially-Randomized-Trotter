# 4方式最小menu：B → ユーザー/A/C/D

2026-11-13 JST。[REPORT](REPORT.md)／[matrix](BASELINE_EVIDENCE_MATRIX.md)／[menu](MINIMAL_COMPARISON_MENU.md)／[次の有限scope](NEXT_STEPS.md)。**menuは未採択。新科学計算・実G・winnerはない。**

提案はPR現1、Full PFの2nd/Yoshida4/M5×幅.1/.4の6、truncated B0p7/S2/.1＋B0Ip7/Y4/.4の2、Full-H RTE K2/K4の明示適応layout2、計11schedule。方式ごとのq/Rを強制一致させず、同H_DF/保存state/sector/目標エネルギー・loss/risk・測定付きfull-wrapper RZ/P0を共通にする。

## 既存の受入経路を使う

1. ユーザー採択後のFull PFは`full_baseline_v2_requests.full_pf_spec`、pure RTEは`full_h_rte_spec(...,kappa=None)`と`layout_by_round`を使用。B0/B0Iのretained Hamiltonianはcontrollerのtruncated経路で元Hと別記する。
2. A/Cは同じ`requests(config)`のV2 identity/law/物理mapping/SOURCEを使う。保存物理IDだけでN/Cを受け入れず、旧73価格と元receiptを参照する。旧118候補や136候補をまとめて新V2へ変換しない。
3. `pr_rpe_rmse_evidence.plan(entries,price_policy='qualified_expectation')`が既存の不足証拠入口。将来の完成schedule/recordだけを供給する。E[S]・local path・旧HP型NをN/Cへ渡さず、現在のG/winner=nullを維持する。
4. 新lowering価格を正式接続するときは、明示lowering IDをversioned price profile/contextに付ける最小変更を行う。元B profileはcompiler hashが同じでもlowering_idを明示していない。新旧価格を同じrecordへ上書きしない。Nを共有するなら物理同値性のSOURCE bridgeも示す。

現在は採択された実menu/新compiled期待recordがないため、この将来のprofile変更を先行実装していない。既存の外部physical mappingで足り、native provider全面再実装も不要。コード変更0、新framework0。40件の既存人工/保存metadata回帰がPASSした。

## SOURCEごとの引渡し状態

- **A97146481**：全事象loss/population risk/target/raw/lawの不足整理をそのまま再利用。Bで別RMSE理論・mean再評価器を作らない。
- **C9d95**：全26構造recordのfull V2 identity/law/physical IDはB canonicalと一致。元kernel SOURCEが旧4値と新22値で異なる点もEVIDENCE_METADATAへ保存。値は再計算せず、compiled C資格false/nullのまま。
- **D e7dca**：保存136 Full PFの73価格SOURCE照合と欠測63を利用。RMSE資格への昇格なし。Full PF6案は各m3/m4に保存signalがあり、他11roundは当該grid外。
- **D54b64**：P1は新loweringの4 path開発cohort。旧C1/長時間/Full-H価格を上書きしない。P2/fit/全law E[C]の代用なし。D現在の作業へ指示・変更・停止をしない。
- **旧H4 B0**：89c91c6に全13診断。m3のraw/referenceは元PF common carrierへ戻り、他12はresult内signal。旧HP型整数NをRMSE Nへ流用しない。

## 次の担当

**ユーザー**：claim水準・11案又は縮小menuを選び、必要な有限科学scopeを明示許可。**A/B**：不足meanの最大20セルscreen、残す候補だけN/RMSEという案を固定する。**C/D**：D既存P2研究を継続し、同loweringの適格C計画を供給。**B**：公開証拠を既存plannerへ接続し、実Gは別許可後だけ。

最初から100個の不足unique mean全てを計算する提案ではない。screen20はその内数であり、保存値再利用と必要candidateだけの完成を優先する。Full PFの同formula・異要求幅で同じm0物理回路になるセルも1回だけ取得する。不要な全reference再計算、新8次、全格子探索、全PF価格の再監査を増やさない。

今回のcommit/push・remote SHA/tree/主要成果物照合後mandatory STOP。新しい科学実行・正式4方式比較へ自動進行しない。
