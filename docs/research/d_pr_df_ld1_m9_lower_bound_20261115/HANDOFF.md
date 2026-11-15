# 統合担当・Cへ：LD1 m9-Xの片側下界SOURCE調査

**結果はUNKNOWN**。physical `06a78c7f9856f373336f`、q512/R1020、X軸25 shot、保存ideal finite-RTE lawに限定し、非自明なh≤C又はC−S≥bの全support論証を確立できなかった。存在しないことを示したのではない。既存のC≥0のみが残り、B0I p7 S2（593,555,352 RZ）・Full PF S2（838,693,580 RZ）との順位は両方未判定。

[REPORT](REPORT.md)／[SOURCE・入力・null状態](ASSESSMENT.json)／[静的検査](CHECKS.json)。作業branchは既存 `codex/d2-pr-df-ld1-m3-acquisition-20261115`、起点85dab58d。LD2 branchは変更していない。

## 対象・再利用

保存linear H4/1.00Å/STO-3G/tol-only DF10、PR-DF LD1/S2/Δ.1/K2/元κ・H/state/snapshot・順序、P=0、scalar c+s_R総T一回、symmetric-directional、新ZZ、測定付きfull controlled X wrapper、Qiskit1.3/rz,sx,x,cx/opt1/seed17を保持。T51.2。

- Q3 a82b0f0e `C_RESULT_JOIN.json#candidates[0].rows[18]` と軸正本950577f4 `AXIS_EVIDENCE.json#rows[18]`、N25を照合。
- 構造正本9d95d3ea `COVERAGE.json#values[18]` はmean_S=108641024.20905557、数値enclosure=[108641024.0948432,108641024.3358428] RZ。今回読取のみ、compiled下界へ未昇格。
- dcf905fc `RANGE_EVIDENCE.json#rows[18]` はC_lower=0、S_upper=1088759301、direct_residual_lower=null。既存のC−S≥−U_Sは非負性より有用な期待下界を与えない。差額は未算出。
- 85dab58dのm9-Xモデル参照加重額約28.29億RZは選定理由だけ。値の下界化・再fit・m4観測への差替えなし。
- §10.5の差分：origin fetch後のD2/C/D/B2/B/統合/A関連refと既知jobを照合。同等のm9-X証明/評価は確認範囲で未同定。LD2の旧receipt公開、Full-H/Pauli/PFの更新は別scope。

## 下界を妨げた箇所

1. **Sはcompiler後の価格ではない**。Accounting8457c443のliteral展開・全op列の逆basis対削除・wire別RZ-foldは、opt1の行列再合成/部分basis相殺を囲まない。P0公開442cafa1にはnew-ZZ形の保存basis対の孤立回路でS334/C194/C−S=−140という局所例がある。m9の反例としても全pathのbとしても使わず、qR倍しない。
2. **境界を越える相殺**。Compose155529e2とNativeBuilder c31ef424はhead/tail/outerをbarrierなしで1wrapperへ連結する。FastPath39ddd59eはSのpending状態もwrapper境界まで持ち越す。局所event/headの最低価格を独立に加算する根拠がない。
3. **active Rustと固定点**。Qiskit commit896d1c8bのZSXXはmod2π/1e−12の閾値・X短縮を使う。1q再合成後、CX等のInverseCancellationが新しい1q runを開き、opt1が反復する。Accountingの非ゼロ判定、1回目のrun数、new-ZZの局所2 RZを、最終まで残るRZとして固定できない。
4. **未取得の証明**。全supportの各wordについて、削除後も残るRZと下側observableの対応、又は相殺損失を一様に囲むpotentialが未成立。compilerの上側gate cap、有限観測残差、凍結モデル、F0のlaw間残差差上界は代用できない。

固定law宣言71ba49bbのrte.pyとprovider c31ef424は同じblob。Accounting/FastPath/NewZZの固定blobと現在fileは一致した。SOURCE不一致を理由に止まったのではなく、この一致する経路でも非自明な下側lemmaが示せなかった。標準release Rustを過去binaryとbit同一と認定する追加検証はしていない。Cの定義には正常なcompiler returnを仮定し、取得失敗を0にしない。

## 数値評価の入口・判定規則

非自明証明未成立のため `new_informative_ell=null`、`new_weighted_25ell=null`、数値評価件数0。既存の自明下限0を新しい有用な価格評価として数えない。

将来 `h≤C` が示せればE_I[h]、又は `C−S≥b` が示せれば保存E_I[S]の受入済み数値enclosure下端+bからℓを得る。そのときだけ25ℓを比較する。

| 基準 | 今回 | 後続で正当な下界が得られたとき |
|---|---|---|
| 593,555,352 RZ | 未判定 | 25ℓ>基準：B0Iが厳密に安い。等号：LD1は基準を下回らないまで。未満/不明：未判定 |
| 838,693,580 RZ | 未判定 | 25ℓ>基準：両PFが厳密に安い。等号：LD1はFull PFを下回らないまで。未満/不明：未判定 |

## ロードマップとSTOP

今回の1軸SOURCE論証調査では、固定LD1対PFを決着する近道は完成しなかった。次に必要な証拠は、同じword文法・opt1反復・数値閾値を含む下側observable/potentialの証明1件。新規科学実行案は採択せず、Y/m12/LD2やpath取得へ自動拡張しない。LD2を含むPR-DF全体の順位や全E[C]/CI/G/winnerも未認定。統合ロードマップ本体は統合担当が更新する。

新path・build・compile・再fit・Q3・F0再実行・構造期待値再計算・全wrapper statevector・演算子回帰suiteは全て0。m3/m4/P2・凍結モデル・旧raw/SOURCE/SCOPE_DEVIATION/STOPを保全。公開物のremote一致確認後 **mandatory STOP**。
