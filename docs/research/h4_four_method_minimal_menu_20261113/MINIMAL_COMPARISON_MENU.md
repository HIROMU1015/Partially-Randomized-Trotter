# 提案：4方式の小さな比較menu

2026-11-13 JST。**11 schedule案を初期menuとする提案。科学的採択・優劣判定・実Gは行っていない。** 数は全候補の網羅ではなく、3種類の強いFull PF、既存truncation、Full-H RTEのK/normalization競合を残すためのもの。以下の幅や新layoutは未検証で、精度を満たす候補と推定していない。

## 共通にすること・方法ごとの自由度

同じ保存H_DF/sector/state/target、εE・既定誤差budget、h=.1/M12/元prior・decoder/count caller、P0、測定付きcontrolled full-wrapper RZ会計を使う。決定論PFのqとPRのq/Rを一致させない。formula・prefix・PF幅、finite K・q/R/整数配分は各方式の設計変数。Nも方式ごとの必要配分を使い、PRのnominalを他方式へ押し付けない。

比較claimはユーザー判断：同じproxy/ideal-lawに対する条件付き経験的診断と、同じ全事象loss/targetに対するpopulation RMSE評価を分ける。原論文型budgetを保証する主張ならA97146481の同target/coupling/L2契約を別に満たす。単に4方式で同じnのMCを実行しただけでは公平な精度達成比較にならない。

## 初期11案

PFは`q_m=max(1,ceil(T_m/Δ_request))`、実幅はT_m/q_m。負時間を持つY4/M5のsigned pieceは元binary64係数・順序を保持する。幅は外側macro-stepの要求幅で、各signed substepと混同しない。

| 提案ラベル | 物理recipe・全schedule案 | 選定理由 | 保存だけで比較可能か／判断を変える追加証拠 |
|---|---|---|---|
| PR-KEEP | 現canonical587d…、S2/L_D1/.1/K2・exact共通κ・現13round | 既存A mean/S3/C構造を最大限保持。新R最適化なし | 不可。population/target資格とDの同law compiled平均が必要。N×C競合を決める |
| F2-FINE | Full10、2nd、Δ=.1 | 既存S2の精度側を残す | m3/m4 signal/C保存。他11mean/Cと全RMSE Nが不足 |
| F2-COARSE | Full10、2nd、Δ=.4 | 同じS2で回路長と誤差の幅自由度を残す | m3 signal/C・m4 signal保存。粗い候補が精度を満たせるかを判断 |
| F4-FINE | Full10、標準Yoshida4、Δ=.1 | 2ndだけとの比較を避ける | m3/m4 signal、m3 C保存。signed負時間を含む全RMSEが必要 |
| F4-COARSE | Full10、標準Yoshida4、Δ=.4 | 高次formulaで粗いPF幅を許す強いbaseline候補 | m3/m4 signal/C保存。高次の追加回路とq減の競合を後で評価 |
| F5-FINE | Full10、M5 best、Δ=.1 | 既存M5 native/係数/testを活かす | m3/m4 signal、m3 C保存。全RMSEと長時間価格不足 |
| F5-COARSE | Full10、M5 best、Δ=.4 | 高次M5を不当に細かい幅へ固定しない | 同上。粗いwidthの適格性とcompiled費用を確認 |
| T2-REUSE | B0 prefix7、2nd、Δ=.1、one-body保持 | 旧H4全13round保存mean/referenceを再利用 | 不可。新signalは初手不要。元Hへの切断bias込みRMSE/NとCを接続 |
| T4-SCALAR | B0I prefix7、標準Yoshida4、Δ=.4、tail identity scalar補正あり | 旧PF commonの強いtruncated候補を残す。B0とB0Iを区別 | m3/T.8/q2とm4/T1.6/q4のsignal保存。切断biasが高次で消えるとは仮定しない |
| R2-ADAPT | Full-H one-body randomized、K2、下記の明示q/R案 | 旧T2研究のq8/R448/K2を設計anchorとして再利用。古いlight K4だけの対照にしない | 全13新mean/normalization/N/Cが必要。旧T2価格は新roundへ使わない |
| R4-ADAPT | 同じ全H/q/R案、K4 | K2に対するfinite-K bias・normalization/回路長の競合を残す | 同上。K4が強いとも弱いとも今回判定しない |

Full10はone-body＋全10DF fragmentを残す。T4-SCALARはB0Iであり単なるB0 label変更ではない。T2-REUSEの旧HP型Nは今回のRMSE Nへ変換しない。Full PFのK/κ/Rはnull、純RTEのDF B3への置換は不可。

## Full-H RTEの具体的なlayout提案

初期2案は、物理準備を専用`prepare_full_h`とし、one-bodyを含む全非scalar成分をランダム化、deterministic blocks=0、identity scalarをtotal Tで一回適用する。次の**新しい明示layout案**をA/Cが同一identityで受け取る。

\[
q_m=\max(1,\lceil T_m/0.25\rceil),\qquad
R_m=\max(1,\lceil 448\,T_m/2\rceil),\qquad s_m=q_mR_m.
\]

各群のevent時間はT_m/(q_mR_m)、割当は単一正時間pieceへ[R_m]、Kは2又は4、κ=None・`FULL_H_EXPLICIT_LAYOUT_NO_KAPPA_V1`。κの逆算やPR共通κへの同一視をしない。qは保存回路groupingとして保持し、s一致だけで価格を共有しない。

理由は、旧T=2のq8/R448/K2という既存有限研究のanchorを残し、長時間ほどtotal eventを概ねT²に増やす一つの有限な設計線を置くため。T²はnormalization/shot競合を調べるための仮説で、原論文の最適κや精度成立の採択ではない。短時間はceil/最小1が効く。**この式から実normalizationやNを計算していない。** K4に同じ定数448が最適という根拠もなく、単なる制御比較案である。

旧lightのT=.8 q16/R16/K4と、上記案のT=.8は別layout・別候補。旧T2はh=.1のdyadic roundでもない。したがって保存価格・N・meanを新13roundへ移植しない。新layout生成自体は既存D/B metadata APIで可能だが、今回は新physical ID群を発行・採択していない。

## 小さな境界補完だけを許す提案

| 観察・証拠が得られた場合 | 次の追加案（別承認） | 今回行わないこと |
|---|---|---|
| 同一formulaで.4は不適格、.1は適格だが費用差が結論へ影響 | Δ=.2を該当formulaへ1本だけ補間 | 最初から全PF幅/qを走査しない |
| 最良候補が.4側又は.1側の境界で、外側を調べないと比較が偏る | そのformulaに隣接1幅だけ追加し、未探索域を表示 | 有限menu内最良を全球最適と呼ばない |
| T2/T4の切断biasが精度を支配し、truncation方式の可能性を判断できない | 同じformula/widthでprefix9を1本だけ追加する案。必要ならp7/p9の比較 | rank全体のprefix走査や、cut biasの根拠なしの破棄をしない |
| R2/R4のbias/attenuation競合が結論を左右する | 当該KのR係数を448の隣接1倍率へ変更する案。K0は低normalizationの別候補として必要性があれば1本 | 全K/R格子や旧q/R価格の比例外挿をしない |

これは自動拡張規則ではない。追加が比較結論を変えないと判断できる証拠があれば実施不要。逆に境界候補しかなく不適格/欠測の方法を「負け」とするなら比較を止め、未解決を表示する。

## qualificationとNの扱い

1. 保存存在・SOURCE/identity照合は技術的な再利用条件で、精度達成ではない。
2. 全round meanと同じtarget/loss/risk契約が揃い、必要Nの評価が可能かを確認する。Nだけ又はCだけで候補を選ばない。
3. 元HへのRMSE資格を満たさない/未評価ならC/Gを空欄で保持。局所phase/radius診断だけで全RMSE INFEASIBLEともPASSとも認定しない。
4. 方式固有のN配分は同じ目標精度とpopulation/経験的claim水準を目指す。PRを含め全方式のshot最適化は別scopeで、固定pilot Nを最小Nと呼ばない。
5. 資格付きNと同じcandidate/law/loweringのCが揃って初めて既存会計器へ渡す。今回はmenu全体G/winner=null。
