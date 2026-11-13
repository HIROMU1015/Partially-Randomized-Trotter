# 未実行：PRのRPE全体探索へ進む最小の有限計画

2026-11-13 JST。**今回、新しい実H4 signal・normalization再解析・candidate探索・最適化・N/C/Gを開始していない。以下はユーザーが条件・scopeを別途承認するための有限案。** 数値範囲は新たに採択せず、旧Track A最良・L7/8/9・m12だけの案を正解と扱わない。

## A. 固定前に必要な選択

| 項目 | 最小pilotの推奨案 | 正式状態 |
|---|---|---|
| 系・target | 保存H4 DF10、全DF-H同sector/保存stateの条件付きtarget | 旧bindingを保全。DF前分子Hの厳密精度保証は含めない |
| h/M/誤差契約 | baselineのh=.1/M12・元prior/decoder/εE/予算を変更せず比較 | 候補最適化の新正式条件には未採択。ユーザーが同条件比較を承認 |
| baseline | 保存PR全13roundのS2/prefix1/Δ.1/K2と元exact共通κ | 固定成果を対照として保持、最適と仮定しない |
| alternative | 一つの別prefix pを旧保存descriptorの集合{2,3,5,7,9}から選び、S2/Δ.1/K2を揃える | **p未指定**。数値順位を今回作らず、L7等を勝手に選ばない |
| κ | 初回はbaselineと同じexactκを使い、各prefixの係数norm λ_pからR_mを再生成する案 | 表示近似でなく元CONFIGURATIONのexact表現。共有することをユーザーが指定 |
| normalization | finite_B_diagnostic、旧allowance/数値gateを減らさない | nominalと全RMSE保証は別。現在の資格を維持 |
| numerical recipe | direct head expm＋既存paired Taylor/power＋exact-tail対照 | 新PR adapterの数値SOURCEを固定、追加のexpm理論検証を目的にしない |

二つの全scheduleでsplitの効果を分離する最小の比較を提案する。共通κでもλが違うとR_mは違うため、同じRで比較したと書かない。最初の2scheduleで最適κや最適Lを確定しない。

Mやtargetを変更する、複数κを同時比較する、formula/Δ/Kも変える場合は、そのfinite menuとpolicyを先に明示する。列挙数gに対し全round候補数はΣ_i(M_i+1)。全候補の巨大Cartesian gridを自動生成しない。1条件でも未指定のまま実dataのB spec・λ scheduleを作らない。

## B. 実行前の技術作業と保存再利用

1. Bの正本`generate/_validate`又はA fixed-grid adapterで全13roundを作り、今回のmetadata helperにphysical対応を渡す。新candidate管理・cost optimizerは不要。
2. 既存SOURCE_INDEX `7a8bc7f132e166390c943e6b32c92cc6abe1baf4`のH4 contextにはprefix1/2/3/5/7/9の`lambda_R/scalar` fieldがある。数値は今回評価せず、実行時にbinding/元hash・source付きでcopyする。prefix8の保存descriptorは確認できないため、この最小pilotへ自動追加しない。
3. bba物理入力契約の全m0…12には独立reference pair fieldがある。H/state/T一致なら全候補で共用する。新referenceは**0 action**、欠測・不一致時は追加計算へfallbackせずSTOP。
4. A169dの保存13meanやTrack A短時間値はphysical recipe・law・backend・allowance・SOURCEが一致する場合のみ再利用候補。direct-headへのbackend変更を理由に、旧spectral meanを新backend結果へ再ラベルしない。必要な保存値の新しいacceptance/allowance診断は別途承認scopeへ含める。
5. 元head expm・finite Taylor・scalar・allocationを組み合わせる最小providerはまだ未実装。承認範囲で実行に進む前に実装SOURCEを固定し、人工非可換行列でexact-tail/raw/corrected・負時間・closure・最初/最後roundのboundaryを確認する。新科学実行の段階ごとに承認証明書を要求しない。

## C. 最初の有限科学call案

**二つの全schedule、13roundずつ＝最大26 round-candidate評価**を承認対象とする案。同一physical recipe・同numerical sourceの重複を事前排除し、exact-tail内部対照は同prefix/formula/δで共有する。保存一致数が分かれば実行前に新規call上限を減らすが、範囲外の候補に振り替えない。

| 処理 | 有限上限・出力 |
|---|---|
| saved input | 同じH/state/contextの復元1回。SCF/DF/state/ground生成0 |
| tail metadata | prefix1と選択pの保存s/λをcopy、各A_p行列を組む。新係数算出・新fit0 |
| macro mean | **最大26候補**、各候補でcorrected/raw/exact-tailの3complex出力（従って出力scalarは最大78、X/Yで倍に数えない） |
| normalizer | S2の1pieceなので最大26 event-normalizer評価。既存finite式、logB/ULP感度/attenuationを保存 |
| independent reference | 保存13 pair共用、新reference0 |
| 検査 | exact-tail norm/unitarity、独立raw/corrected closure、数値allowance、raw radius・local phase/bias。未知actual law/ground誤差はnull |
| 含めない | 実nominal/十分N生成、B decoder simulation、価格compile/fit・G比較、追加κ/L探索、trajectory sampling |

「26 complex mean」と省略してraw/corrected/対照の内訳を隠さず、上記round-candidate単位と最大78出力を区別する。内部operator・spectra・cacheはcandidate生成や新ground計算ではないが、新実H4のsignal評価に含まれる科学作業である。

baseline側を保存13meanの新policy/allowance再解析で扱うことを明示承認し、backendの差が比較に使えるという証拠が揃う場合には、新候補13＋baseline保存13の限定診断へ減らせる。ただし現時点でその科学的同等性を採択せず、上限26の同backend比較が最も明瞭な初回案である。旧m3/m4のnorm欠測を0で補わない。

全round入力を初めから要求するが、実処理はearly/中間/terminalの順序やshared cacheで効率化できる。m12の結果だけを見て他roundを未評価のまま「最適」「FAIL」と呼ばない。実行は一つの有限batchとして扱い、閾値変更・候補差替え・科学的retry・範囲外callをしない。

## D. STOPと判定できる範囲

SOURCE/候補/λ・scalar/law/preparation/reference不一致、非有限、normalizer精度未解決、元norm/unitarity/closure gate失敗では残差・例外型・未開始/取消を保存してSTOP。corrected/raw finite-K operatorの非unitarity自体を、exact-tailのunitarity gateで誤って失敗にしない。旧SOURCE/input/raw/KEEP/grant/ledger/marker/STOPは変更しない。

このpilotは全roundのsignal・B・局所biasと数値安定性を比較する入力を得るだけで、同energy RMSEの十分NやG最小を証明しない。large bias/logBは次の評価順の材料とし、energyへ無根拠に換算しない。正規化だけによるR最適化や単一時間価格での選抜はしない。

## E. 次の段階（今回非実行）

pilot後、**B**が同じ誤差契約で全scheduleのN/RMSE証拠をどの範囲まで評価するか提案し、**価格担当**が同physical全round Cのmodel/direct coverageを返す。cheap normalization/構造指標は明示した有限menuの評価順にのみ使い、deferred候補を残す。

共通κ等の有限集合が決まれば、その全scheduleをBへ渡し、qualifiedなN/C/全riskが揃った範囲だけB cost/selectで比較する。K・PR-Y4の現V2未対応が必要なら、先に担当Bの最小API差分を解決する。全探索結果を見た後の統計的再検証・holdoutは、事前の科学計画へ別途含める。

**ユーザーが次に決定・承認する最小事項**：選択p、初回2scheduleと元exactκ/h/M/誤差契約の維持、direct-head実装SOURCE、上記最大26候補（78complex出力）・normalizer26・新reference0のsignal-only scope。N/decoder/C/Gや次の候補探索はその許可に含めない。今回の技術成果公開・照合後mandatory STOP。
