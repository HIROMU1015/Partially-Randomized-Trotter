# A P0 → B/C/D

科学SOURCE `fd111f989b9cd2f052d7189e4b7b6fab10aa1a9f`、計画 `75fb0ba114ac518bfe5d65d0164cee07d7469bb4` P0。保存H4 DF10/Linear1.00Å/STO3G、sector(2,2)/36・full256、同H/state/snapshot、LD1/LD2。operator/辞書の81数値gate通過、mean/N/価格0。[REPORT](REPORT.md)／[機械可読HANDOFF](HANDOFF.json)／[入力binding](run01/FIXED_INPUTS.json)。

## 今回渡せるもの

- 全Pauli辞書：`run01/LD{1,2}_DICTIONARY.json`。非identity184項、保存36basisからsector外へ遷移するlabel136項。全係数・符号・p_j・bit作用・辞書hash・SOURCEを保持。
- LD1 λ_P=3.3066052942233157、s_P=.4177175120626943、保存s_R=.4177175120626939。LD2 λ_P=2.0550998769244275、s_P=.22237590941070173、保存s_R=.2223759094107018。保存DF λも併記。c=2.29310124732とheadは従来のまま。
- `COMMON_OPERATORS.npz`は元one-body/10weighted block・full H・保存sector index、LD別`OPERATORS.npz`はhead/tail/非identity Pauli和。数値gateとraw JW係数は同runのJSONに保存。state本体の新規複製なし。
- Pauli labelは既存Qiskit little-endian。保存CAR256/36に適用する場合は明示的なbit reversalが必要。`label_in_saved_tensor_order`と`mapping`を参照し、辞書hashだけを見て別basisへ作用させない。

## Bのcanonicalと将来のsignal受入

`original_DF_configuration_ids`は比較元の**DF専用ID**として保持。辞書のSHA256はpayload同定であり、Bのcanonical IDではない。今回Pauli configuration/schedule/CIDは生成していない。旧DF IDの再ラベルやA独自ID生成はしない。

Bが次のdictionary identityをsplit/law/context/cacheへ結びつける：`GLOBAL_COLLECTED_JW_PAULI_OF_SAVED_DF_TAIL`、prefix、全label/係数hash、s_R/λ_P、mode順、mapper/collector SOURCE・係数policy。同じR/K/timeなら合計Aのfinite polynomialが同じになる条件と、共通κでλが変わるとR/allocationも変わる点を区別する。κの意味・K・prior/decoder/誤差契約を変更しない。

P0では合計operatorのsector closureだけを確認した。P1で既存paired polynomial/normalizerと独立raw/corrected・exact-tailを接続し、LD1/m0の同条件full/36 mean対照を行うまで、mean受入済みとしない。unknownなactual law誤差・厳密ground同定はnull/未資格を維持する。旧DF mean/NをそのままPauliへ使わない。Bのdecoder/risk/N器は変更していない。

## C/Dのevent・価格

個別Pauli eventはfull256作用を基準とする。射影Pauliは一般にinvolutionでなく、36のPauli回路として扱えない。辞書のQiskit label、support、sign、product/application順、order phase、共通scalarの一回会計、DF head境界を既存controlled Pauli primitiveへ結びつける技術作業が必要。P0では回路生成・sampling・native loweringを呼んでいない。

λ_PがDF値より大きいことからC/Gの順位は分からない。旧P3/O1の価格・K/identity/lawを移送せず、Bの同canonical候補とwhole scheduleを価格contextへ渡す。既存DのDF用tail factoryへこの辞書を偽装投入しない。

## 次の実行境界

ユーザーがP1を承認した場合だけ、元計画の最大6 mean/18complex・finite normalizer最大5・保存reference共有/新reference0のscopeを固定する。新科学SOURCEはPauli provider接続差分のcommitで記録する。K/κ・追加prefix・全13round・N/RMSE・compile/価格/Gへ自動進行しない。

今回の成果を公開・remote照合後mandatory STOP。ユーザーが次の有限科学範囲を決め、A/B/C/Dが各担当分を別途進める。
