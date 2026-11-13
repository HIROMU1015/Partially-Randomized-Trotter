# 承認済みP1の固定実行経路

75fb0ba1の[SCIENCE_PLAN P1](../df_head_pauli_tail_preparation_20261114/SCIENCE_PLAN.md)とユーザーの今回の明示承認に従う。P0結果は4220f3006。新コード・本書・人工テストをcommitして科学SOURCEを固定し、fresh run01を一回起動する。

## 対象・上限

保存H4 DF10/Linear1.00Å/STO-3G、full256/sector(2,2)36、S2/ΔPF=.1/K2/h=.1/M12/P=0、同H/state/snapshot・completion・fragment順・元prior/decoder/誤差契約を維持する。

共通κは正本の有理数
`13112614441153360648422410673503886836581176429267589536059228160/539496659426909122096437389293855970805348475396548904610992201`。
Pauli λはP0辞書の保存binary64値をexact Fractionへ写してB mapperに渡す。κ値は変更しない。

1. LD1/LD2×m0/T=.1・m6/T=6.4のPauli finite-K mean：4件。
2. LD1/m0の保存DF R=2に一致するPauli対照：必要時1件。共通κと物理条件が完全一致すれば重複計算しない。
3. 上記LD1/m0対照と同じ条件のfull256 mean：1件。

合計最大6 mean、raw/corrected/exact-tail最大18complex、unique finite normalizer最大5組。full/36の同tau/Kはcaller-owned cacheで共有する。新reference・DF mean・P0辞書/演算子再構成・固有値・sampling/circuit build/compile・RPE/N/MSE/C/Gは0。保存DFのLD1/2×m0/m6の4rowをSOURCE/数値gate受入に限って参照し、旧値を変更しない。

## B identity・schedule

最新公開B `ce0517de6e7b58cf4bf4c69f99f998d86099c406` の`pr_rpe_rmse_controller.generate/_validate/paper_event_mapper`を、既存の固定B metadata loader経由で再利用する。A独自のcandidate/hash生成器やcontrollerは作らない。

元のDF configurationは元ID/SOURCEで保持。Pauli辞書kind・hash・SOURCE・λ/scalar・bit順をsplitとlawへ明記し、B自身が新canonical configuration/candidate IDを生成する。元Aのκ定義名を保持しつつ、B native APIのenum `PAPER_TOTAL_ROTATIONS`へ明示対応させる。両者は同じr*=κλ²T²、ceil/clamp/uniform-outer方式で、normalization上限型κではない。

共通κの2構成は全13round metadataを生成するが、科学評価はm0/m6だけ。fixed-R対照が必要ならBのexternal mapper APIでm0だけDF Rへoverrideした診断用configurationを作る。κ値は保持し、当該roundではκを適用しないことを明記する。他roundは共通κのmetadataのみ保持し、科学実行しない。fixed-Rを本番scheduleや新κの最適値として採択しない。

## 数値計算・比較

- P0のCOMMON/LD別operator NPZをimmutable commitから読み、保存stateは既存save-only loaderで一回読む。full256 extension/Pauli辞書の再計算なし。
- headは保存block、tailは**P0 Pauli_nonidentity行列**そのもの。保存s_Rとcは従来どおり全Tで一回だけ適用する。s_Pとs_Rの微小差やP0 operator残差はそのまま記録し、補償identityを新eventへ追加したり、saved scalarを変更したりしない。
- 既存`pr_rpe_signal_provider`のdirect-expm head、paired Taylor、normalizer、行列累乗、独立raw/corrected、exact-tail、R-aware empirical allowanceを再利用。providerの最小変更は辞書付きB ID検査と任意normalizer共有cacheのみ。旧DF default経路・数値式・gateは維持する。
- 保存full-H独立reference pairを共有し、新actionは0。reference pair gate1e−10、state norm1e−12、unitarity/closure/full-sector比較1e−9を維持。最大normは従来と同じmacro/power境界の観測であり、全event内のrigorous上界ではない。
- 保存DFのphysical recipe・B identity/SOURCE・raw/corrected独立性・数値gate/allowance・reference anchorを確認。同じH/state・head・formula・q/R/K・signed allocation・scalar規則を確認できた場合だけsignalを比較する。R不一致なら数値差を計算せず「同条件比較不可」を残す。
- fixed-RのDF/Pauliは同Aの数値同一性がP0 gateの範囲で成立する条件付き比較である。exact-tail/correctedの差を1e−9で照合するが、λ依存のraw/Bは異なり得る。full/36ではraw/corrected/exact-tailとlogB/attenuationを同gateで照合する。

## 出力・STOP

新runのPLANにB正本configuration/13round metadata・selected round・control区別を保存し、各mean JSON、normalizer cache、数値診断、保存DF受入・比較、partial/final resultを保存する。B向けpacketは共通κ各構成のm0/m6だけ取得済みとし、他11round・十分N/global RMSE/C/Gはnullのまま。

SOURCE/input/law/dictionary/candidate/scalar不一致、数値gate/normalizer精度/非有限/full-sector不成立ならpartial保存後mandatory STOP。閾値変更・再正規化・fallback・retry・候補差替え・追加scienceをしない。最初の失敗で停止する契約とfull/36の依存のため小行列を順に処理し、同matrix/normalizerはcache再利用する。資源の恣意的capや共有設定の変更は行わない。

人工6件、既存provider10件・planner7件がPASS。metadata-only preflightは新実schedule/配列load/科学leafを起動しない。承認証明書・新grantは不要。完了後REPORT/HANDOFFをcommit/push・remote照合してmandatory STOP。
