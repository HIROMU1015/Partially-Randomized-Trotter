# DF Truncated PF：相殺安定性P1・相殺依存度P2

2026-11-08 JST、SERVER 2 / Codex C。独立したP1→P2研究。基点は静的調査`12e74dd68568f493ed39937f8856f26ce48d33e1`、固定科学入力・参照は`da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13`。旧SOURCE、結果、STOPは維持する。

**P1は指定したH6の有限5点で安定性を支持した。P2はH4で保守的評価でもFull PF・Closed P3 PRより低コストなpolicyを確認した。H6では保守的policyはFullより安いがPRより高く、今回のcohort・選択では両者に対する低コスト性を確認できなかった。** このP2の負結果は元combined評価Aの保証・低コスト性を否定しない。

## 1. 結果の要約

| 分子 | Truncated A 総RZ | Truncated B 総RZ | 参照Full 総RZ | 参照Closed PR 総RZ ± MC SE | P2判定 |
|---|---:|---:|---:|---:|---|
| H4 | 3.337772238e+11 | 3.864575491e+11 | 5.647934021e+11 | 1.303898609e+12 ± 4.834e+09 | 結果A |
| H6 | 1.222907256e+12 | 1.868265339e+12 | 2.082026044e+12 | 1.617753939e+12 ± 2.001e+09 | 結果B |

「A/B」は誤差評価の違い。PF回路・Hkeepを変更する操作とは区別する。表のTruncated policyはそれぞれの評価で有限cohortから選択したため、選ばれたretention/h/orderが変わる場合がある。別途、**Bで選んだ同一回路をAで評価する対照**も取得した。結果A/Bという判定名と評価方法A/Bを混同しない。

全て同じ元Hのenergy精度1.5936e-4 Ha、方式1回のfailure上限.05、18round、δ0=.02、位相予算.4 radの課題。保存Hermitian元H・exact元ground oracle・理想native gate・fresh独立測定を仮定し、状態準備費用は除外する。新しい実機shot・価格MCは0。MC SEは再利用PR価格のnominal SEであり、RPE failureやhardware不確かさとは別である。

## 2. 再利用と固定科学条件

Linear H4/H6、隣接1.00 Å、STO-3G、RHF積分由来。H4は保存DF12・generation order、N=4 sector70・8 system qubits。H6は保存full XDF18・元frame順、Nα=Nβ=3 sector400・12 system qubits。元Hは保存行列のHermitian部分、DF再構成差も誤差に含める。推定対象をHkeepのground energyへ変更せず、既知E0で結果を補正しない。

Qiskit1.3.0、compiler A、rz/sx/x/cx、既にcontrolledなjoint回路、scalar相対位相・X/Y readoutを維持した。lower後に新しいcontrolを付けない。共通global phaseのみを許容する。詳細は[元REPORT][OLD]と[静的調査][SURVEY]。

主な再利用は`track_c_df_truncation`のpublic replay、DF係数/keep、modular native builder、directed qualification/supplement、独立expm/Schur/native X/Y、既存整数α配分とRPE decoder。既存SOURCEと合う箇所を利用し、**非dyadic hの時刻差の追加会計**と、**個別位相のdirected disk評価**だけを新adapterへ接続した。旧library本体・旧科学SOURCEは変更していない。

- 既存INPUT_ARRAYS/INPUT.jsonからH/state/B/sp・exact係数・gate IRをreplay。新SCF/DF fit・新初期状態生成は0。新しいcombined資格では既存ground residual補助関数（固有分解を含む）を使用するが、元H・stateは変更しない。
- 既存の同一点のcombined資格・source/native証拠・native価格を再利用。P1 n0は再simulation/compileしない。
- P2のHkeep spectral証拠16構成は個別誤差評価のため新たに生成し、H/eigenvalues/eigenvectorsとdirected証拠をrawへ保存した。
- H6のsignal計算は400次元sector、native照合は8192成分のfull controlled statevectorであり、途中でspin sectorへ射影しない。全Fock空間の巨大密行列やm17全回路は生成していない。

入力hash・実load SOURCE・環境・再利用refは[SOURCE.json](../../../artifacts/track_c_df_cancellation_validation/2026-11-08/SOURCE.json)、生データは[RAW_INDEX](../../../artifacts/track_c_df_cancellation_validation/2026-11-08/RAW_INDEX.json)、再現は[REPRODUCE](REPRODUCE.md)を参照。

## 3. P1：総時間固定の有限近傍

H6 / prefix13_I0 / S2 / 先頭2013（後続は元順）、m17を固定。`t=(1/50)×2^17`を先に固定し、n=32752,32767,32768,32769,32784についてh=t/nを有理数で定義、実装hはbinary64への一回の丸め。n0=32768は既存点を再利用し、新規4点だけを評価した。登録順序0213・元順の同一点は旧対照として保持し、新たに再計算しない。

| n | h=t/n（exact） | binary64 h | signed Trunc / PF / combined [rad] | combined上界 [rad] | rho下限 | β下限 [rad] |
|---:|---|---:|---|---:|---:|---:|
| 32752 | 4096/51175 | 0.080039081582804109 | +0.279809992 / -0.314122767 / -0.034312775 | 0.034895025353 | 0.999388576989 | 0.365065032661 |
| 32767 | 65536/819175 | 0.08000244148075808 | +0.279809992 / -0.313835179 / -0.034025187 | 0.034607613841 | 0.999388399695 | 0.365352444098 |
| 32768（保存点） | 2/25 | 0.080000000000000002 | +0.279809992 / -0.313816020 / -0.034006028 | 0.034588467134 | 0.999388387775 | 0.365371590801 |
| 32769 | 65536/819225 | 0.079997558668253532 | +0.279809992 / -0.313796863 / -0.033986871 | 0.034569322157 | 0.999388375941 | 0.365390735773 |
| 32784 | 4096/51225 | 0.079960956564177649 | +0.279809992 / -0.313509722 / -0.033699730 | 0.034282358070 | 0.999388198468 | 0.365677699746 |

signed値は点診断、combined上界・rho/βはdirected qualification。Truncation点値は固定Hkeep/state/tの旧データを再利用し、PF/combined点値は新しい実装時刻に対応するmeanから取得した。正本は[P1_RESULTS.csv](../../../artifacts/track_c_df_cancellation_validation/2026-11-08/P1_RESULTS.csv)、全予算・独立decoder・価格は`P1/FINAL.json`。

### 時刻・SOURCE/nativeの追加会計

元qualifierは主にdyadic hを前提としていた。今回は`round(t/h)`を比較の定義とせず、指定整数nを先に固定し、実装binary64の差を残す。

1. Dのlogical h=t/nとbinary64 hの差について、`|n h_bin−t_exact|`に保持Pauli係数のexact L1を掛けたSOURCE diskを追加。Gaussian conjugationのunitarityとfull number sectorのPauli normを使う。
2. 元qualifierのkernel内scalarはn h_bin、実際の長時間wrapperは固定総時間t_binを使う。その差を別のscalar相対位相diskとして加えた。**Hamiltonianの既知constantの処理であり、E0によるenergy補正ではない。**
3. 元のdecimal/binary target時刻、binary64角度、basis/source/native/readout、complete-number-sector supplementは継承。新hのfirst/middle/lastを実compileし、exact rational角度合算から一回の丸めを会計した。

時刻transferは保守的なdisk移送と追加absolute phase chargeを用いるため、tightest boundではない。SOURCE disk・scalar diskは上表の合成予算へ含める。初期の実装mean資格を保全し、logical hの追加会計後の最終値を採用した。raw `P1/FINAL_before_logical_h_supplement.json`は履歴であり、正本は`P1/FINAL.json`。

### 全18roundと安定性の意味

新4点を独立expm+complex Schur、full native X/Y作用で照合した。元policyの他17round・α配分をそのまま使用し、m17の十分整数Nを再評価。phase/qだけをdecoderへ渡し、元E0はenergy誤差の検証にのみ使う。全5点で18round mean decoderと固定α下の資格が閉じた。m15/m16の新しいh評価は不要だった。

| n | N_X / N_Y（元α固定） | 他17roundを固定した全18round総RZ | mean decoder誤差 [Ha] |
|---:|---|---:|---:|
| 32752 | 158 / 156 | 1226373992802 | 1.308928507e-05 |
| 32767 | 158 / 156 | 1226643828702 | 1.297957867e-05 |
| 32768 | 157 / 155 | 1222907255629 | 1.297227023e-05 |
| 32769 | 157 / 155 | 1222925130109 | 1.296496247e-05 |
| 32784 | 157 / 155 | 1223193247309 | 1.285542672e-05 |

n0より小さいnの2点では整数ceilによってNが1ずつ増えた。これは保存した有限点での小さいresource変化であり、元点の保証取消しや局所的な資格崩壊を意味しない。**結論は「指定した有限5点で安定性を支持」**であり、連続h領域・別retention/stateへの頑健性の証明ではない。P1の新規4点に不適格点はなかった。

## 4. P2：相殺を要求しない評価Bの根拠

同じ元groundについて、`zK=<ψ0|exp(-it Hkeep)|ψ0>`、`zP=PF/nativeのX/Y複素mean`を考える。

- **評価A**：元H理想位相に対するcombined信号を直接囲む従来の十分条件。
- **評価B**：Truncationの位相と、HkeepからPF/native meanへの位相差を**別々にdirected評価**し、その非負上界を加算する。

### 複素diskとbranch

新しいHkeep spectral certificate、形成・加算丸め、SOURCE generator、ground/state residualとnormalization、時刻差から、`zK∈D(cK,eK)`を得る。PF側は同じ回路のcombined資格にあるnumerical disk、SOURCE/native coordinate disk、complete-number-sector/target supplementから`zP∈D(cP,eP)`を得る。相関を都合よく差し引かない。

積`zP conj(zK)`は、中心`cP conj(cK)`と、次の半径で囲める。

\[
 e_{PK}\le |c_P|e_K+|c_K|e_P+e_Pe_K+e_{\rm product\ rounding}.
\]

中心積とその丸めも区間算術で囲む。diskが原点を除くことをdirected norm下限で確認し、中心のabsolute Argとdiskのangular radiusを加えて位相上界を得る。Truncation側には元E0のintervalも会計する。

\[
 B_T\ge |\arg(z_K e^{+iE_0t})|,\quad
 B_P\ge |\arg(z_P\overline{z_K})|,\quad
 B_B=\operatorname{upper}(B_T+B_P).
\]

Bはsaved signed点値をabs加算したものではない。nonzero diskと`B_B<.4<π`によって、判定に使う二つのArgのbranchを保持した保守的な角度評価とする。rhoは同じactual native meanの既存下限を利用し、`β_B=lower(.4−B_B)`から十分Nを求める。native/source等をP側diskへ含めているため、統計余裕から再度二重に引く必要はない。

元候補の非共鳴screen・dominant overlap診断も保存するが、それをdirectedなunwrapped Floquet energy定理へ格上げしない。小さいwrapped endpointだけでは判定せず、個別disk・各roundの予算・元Hのalias・全18round branch/decoderを確認した。continuous timeのenergy分枝の一般定理は主張しない。

### Hkeep SOURCEの移送と有限精度

S2/Y4の保存時刻列について、各blockのPF absolute time総和が正時間t以上であることをexact rationalで検査した。従って元PF資格にあるSOURCE generator diskは、Hkeepの正時間evolutionへの保守的な移送にも使える。complete-number-sector supplementを含めてspin leakageを課金し、Hkeepの数値Hermitian generator形成・総和丸めとspectral evaluation diskを追加する。PF native/readout/角度/有限bit誤差はP側に残す。

行列積・内積・transcendental・phase diskは既存の192-bit interval/spectral residualを再利用した。Hkeepの実装を量子回路へ追加したのではなく、**資格判定の参照信号を古典計算で追加した**ものである。回路価格へ架空のHkeep評価回路を課金しない。

## 5. P2候補の固定と負結果

P1完了後、新しい個別誤差を評価する前に`P2/PLAN.json`を保存。既存h/formula/order/retentionの保存scan、corrected資格、完全48点viewから、H4/H6の10,514 round点の有限poolを構成した。新retention・新PF family・P1の新hをP2へ追加していない。

事前規則は、元のTruncated資格点を全て含め、各分子・roundについて保守的proxy上位4点とcombined proxy上位2点を加えること。proxyは既存dominant bias/overlap・保存native価格から作った選択用の診断であり、保証には使わない。重複除去後157点（H4 74、H6 83）をdirected評価した。10,514点全てを新たに資格付けしたという意味ではない。

157点中154点がBでround資格を満たした。Bで閉じなかったのは次のH6 m17であり、いずれもAの資格はtrueのままである。

| H6 m17 / prefix13_I0 / h=.08 | 個別directed上界の和 [rad] | Aはclosedか |
|---|---:|---|
| S2 | 0.641723561654 | true |
| S2_order0213 | 0.613345045899 | true |
| S2_order2013 | 0.594300003431 | true |

この「不適格」は**評価Bの十分条件が閉じない**という意味。元Aの科学的成功、実際のPF誤差、物理的な相殺を否定しない。計算失敗・欠測を不適格や0費用に置き換えていない。全点のindividual/combined予算、signed診断、native価格は[P2_POINTS.csv](../../../artifacts/track_c_df_cancellation_validation/2026-11-08/P2_POINTS.csv)、raw `P2/qualified/`にある。

## 6. 全18roundのpolicyと総資源

選択は既存二開始点のcandidate/α交互選択とinteger allocationを再利用。36軸のα exact和≤.05、`N=ceil_upper[4 log(2/α)/(rho² sin²β)]`を確認した。有限cohortの探索でありglobal optimumの証明ではない。

### H4：Bで選択した18round

| m | retention / formula / h | directed BT / BP / sum [rad] | β下限 | N_X / N_Y |
|---:|---|---|---:|---|
| 0 | prefix7_I0 / S2 / 0.02 | 7.39313e-07 / 1.07203e-07 / 8.46516e-07 | 0.399999153 | 419 / 418 |
| 1 | prefix7_I0 / S2 / 0.04 | 1.47861e-06 / 8.55912e-07 / 2.33452e-06 | 0.399997665 | 419 / 419 |
| 2 | prefix7_I0 / S2 / 0.08 | 2.9572e-06 / 6.8489e-06 / 9.8061e-06 | 0.399990194 | 419 / 419 |
| 3 | prefix7_I0 / S2 / 0.16 | 5.91438e-06 / 5.49019e-05 / 6.08163e-05 | 0.399939184 | 419 / 419 |
| 4 | prefix7_I0 / S2 / 0.32 | 1.18288e-05 / 0.000442761 / 0.00045459 | 0.399545410 | 419 / 419 |
| 5 | prefix7_I0 / S2 / 0.64 | 2.36576e-05 / 0.0036481 / 0.00367176 | 0.396328244 | 426 / 426 |
| 6 | prefix7_I0 / S2 / 1.28 | 4.73155e-05 / 0.0317251 / 0.0317725 | 0.368227544 | 487 / 487 |
| 7 | prefix7_I0 / S2 / 1.28 | 9.46331e-05 / 0.0596751 / 0.0597698 | 0.340230231 | 538 / 537 |
| 8 | prefix7_I0 / S2 / 1.28 | 0.000189268 / 0.118664 / 0.118854 | 0.281146307 | 723 / 721 |
| 9 | prefix7_I0 / S2 / 0.64 | 0.000378537 / 0.0568829 / 0.0572614 | 0.342738570 | 455 / 455 |
| 10 | prefix7_I0 / S2 / 0.64 | 0.000757071 / 0.113734 / 0.114491 | 0.285509004 | 595 / 594 |
| 11 | prefix7_I0 / S2 / 0.32 | 0.00151414 / 0.0562645 / 0.0577786 | 0.342221405 | 382 / 382 |
| 12 | prefix7_I0 / S2 / 0.32 | 0.00302828 / 0.112528 / 0.115556 | 0.284443813 | 493 / 491 |
| 13 | prefix7_I0 / S2 / 0.16 | 0.00605657 / 0.0561126 / 0.0621691 | 0.337830853 | 315 / 314 |
| 14 | prefix7_I0 / Y4 / 0.32 | 0.0121132 / 0.0258125 / 0.0379257 | 0.362074338 | 245 / 244 |
| 15 | coeff_L1_2 / Y4_order0231 / 0.32 | 0.00546323 / 0.0466928 / 0.052156 | 0.347843962 | 236 / 236 |
| 16 | coeff_L1_2 / Y4_order0231 / 0.32 | 0.0109264 / 0.0933856 / 0.104312 | 0.295687928 | 275 / 274 |
| 17 | coeff_L1_2 / Y4 / 0.16 | 0.0218529 / 0.0130822 / 0.0349351 | 0.365064868 | 153 / 152 |

### H6：Bで選択した18round

| m | retention / formula / h | directed BT / BP / sum [rad] | β下限 | N_X / N_Y |
|---:|---|---|---:|---|
| 0 | prefix13_I0 / S2 / 0.02 | 2.13537e-06 / 1.86824e-07 / 2.32219e-06 | 0.399997678 | 422 / 422 |
| 1 | prefix13_I0 / S2 / 0.04 | 4.27029e-06 / 1.39428e-06 / 5.66457e-06 | 0.399994335 | 422 / 422 |
| 2 | prefix13_I0 / S2 / 0.08 | 8.54012e-06 / 1.10478e-05 / 1.95879e-05 | 0.399980412 | 422 / 422 |
| 3 | prefix13_I0 / S2 / 0.16 | 1.70798e-05 / 8.84232e-05 / 0.000105503 | 0.399894497 | 422 / 422 |
| 4 | prefix13_I0 / S2 / 0.32 | 3.41593e-05 / 0.000712699 / 0.000746858 | 0.399253142 | 423 / 423 |
| 5 | prefix13_I0 / S2 / 0.64 | 6.83188e-05 / 0.00586267 / 0.00593099 | 0.394069014 | 433 / 433 |
| 6 | prefix13_I0 / S2 / 1.28 | 0.000136641 / 0.0505229 / 0.0506595 | 0.349340489 | 541 / 541 |
| 7 | prefix13_I0 / S2 / 1.28 | 0.000273287 / 0.0964735 / 0.0967468 | 0.303253219 | 668 / 668 |
| 8 | prefix13_I0 / S2 / 0.64 | 0.000546577 / 0.0457773 / 0.0463239 | 0.353676134 | 457 / 457 |
| 9 | prefix13_I0 / S2 / 0.64 | 0.00109315 / 0.0916833 / 0.0927765 | 0.307223521 | 557 / 555 |
| 10 | prefix13_I0 / S2 / 0.32 | 0.0021863 / 0.0453147 / 0.047501 | 0.352499019 | 390 / 390 |
| 11 | prefix13_I0 / S2 / 0.32 | 0.0043726 / 0.090626 / 0.0949986 | 0.305001413 | 472 / 469 |
| 12 | coeff_L1_8 / S2 / 0.16 | 0.00229305 / 0.0451963 / 0.0474894 | 0.352510636 | 320 / 320 |
| 13 | coeff_L1_8 / S2 / 0.16 | 0.00458612 / 0.0903925 / 0.0949786 | 0.305021372 | 379 / 379 |
| 14 | coeff_L1_8 / Y4 / 0.32 | 0.00917642 / 0.0400582 / 0.0492347 | 0.350765334 | 262 / 262 |
| 15 | prefix14_I1 / Y4 / 0.32 | 0.00338077 / 0.0801762 / 0.083557 | 0.316443046 | 279 / 278 |
| 16 | prefix14_I1 / Y4 / 0.16 | 0.00676153 / 0.0108848 / 0.0176463 | 0.382353652 | 165 / 164 |
| 17 | prefix15_I0 / Y4 / 0.16 | 0.00178651 / 0.0216446 / 0.0234311 | 0.376568896 | 147 / 145 |

省略表示の数値は表の可読性のための丸めであり、正本は[P2_ROUNDS.csv](../../../artifacts/track_c_df_cancellation_validation/2026-11-08/P2_ROUNDS.csv)、`P2/POLICY.json`と`P2/FINAL.json`。各roundのexactなBT/BP、source/native/diskはselected pointのJSONとP2_POINTSへ残す。全roundが単一Hkeepという制限は課さず、retention切替を明示する。

| 分子・policy | 総RZ | 総CX | 総size | serial depth上界 | 十分trial総数 |
|---|---:|---:|---:|---:|---:|
| H4 A_combined | 3.337772238e+11 | 5.830379151e+10 | 4.941682283e+11 | 1.188928069e+11 | 14629 |
| H4 B_conservative | 3.864575491e+11 | 6.802588264e+10 | 5.724967151e+11 | 1.385172409e+11 | 14825 |
| H4 A_on_B_fixed_circuits | 3.444551851e+11 | 6.063755572e+10 | 5.102746551e+11 | 1.234796243e+11 | 14459 |
| H4 reference_Full | 5.647934021e+11 | 1.012313194e+11 | 8.371684838e+11 | 2.088520659e+11 | 14225 |
| H4 reference_PR | 1.303898609e+12 | 1.824266292e+11 | 1.904338949e+12 | 4.992174341e+11 | 16791 |
| H6 A_combined | 1.222907256e+12 | 2.106713327e+11 | 1.809978048e+12 | 3.347103693e+11 | 13539 |
| H6 B_conservative | 1.868265339e+12 | 3.249293075e+11 | 2.767213795e+12 | 5.159556638e+11 | 14353 |
| H6 A_on_B_fixed_circuits | 1.863969348e+12 | 3.242093835e+11 | 2.760865644e+12 | 5.148324004e+11 | 14218 |
| H6 reference_Full | 2.082026044e+12 | 3.657833912e+11 | 3.085775668e+12 | 5.917635392e+11 | 14248 |
| H6 reference_PR | 1.617753939e+12 | 2.708191202e+11 | 2.389309688e+12 | 4.422261561e+11 | 15754 |

RZ優先で選んだ**同じ実native module**のCX/size/depthを使う。RZ/CX/sizeは加法値、depthはserial連結上界。native scalarとX/Y wrapperを含む。Full/PRの新しいretuning・価格MCは行わず、同じ科学条件で資格済みの元policy・PR独立holdoutを再利用した。PRのweighted estimatorを無重み測定へ置換していない。

選択とα/Nを凍結してから、P2では新規6点を独立expm/Schur/full native X/Yで照合。保存の同一点はkeyとfirst/last実価格の一致を確認して再利用した。uniqueなP2照合点は43。新しい価格はdeterministic countなので新MC holdoutは不要。既存PRのnominal価格SEを新しいRPE failureへ混ぜない。全6 policyのmean decoder・alias・十分Nとα予算が成立した。

## 7. 限定ablation：評価規則と実回路の区別

Aで自由選択したpolicyに対し、Bで選んだ同じ回路をAで評価する`A_on_B_fixed_circuits`を別に凍結した。Bとこの対照は、全18roundのkey・native単価が完全一致し、誤差評価とそれに基づくα/Nだけが変わる。

| 分子 | A自由選択 RZ | B回路をA評価 RZ | 同じB回路をB評価 RZ | 回路policy変更の経路差 RZ | 同一回路の評価規則差 RZ |
|---|---:|---:|---:|---:|---:|
| H4 | 333777223814 | 344455185076 | 386457549139 | 10677961262 | 42002364063 |
| H6 | 1222907255629 | 1863969347635 | 1868265339430 | 641062092006 | 4295991795 |

この表は`A自由選択 → B回路をA評価 → 同じB回路をB評価`という**順序付きの比較経路**。唯一の因果分解ではなく、policy選択とα最適化の相互作用が残る。存在しないcounterfactualを補間していない。

H4ではnear-zeroのcombined点に依存せず、元順Y4・coeff_L1_2などのB-qualified policyで両参照方式より低コストとなった。H6ではAの採択m17をBへそのまま移すことはできず、Bは切り捨てを抑えたprefix15_I0/Y4等へ切り替える。同じB回路をAで評価しても、元Aの総RZへは戻らない。Bの余分なNだけで全差を説明することはできない。

切り捨てによるmodule構造の価格減少、相殺によるβの余裕、h/order/retention変更、α/N配分は別の寄与である。H6は保守的評価でもFullを下回るがPRより高い。これは今回の保守的上界・cohort・選択の結果Bであり、他候補のno-go定理や「Truncatedは実際には高コスト」という主張ではない。

## 8. 条件・精度・技術履歴

- P1の5点、P2の6 policyで元Hのmean energy誤差は指定精度内。E0は診断・資格検証だけに使い、decoder入力はphase/qだけ。位相予算からの最終energy幅は元研究と同じ1.52587890625e-4 Haで、精度を緩めていない。
- 保証は固定保存H、元ground oracle、理想native、独立測定という条件付き。実際のmeasurement repetition/量子backend、state prep、hardware noise、T合成、routingの評価ではない。
- 新adapterの人工testと既存回帰は13 PASS。scalar時刻差、rational hのSOURCE disk、複素積disk、逆符号の相殺を別上界へ持ち込まない例を含む。人工PASSと分子の性能結論を区別する。
- 最初のP2 keep呼出しはPLAN完成前のFileNotFoundErrorで、科学keep call前に終了。logを保全し、PLAN完成後の実行を正本とした。計画のproxy算出中は停止せず、完成済みpoolを再計算しなかった。後で不変wrapperのcacheを追加して再現時の重複処理を除いた。
- P1のlogical h丸めの追加会計では、元の4点のdirected計算・native検査を再実行せず、保全した結果へ追加diskを移送した。元の中間FINALも保全。
- exact-ground/sector、既存SOURCE・input・raw・grant・ledger・marker・STOPは変更しない。全処理はprimary agentのみ。CPU affinity64、利用可能RAMを確認し、科学workerはtask数とaffinityに応じ展開。BLASは各worker1threadで、OS・共有環境・他者processは変更しない。

## 9. 完了・未確定事項・STOP

**P1：指定5点で安定性を支持。P2：H4は結果A、H6は結果B（Fullより安いが、今回の保守的選択ではPRより高い）。** Aの既存保証・低コスト結果は維持する。

未確定なのは、連続h領域の頑健性、cohort外や全poolでの大域最適性、保守的diskのさらなる鋭さ、物理的な近似ground入力・別状態への移送、hardware費用である。特に「相殺なし」の意味は今回の個別位相上界の加算による十分評価であり、物理的相殺を回路から取り除いた実験ではない。

[SOURCE](../../../artifacts/track_c_df_cancellation_validation/2026-11-08/SOURCE.json)、[保存結果](../../../artifacts/track_c_df_cancellation_validation/2026-11-08/RESULTS.json)、[検証状態](../../../artifacts/track_c_df_cancellation_validation/2026-11-08/VALIDATION.json)、[最終check](../../../artifacts/track_c_df_cancellation_validation/2026-11-08/FINAL_CHECK.json)、[raw index](../../../artifacts/track_c_df_cancellation_validation/2026-11-08/RAW_INDEX.json)と[再現手順](REPRODUCE.md)を公開する。

**P1・P2の合理的な検証を完了し、mandatory STOP。** 次の担当はユーザー：研究結果の採否と後続研究の要否を判断する。P3/P4、無制限retention探索、別分子、H8以上、新PF/sampling family、詳細GPT科学レビューへ自動進行しない。

[OLD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/docs/research/track_c_df_truncation_20261108/REPORT.md
[SURVEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/12e74dd68568f493ed39937f8856f26ce48d33e1/docs/research/track_c_df_cancellation_evidence_20261108/REPORT.md
