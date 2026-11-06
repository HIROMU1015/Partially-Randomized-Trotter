# Track C：compiler Aの18round誤差会計

2026-11-06 JST、SERVER 2 `GPUA100-gleap02`。
基準commit：`f45e9b07bf730f7272933b28055914cfe76fdaab`。
専用branch：`codex/track-c-compiler-a-18round-qualification-20261106`。

**compilerの局所区間上界を全support・18roundへ接続すると、保存SOURCE meanモデルの下ではS2 LD3とFull Y4 LD12の全36条件で正の半径と統計位相予算が残った。PR＋RPE全体の科学的資格はUNQUALIFIEDとする。**
compilerが予算を使い切る結果ではない。保存されたPF/meanの数値disk、生成SOURCEと解析meanの対応、外部alias前提を数学的保証へ格上げする証拠が不足している。
局所上界、条件付き数値判定、最終保証を区別して保存した。価格取得・T合成・旧run01再開は0件。承認範囲完了でmandatory STOP。

## 1. 固定条件と実行範囲

Linear H4、隣接1.00 Å、STO-3G、保存DF rank12、生成順LD3のpartial S2とFull LD12のY4、h=.02。
元の物理Hの厳密groundを供給する仮定、状態準備費除外を維持する。
`epsilon_E=1.5936e-4 Ha`、全失敗確率`.05`、全位相予算`.4 rad`、`delta0=.02`、m=0..17・q=2^m。
S2は既存`r_m=[1]*8+[2]*2+[4]+[8]*2+[16]*2+[32]*3`、finite-RTE K=2。
Full Y4にはRTEがなく、全stageの有効rは0。保存行に残る登録r fieldとは分けて記録した。

[前段の人工44ケース・H4 q1結果](../track_c_compiler_a_correctness_20261106/REPORT.md)を引き継ぎ、再実行していない。
SCF・DF fit・ground solve・H4入力照合も実施していない。
旧専用runtimeと準備cacheを読み取り専用で再利用した。[SOURCE/input指紋][source]、[環境][environment]。

今回追加したのは、局所gate catalogue、contextを考慮した同じAによるlowering、最大8×8の局所区間評価、構造加算、保存36行への誤差・shot会計である。
full H4 q1や長roundの回路build/compile/simulation、fresh sampling、C-S/C-P追加実験はない。
本計算は128本の3-qubit局所context回路をcompileした。その前に1本の局所context確認probeを実施した。
compiled gate列は誤差の証拠として保存し、回路価格・RZ/CX/T/depthの資源集計には使っていない。

128独立contextをOS CPU affinityの64 workerで並列処理し、BLASの多重poolは避けた。
worker数は`min(task数, CPU affinity)`であり恒常的な独自上限ではない。
共有パッケージ・OS設定・他ユーザーのプロセスには変更を加えていない。
主計算は約6.9秒。ローカル開発worktreeでの実行であり、immutable CIや外部再現ではない。

## 2. 成立した関係と保証の位置

| 分類 | 今回の根拠 | 到達した範囲 |
|---|---|---|
| 数学的関係 | unitary積のtelescoping、共通joint phaseの不変性、Frobenius normがspectral normを上から押さえること | 局所差上界が有効なら積全体の差を加算可能 |
| 局所の計算機支援上界 | binary64入力を正確に取り込み、mpmath 1.3.0の192-bit directed intervalで1,302変換を評価 | 理想primitive（保存された入力角）からnative ideal-gate列までの差。区間演算実装の正しさを前提とする |
| 全supportへの構造移送 | 固定SOURCEのgate family、全128入力basis集合、K2の最大event構造、実r/負時間recipe | 同じcompiler/equivalence環境でのfull・strong共通の保守的上界 |
| RPEの条件付き数値会計 | 保存PF/RTE/input/numerical行を維持し、新diskをraw信号へ追加 | 両方式全18roundで正半径・正統計予算。shot計算と分解能条件を確認 |
| PR＋RPE全体の最終保証 | 保存meanと生成SOURCEの完全な誤差対応、rigorous PF/数値disk、base-windowが不足 | **UNQUALIFIED**。成功確率95%を無条件には認定しない |

q1の観測誤差をq倍した値は使っていない。
また、float64のdense matrix積で観測されるunitarity defectを、そのまま物理native gateの誤差として加算していない。
nativeのRZ/SX/X/CXは理想unitaryとして扱う。backend noise・T合成は今回の対象ではない。

## 3. 局所変換とcontextの網羅

参照Aのpass列と既存SOURCEを変更していない。
固定gateの正確な逆相殺、UnrollCustomDefinitions、BasisTranslator、固定X/CX逆相殺を使い、近似的な交換・小角省略・数値再合成は使わない。

準備済みDおよびtail basisのruntime gateはRZとXXPlusYYであり、実inverseも含めた全入力パラメータを列挙した。
正負のY4 stage、D diagonalのcontrolled RZ/RZZ、S2の全r・order0/2・係数の両符号を含む。
tailには218 component、identity componentは0、支持はZまたはZZである。

controlled RZZのUnroll後にはCRZとCCXが現れる。
入力basisの非native familyは`xx_plus_yy, crz, ccx, p, h, sdg, cz`。
full graphとstrong piecesで入力集合が異なるため、7 familyの全128部分集合を実際にlowerした。
各caseの後にbarrierを置き局所出力を分離し、Unroll後のfamily集合が指定contextと一致することを確認した。
native RZ/CXとbarrier/measureのpassthroughは変換探索の対象から除かれる。

Qiskit公式1.3.0の固定Rust SOURCEでは、[basis抽出と局所適用][rust-main]、[gate別symbolic mapの構成][rust-compose]、[target basisを除いた探索][rust-search]を確認した。
入力gateのname/arityとパラメータ数を用いたmapを作り、各gateへ代入する構成なので、固定環境のcontextを局所回路で調べる方法を採った。
全contextを通じ、各入力variantに対するnative列は1通りとなり、重複を除く1,302列を保存した。
実行wheelと公式Rust SOURCEの対応は版・既存provenanceを前提とし、wheel再buildによる認定はしていない。[対応条件][transfer]。

barrierを除いたfull回路で追加され得る相殺は、Aに含む固定gateの正確な逆相殺である。
数値小角の近似削除が復活する経路はない。この構造移送は、別pass・別equivalence登録・別topologyへは自動適用しない。

### phaseの扱い

全gateは必要なcontrolを付けた後にlowerする。出力へ後から量子controlを付ける用途は今回も未資格。
既にjoint回路に埋め込まれた各局所gateの共通phaseは、積全体でも共通joint phaseになるので許容できる。
control1側だけの独立phase fitは行わない。

局所差は、一つのbinary64角gammaで表す厳密なunit複素数`exp(i gamma)`を共通gaugeとして評価した。
gauge推定を誤差ゼロと仮定せず、その有限精度角も区間評価へ含めた。
したがって、物理的なcontrolled相対位相の差をfitで消す手順ではない。

P gateは全contextでsymbolic入力alphaが単一の`RZ(alpha)`へ変換されることを確認した。
恒等式`P(alpha)=exp(i alpha/2) RZ(alpha)`により、ancillaの相対位相を保ったまま、任意の保存入力角を一つの共通joint phaseの差だけで扱える。
Pの入力を生成するときの丸めは、このcompiler恒等式とは別に扱う。

## 4. 区間上界と構造加算

mpmathのinterval実装は加減乗除・三角関数・平方根を上下方向へ丸める。
実数binary64角は192-bitで正確に取り込み、source gateの解析式とnative列を複素区間で評価した。
RZの位相、exact SX行列、X/CX permutationを使い、差のFrobenius norm上端をbinary64へ外向きに出力した。
中間の誤差加算・整数倍も区間内で行い、最後のfloat変換だけを外向きにする。

局所差上界の最大は **`1.1467503820272306e-15`**。
この値はq1の全H4観測誤差でも、新しいcompiler予算の割当てでもない。
1,302 source/native小行列の独自評価器をQiskitの独立したOperator実装へ照合し、最大成分差はsource約`1.11e-16`、native約`1.05e-15`だった。
これは評価器の数値実装検査であり、区間上界そのものをfloat64観測値で置き換えてはいない。
意図的にCRZをidentityとしたcaseを上界受理できないこと、binary64入力を点区間で保持することも検査した。[技術検査][checks]。

局所primitive誤差をe_gとすると、実現xのfull unitary誤差は

`d_mx <= sum_g e_g`

で上から押さえられる。Dは実際の各符号stageと両halfの登録gate出現数を加算した。
S2 tailは各eventのorderが0/2なので、最大3回のbasis conjugation、最大4個のCZ、1個のcontrolled rotationで押さえる。
各componentのbasisとその実inverseの誤差和をbとし、全登録basisの最大bを使う。
同basis再利用で省略される正確な逆対はこの上界を増やさない。
これを実r、stage、反復数へ加算し、最後にX/Y wrapperの最大側を加えた。
このため、乱数trajectoryや巨大回路を列挙せずに全supportを覆う。
strong baselineも同じAと同じprimitive集合の境界分割なので、同じ上界が有効である。actual fullとstrongの価格や出力が同一だという主張ではない。

### SOURCE内のevent相対phase生成

既存RTE builderは各event phaseと負係数product phaseを浮動小数点で累積している。
identity componentがない今回のK2 supportでは、1 occurrenceのr=nについて非ゼロ項は最大3n個のpiである。
compiler変換とは別に、`k=3n, u=2^-53, gamma_k=ku/(1-ku)`として

`e_phase <= k abs(pi_float-pi) + gamma_k k abs(pi_float)`

を会計した。これはIEEE binary64・round-to-nearest、overflow/underflowなしの有限pi加算の標準誤差評価である。
解析meanを評価する旧70次元matrixの数値diskへ、このbuilder内部のphase加算は含まれていないため、別項で追加した。
Full Y4にはこのRTE生成phase項はない。
Gaussian基底の実現、D/atanパラメータ生成、sampling分布normalizationの全誤差まで、この項に含めたとは主張しない。

## 5. 全18roundの位相・半径・shot会計

[全36行のTSV][table]と[ROUNDS.json][rounds]に、各mのcompiler上界、SOURCE phase上界、raw disk、半径、残り位相予算、shot数を記録した。
compilerとSOURCE phaseのdiskは共にraw native trajectory meanへ加える。
各trajectoryの測定rangeは1で、正の分布の平均なので

`Delta_comp <= 2 sqrt(2) sup_x d_mx`

を使う。raw信号への加算に、さらにinverse attenuationを掛けていない。
corrected信号へ換算する場合は信号とdiskの両方を同じBで掛け、rangeもBへ変更する必要がある。
normalizationの二重適用やraw/corrected単位の混在はない。

旧raw半径下限rho0から新disk Deltaを引き、`rho_new=rho0-Delta`。
`Delta<rho0`のとき、`beta_add=asin(Delta/rho0)`を旧beta_statから差し引いた。
旧PF・finite-RTE・input/numerical項は維持し、compiler項をそれらへ再度含めない。

| m=17、q=131072（今回の最大追加誤差） | S2 LD3 | Full Y4 LD12 |
|---|---:|---:|
| compiler operator上界 | 6.1637974e-7 | 6.7272367e-7 |
| SOURCE event phase生成operator上界 | 4.2286137e-7 | 0 |
| raw compiler disk | 1.7433852e-6 | 1.9027499e-6 |
| raw event phase生成disk | 1.1960326e-6 | 0 |
| 追加位相幅 rad | 5.1325487e-6 | 1.9027499e-6 |
| 新raw半径下限 | 0.5726984460 | 0.9999980613 |
| 新beta_stat rad | 0.3860064103 | 0.3999948566 |
| 各軸shot数（条件付き） | 626 | 192 |

全18roundでrho_newとbeta_statが正だった。
S2の18round・両軸合計shotは **8,598**、Full Y4は **6,912**。
今回の上界を追加しても、整数化したshot数は旧一様配分と全軸で一致した。これは総コスト順位ではない。

alpha_axisを厳密な`1/720`、36軸の合計を`1/20=.05`として、range1の各軸で

`eps = rho_new sin(beta_stat)/sqrt(2)`、`N >= 2 log(1440)/eps^2`

を区間評価し、必要Nを整数へ切り上げた。
Hoeffding・union boundとRPEのbranch条件が適用できるモデル下での十分条件であり、shotを新たにsamplingした実測成功率ではない。

18roundの分解能条件は`0.4/(0.02*2^17)`を区間評価し、上端`0.00015258789062500003 Ha`が目標`0.00015936 Ha`を下回る。
局所branch連結の`3 beta < pi`も確認した。
base phaseを一意に決める外部window/endpoint marginは保存結果の仮定を引き継ぎ、今回新たに認定していない。[RPE条件][rpe]。

## 6. UNQUALIFIEDの内訳と次の最小作業

**Aがcompiler誤差予算を超過した、という負結果ではない。**
保存meanの条件下ではcompiler寄与を含め全roundに余裕があり、Aへの追加の数値最適化やprofile変更は今回の結果からは必要とされない。
一方、次の鎖を埋めずにPR＋RPE全体を数学的な成功保証へ格上げすることはできない。

1. **保存PF/source meanの数値誤差**：旧`signal_task`の`numerical_disk`は`conditional_float64_model`と明記された診断。今回の区間上界で旧PF位相値・原H参照・state誤差を逆に保証したことにはならない。
2. **生成SOURCEと解析meanの対応**：保存Gaussian basisの実現、D diagonalとatanの入力角生成、有限RTE sampling確率・normalizerが、解析meanへどの誤差で接続するかを一括では認定していない。event phase項とcompiler項だけで全生成誤差を覆ったとは扱わない。
3. **alias/base-window**：保存結果は外部windowと`.4 rad` endpoint marginを仮定する。再帰の局所不等式と分解能だけでは初期aliasの一意性は保証しない。
4. **compiler移送の適用条件**：固定Qiskit1.3.0・equivalence環境・pass列・gate集合に限定。未知のcustom gate、別layout、lower後の新しいcontrolへ一般化しない。

必要な最小追加検証は、保存DF/groundを再利用したまま、SOURCE→解析meanの基底・角度・分布誤差を区間または明示上界で閉じ、既存PF/入力数値diskの保証範囲とalias windowを確定すること。
event phase生成だけをさらに抑えたい場合はpi parityの整数管理が候補になるが、今回の上界は予算内なので実装していない。
既存SOURCEを無断変更せず、新しいprovenanceと許可範囲で扱う。

価格評価へ進むには、全36条件で有効なSOURCE/mean・compiler・PF/RTE・alias・統計条件を同じ保証水準で接続する必要がある。
条件付きpilot価格として進める判断をする場合も、その証拠水準と未資格事項の採用はユーザー判断であり、今回の結果だけで採択・価格開始しない。
いずれの場合も新価格cohortは同じAを候補と強baselineへ適用し、旧opt2価格と混ぜない。

次の担当：**ユーザー**が残る保証水準・SOURCE→mean/alias検証の範囲を判断して実行許可する。
**Codex**はその許可後に必要最小限を実施する。今回の価格取得、T合成、C-S/C-P追加、3pilot再開は行わない。

## 7. 成果物・保全・STOP

- [新validation SOURCE](../../../src/trottertracks/track_c_compiler_a_18round.py)、[runner](../../../scripts/run_track_c_compiler_a_18round.py)。参照Aと旧科学SOURCEは不変更。
- [SUMMARY][summary]、[全36行][rounds]、[TSV][table]、[RPE条件][rpe]。
- [局所証拠archive][evidence]と[member hash一覧][archive]：catalogue、128 context、構造出現数、1,302 native列、局所区間上界をlossless保存。
- [評価器照合][checks]、[算術test][tests]、[SOURCE/input][source]、[環境][environment]、[実行log][log]、[STOP][stop]。

主計算1回、保存局所結果に対する評価器照合と補助RPE算術を実施した。
最終runnerには同じ補助出力も統合したが、完成後のend-to-end再実行はしていない。
既存人工44ケースや旧H4 q1 suiteを再開したものではない。旧raw・grant・ledger・marker・STOPを保全した。
**承認範囲を完了しmandatory STOP。science workerは終了済み。価格・3pilotへ自動継続しない。**

[source]: ../../../artifacts/track_c_compiler_a_18round/2026-11-06/run01/SOURCE_INPUTS.json
[environment]: ../../../artifacts/track_c_compiler_a_18round/2026-11-06/run01/ENVIRONMENT.json
[summary]: ../../../artifacts/track_c_compiler_a_18round/2026-11-06/run01/SUMMARY.json
[rounds]: ../../../artifacts/track_c_compiler_a_18round/2026-11-06/run01/ROUNDS.json
[table]: ../../../artifacts/track_c_compiler_a_18round/2026-11-06/run01/ROUND_TABLE.tsv
[rpe]: ../../../artifacts/track_c_compiler_a_18round/2026-11-06/run01/RPE_CONDITIONS.json
[checks]: ../../../artifacts/track_c_compiler_a_18round/2026-11-06/run01/MATRIX_MODEL_CHECKS.json
[tests]: ../../../artifacts/track_c_compiler_a_18round/2026-11-06/run01/ARITHMETIC_TESTS.json
[evidence]: ../../../artifacts/track_c_compiler_a_18round/2026-11-06/run01/local_rule_evidence.tar.xz
[archive]: ../../../artifacts/track_c_compiler_a_18round/2026-11-06/run01/EVIDENCE_ARCHIVE.json
[transfer]: ../../../artifacts/track_c_compiler_a_18round/2026-11-06/run01/UPSTREAM_TRANSFER.json
[log]: ../../../artifacts/track_c_compiler_a_18round/2026-11-06/run01/RUN.log
[stop]: ../../../artifacts/track_c_compiler_a_18round/2026-11-06/run01/STOP.json
[rust-main]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/basis/basis_translator/mod.rs
[rust-compose]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/basis/basis_translator/compose_transforms.rs
[rust-search]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/basis/basis_translator/basis_search.rs
