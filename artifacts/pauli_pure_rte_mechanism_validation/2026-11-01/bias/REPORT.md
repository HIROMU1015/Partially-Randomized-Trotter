# H6 Pauli-JW P1：複素bias原因分離の限定結果

2026-11-01 JST。**8伝播を正常終了し、未解像成分を保持して STOP_UNRESOLVED_COMPONENTS。追加計算・retry・高精度化・別solver・P1N/P2/P3には進んでいない。**

固定条件の数値診断では、Hybridのr16/32/64に共通する大きな信号差が保持／tailの外側対称分割で現れた。保持78項は可換なI/Zであり、BとCは1e-10の整合目安内で一致した。有限Taylor項はH16とR64で診断分解能を上回ったが、H32/H64の小さい有限項と全候補のgrid差はUNRESOLVED。これらを0や一意な因果寄与率へ置換しない。

これは主agentによるP1の実装・数値検証であり、独立科学レビュー、主要claim採択、実compiled精度・母平均費用・新しい方式順位の認定ではない。

## 1. 条件・入力・SOURCE

linear H6、隣接1Å、STO-3G、中性singlet、保存Pauli-JW918非identity項＋constant、12system qubit、T=.8、S2、q8、K2（理想多項式次数3）、ε_sig=.0005を確認対象とした。全4096次元、保存raw φ・exact ν、保存full-H referenceを使い、sector射影も状態の再正規化もしない。

|ID|保存候補|今回の用途|
|---|---|---|
|H16|PR_p0078_q8_r16_K2|Hybrid r系列|
|H32|PR_p0078_q8_r32_K2|保存Hybrid family best。新しい選抜ではない|
|H64|PR_p0078_q8_r64_K2|R64との同q/r/K対照|
|R64|H6_p000_q8_r64_K2|純RTE L0|

H32対R64はrも違うfamily比較、H64対R64はq/r/K matchedである。このP1ではどちらのN/C/Gも再会計・再ランキングしていない。

実行前に公開・local tree一致を確認した科学SOURCEは [a0a3aff915f98456733d6bf09ed641a1be423628](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a0a3aff915f98456733d6bf09ed641a1be423628)。親は設計資料6f0759df7c1df7fd6d835922b0246753a8a98d6c。実行後、SOURCEの5file hashを再照合して一致した。

[SOURCE・入力・reference・archive memberの固定hash](source_and_inputs.json)に、Hybrid結果00958ae1／SOURCE887b69e4、純RTE結果10cb3e5c／SOURCEe755ecf5、model/reference、保存4law/targetの所在を記録した。過去の記録済みcode hashと選択member bytesを照合済み。旧科学SOURCE・入力・raw・result・grant・ledger・marker・STOP・元branch/mainは変更していない。

Aは保存physical_referenceをそのまま読んだ。Dgridは保存physical_corrected_meanを読んだ。A/DgridへBまたはνを再適用せず、保存target/law/B/bias/u/N/C/Gの再評価は0。新B/C/Dpolyだけを〈φ|U|φ〉/νで評価し、νで一度割った。DpolyへBは掛けていない。新full-H reference・SCF/FCI/JW/DF生成は0。保存係数からのH_D/H_R sparse組立は今回の内部処理である。

## 2. B=C、L0、数値整合

- Hybridの |Z_C−Z_B| = 7.731011545855e-15。1e-10の整合目安を通過した。
- tail exponentialを二つのhalf actionへ分けた自己整合差は、Bが 9.420554752103e-16、Cが 9.485749680535e-16。これは同じSciPy solverのsemigroup整合であり、独立した誤差証明ではない。
- L0のB=C=Aは、H_D=0・tailが全非identity項・constant一回という構造的恒等式として確認した。実H6の独立B/C再伝播や新full-H referenceは行わず、保存Aを同じfunctionalのaliasにした。L0の独立dense oracleとの数値照合は人工2qubitで行った。
- 全4候補で複素閉合残差は保存出力上 0+0i。これは同じ中心値のtelescoping閉合で、独立したaccuracy証明ではない。
- scalarは元のforward→tail→reverseの後で各q stepに一回、負時間・係数符号・順序は人工oracleで検査した。位相alignmentで誤差を除去していない。実compiled残差ρは未評価のまま。

## 3. 符号付き複素差と数値分解能

新信号には各軸1e-10という事前設定の**engineering診断分解能**を使った。厳密誤差上界・信頼区間ではない。保存A/Dgridについては既存包含の半幅とbinary64中心への変換分を読み、差の軸別分解能は両入力分を加えた。複素差の絶対値が保存されたradial分解能以下ならUNRESOLVEDとした。以下の小さい点値の表示桁は、その桁まで有意という意味ではない。

Δ_outer=B−A、Δ_hold=C−B、Δ_trunc=Dpoly−C、Δ_grid=Dgrid−Dpoly。実部はcosine、虚部は+Im sine。

### Hybridに共通する2成分

| 成分 | 符号付き複素差 | 絶対値 | 軸別分解能 Re / Im | 数値status |
|---|---|---|---|---|
| Δ_outer（H16/H32/H64共通） | 7.016398502702e-05 + 1.136269663863e-04 i | 1.335442708806e-04 | 1.13428e-10 / 1.13428e-10 | RESOLVED_EMPIRICAL |
| Δ_hold（同上） | -6.328271240363e-15 + 4.440892098501e-15 i | 7.731011545855e-15 | 2.00000e-10 / 2.00000e-10 | UNRESOLVED |

保持I/Zの可換性から理想Δ_holdは0であり、観測点値はそのcorrectness確認に整合する。数値差そのものは分解能以下なので0と測定確定したとは言わない。H_DとH_Rの可換性を示したわけでもない。

### 候補別の有限Taylor・実効law差

| 候補・成分 | 符号付き複素差 | 絶対値 | 軸別分解能 Re / Im | 数値status |
|---|---|---|---|---|
| H16 Δ_trunc | 6.388587436845e-10 − 3.971583062423e-10 i | 7.522467777262e-10 | 2.00000e-10 / 2.00000e-10 | RESOLVED_EMPIRICAL |
| H16 Δ_grid | 2.242650509743e-14 − 1.021405182655e-14 i | 2.464295001822e-14 | 1.20571e-10 / 1.20571e-10 | UNRESOLVED |
| H32 Δ_trunc | 7.994338524497e-11 − 4.949962661982e-11 i | 9.402743152893e-11 | 2.00000e-10 / 2.00000e-10 | UNRESOLVED |
| H32 Δ_grid | -4.107825191113e-15 + 4.218847493576e-15 i | 5.888370060958e-15 | 1.16497e-10 / 1.16497e-10 | UNRESOLVED |
| H64 Δ_trunc | 9.993894600768e-12 − 6.177613975922e-12 i | 1.174907842032e-11 | 2.00000e-10 / 2.00000e-10 | UNRESOLVED |
| H64 Δ_grid | 2.786659791809e-14 − 1.587618925214e-14 i | 3.207180482446e-14 | 1.15539e-10 / 1.15539e-10 | UNRESOLVED |
| R64 Δ_trunc | 7.756000952597e-09 − 4.823580646018e-09 i | 9.133590806760e-09 | 1.13428e-10 / 1.13428e-10 | RESOLVED_EMPIRICAL |
| R64 Δ_grid | -4.507505479978e-14 + 1.953992523340e-14 i | 4.912809016571e-14 | 1.25613e-10 / 1.25613e-10 | UNRESOLVED |

R64のΔ_outer/Δ_holdは構造的alias差0で、新しい測定値ではない。JSONの一般差分routineが同じ保存Aの区間幅を保守的に二重加算していることは、相関するX−Xに非zero誤差があるという主張ではない。

各差の元のbinary64中心、絶対値、軸別/radial分解能、閉合は [signals_and_signed_differences.json](signals_and_signed_differences.json)、新8伝播は [new_signals.json](new_signals.json)、整合は [numerical_checks.json](numerical_checks.json)。保存A/Dgridの新科学結果としての再発行はせず、出所をSOURCEへ結合した。

## 4. 固定条件で説明できたこと・残るもの

Hybrid3候補でB/Cは共通で、外側分割の信号差が既存の残存biasのスケールに対応する。保持78項の内部PF誤差という説明は、この可換保持モデルの理想演算には当てはまらない。rを増やしてもこの外側分割は同じq8のまま残る。これは固定の分解経路に沿った数値観測であり、保持とtailの選択を変える独立介入の寄与率ではない。

H16の有限項とR64の有限項は事前の診断分解能を超えた。H32/H64の点値は小さいが未解像なので、減少率・符号・厳密上界を採択しない。全grid差も未解像であり、CDF/角度/正規化の差がゼロとは言えない。grid差には数値評価・binding政策の違いも含まれ得る。実compiled残差ρの保証へ読み替えない。

既存のheadroom/N会計を参照する上では、rだけを増やしてもq8の共通残差が残る理由を説明する材料になった。ただし新しいheadroom値、仮想N、G、SE、方式順位、因果寄与率は計算していない。単回価格CやDF/Pauliの物理分割についても本P1から新結論は得ない。

## 5. 再利用・最小拡張・人工検査

既存PauliHamiltonian.sparse_operatorとsparse_mean_actionのpaired recurrenceを再利用した。degree0…K+1の短いrecurrenceをpaired_polynomial_actionへ共通化し、legacy raw meanは元のfinite normを渡す。新Dpolyはnormalization=1なので、保存law/Bを再評価せずに補正済み理想多項式だけを適用する。旧commitのSOURCEは不変。

新kernelは、保持和Bと元の項別forward/reverse Cの切替、exact tailとpolynomial tailの切替、保存νによる一回の除算だけを加えた。DF runner全体は流用していない。DF exact_tailはCに対応するという設計上の区別を維持した。full4096次元のcomplex128、explicit CSR/expm_multiply(traceA=0)、元と同じscalar形成順序を使う。Qiskitは保存Pauli係数のsparse matrix化に使用し、回路は生成していない。

最終人工テストは12件PASS。可換/非可換保持、±時間、scalar、非unit raw norm、K0/2、r/q反復、旧raw meanとの関係、L0の独立dense oracle、tail semigroup、保存包含の読取とUNRESOLVEDを検査した。人工検査のPASSを実分子の包含保証へ格上げしない。

[kernel](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a0a3aff915f98456733d6bf09ed641a1be423628/src/trotterlib/pauli_mechanism_bias.py) [recurrence](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a0a3aff915f98456733d6bf09ed641a1be423628/src/trotterlib/pauli_signal_adapter.py) [runner](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a0a3aff915f98456733d6bf09ed641a1be423628/scripts/run_h6_pauli_p1_bias.py) [tests](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a0a3aff915f98456733d6bf09ed641a1be423628/tests/test_pauli_mechanism_bias.py) [固定条件](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a0a3aff915f98456733d6bf09ed641a1be423628/docs/research/h6_pauli_p1_bias_20261101.md)

## 6. 実行件数・終了・保全

主要6伝播＋予定済み整合2伝播＝8伝播を一回ずつ完了。親Pythonはnative exit0、全threadはexecutor終了時にjoin済み。科学retry0、所要 21.18秒。64 CPU affinityと十分な空きRAMを観測し、独立Dpoly4本だけを4threadで並列化した。共有設定や他利用者への変更はない。

| 伝播 | exponential action呼出し | polynomial matvec |
|---|---|---|
| B_Hybrid | 24 | 0 |
| C_Hybrid | 1256 | 0 |
| B_tail_semigroup | 32 | 0 |
| C_tail_semigroup | 1264 | 0 |
| Dpoly_H16 | 1248 | 384 |
| Dpoly_H32 | 1248 | 768 |
| Dpoly_H64 | 1248 | 1536 |
| Dpoly_R64 | 0 | 1536 |

SciPy内部matvecは個々のexponential actionを構成する内部処理であり、新候補・新信号の別実行と数えていない。保存target/law再評価、新full-H reference、RTE sampling、回路build/compile、新N/G/SE/順位は全て0。保存raw・SOURCE・入力・過去STOPへの書込0。

[execution_status.json](execution_status.json)と[STOP.json](STOP.json)を保全した。STOPはgrid4成分とH32/H64のtruncが未解像という科学的な分解能限界であり、8伝播の実行失敗を意味しない。条件や分解能を実行後に緩和してPASSへ変えていない。

## 7. 次段階の判断材料とmandatory STOP

- 小さいfinite/grid項自体を精密claimにする場合だけ、P1追加精度・数値誤差評価の新scopeが必要。今回の外側分割の観測を述べるために全成分の高精度化を必須にしない。
- P1Nは、必要shotの量的影響を知りたい場合の別許可事項。今回新Nは0。
- P2は単回Cの保持/event/結合差を知りたい場合の別許可事項。本P1はその価格を答えない。
- P3はDF/Pauliの共通一体補正・残差定義が必要な別研究scope。今回のH6入力を無条件に統合しない。

**計算はSTOP_UNRESOLVED_COMPONENTSで停止済み。結果公開・remote SHA/本文照合後mandatory STOP。次の担当はユーザー/GPT：この限定結果の扱いと、必要な場合のみ次のscopeを判断する。Codexの次の科学実行作業なし。**
