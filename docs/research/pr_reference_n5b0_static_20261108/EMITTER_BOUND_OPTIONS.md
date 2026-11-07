# N5-B0：emitter上界の意味・改善候補と証明課題

2026-11-08 JST。[REPORT](REPORT.md)。**以下は静的な条件付き代数と技術候補。新しいemitter上界、RPE誤差規約、policyの正式採択ではない。** 保存64u／辞書emitter／shell certificateを変更していない。

## 1. 64uは何を覆うか

[N3-B1a](../pr_reference_n3b1a_emitter_static_20261108/PHASE_ERROR_BOUND.md)はu=2^-53、P=Pi0/4、tau=8Pとして、local wordのW loopを8状態へ閉じる。Localは0 phaseから始まり、任意の有限W個数Kに対して保存phaseは `(K mod8)P`。Negativeは全文字列のactual inverseを一度使う。

同証明の64uは、local W定数差と任意のcarrier位相へのjoin、周期差等を一様に覆う**局所1-qubit Frobeniusの安全上界**。W数へ比例する実測誤差でも、最良のword別評価でも、不可避の誤差でもない。辞書の保存係数はさらに43-carrier等の経路上限を覆うため、実際のcomplete-word経路の回数より保守的となり得る。

Actual C0 metadataには全residueがある。

| W mod8 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 保存C0件数 | 49 | 31 | 43 | 29 | 27 | 36 | 42 | 42 |

これはwordの整数metadataで、角度をmodで置換したものではない。49個のresidue0 wordがあっても、全物理位置のprefix phase、他residue、shellの誤差が消えたとは言えない。

## 2. 経路S：同じemitterのstrict評価を鋭くする

Exact named-gate意味論と既存8状態補題を仮定する。Freshなphase0 localへpositive wordをemitし、他の一般carrierへjoinする前の比較なら、k=K mod8について

`U_emit_local = exp(i k P) A_word`,

`U_logical_word = exp(i k pi/4) A_word`

となる。A_wordはW以外のnamed gateの同じ行列積である。この限定local差は

`sqrt(2) * |exp(i k P) - exp(i k pi/4)|`

という式に帰着する。負角もactual inverseと対応する符号residueを使う。k=0のfresh localではscalarが一致するが、これを任意のprefixへのjoin後のcertificateへ一般化しない。

Carrier側では、prefix phi、local residue、加算丸め、rem_euclid、inverse／copy等の実経路を対応させる必要がある。LocalのW loopの8状態閉包だけから、一般prefixを含む全累積位相が常に同じ8状態に閉じるとは言えない。Local閉包とcomposerの閉包は別命題である。

必要な追加証拠は、word／位置ごとのphase処理回数、可能なprefix状態の安全な扱い、真piとPi0の保証付き関係、全体inverse、非zero初期phase／tau境界の対応である。今回これらの新しい数値上界を評価せず、64uを小さい係数へ置換していない。単に保存W residueを知るだけではm17のstrict capを満たす証明にならない。

## 3. 経路G：共通joint scalarを信号保証へ分離する

### 条件付き命題

比較の開始点を**全controlを付けてcompiler Aを通したjoint native回路**に固定する。以下を仮定する。

1. 各RZへ渡すlogical wordは保存文字列そのもので、exact Wを含む。負角はcomplete actual adjoint。
2. EmitterはH/S/T/Tdag/Xとそのnamed inverse、既存SX=HSH、X/CXだけを同じ順序・wireへ置く。Named gateは既存ideal論理意味論。
3. Float phase算術はQuantumCircuitのglobal_phase欄だけを変え、named gate列・引数・control条件へ転用されない。
4. Consumer後に新しいquantum control、relative phaseへの転送、再合成・新compilerを加えない。Reset／measurement／clbitは元と同じ。

この場合、固定trajectory omegaごとに、logical word置換後のunitary body L_omegaと実emitterのbody E_omegaについて、ある実数chi_omegaが存在して

`E_omega = exp(i chi_omega) L_omega`

と因子分離できる、という証明候補になる。

### 代数的根拠

W以外のnamed gate列をAとすれば、logical wordはexact scalar×A、実localは保存numeric scalar×同じA。Actual inverseもscalarの共役とnamed列の逆転に分離する。Wire embeddingではscalar×Iをtensor Iしてもjointのcommon scalarである。積の各scalarは全gateと可換なのでwhole bodyの一つのscalarへまとめられる。

Prefix設定、local→segment、segment→outerのfloat算術がnamed列を変えないなら、それらもcommon scalarにだけ影響する。この議論はchiの数値を求めず、targetへのphase fitも行わない。SOURCEのこの型付き対応を全consumer経路へ確実に移せることが適用証拠となる。

実際に読んだ[lower_native](../../../src/trottertracks/track_c2_b_verify.py)ではWをphase欄へ加え、他文字をnamed gateへ置き、負側だけlocal.inverseする。[consumer](../../../src/trottertracks/pr_reference_common_word_consumer.py)はglobal_phaseを別欄に持ち、unitary segmentをinplace joinし、合成後のcontrolを作らない。公式Qiskit SOURCEに対応するinstalled quantumcircuit.pyでも、composeはphase欄の加算と命令wire写像を分け、inverseはphase否定とnamed列の反転を分けている。

### 信号に対する帰結と厳密な限界

上の因子分離が適用できれば、任意の入力density rhoと同じwhole-joint observable Mに対し、

`Tr[M E_omega rho E_omega†] = Tr[M L_omega rho L_omega†]`

である。X/Yの各trajectoryの測定確率について成立するので、chiがomega依存でも、同じcanonical sampling lawで平均した複素raw信号を変えない。これは**joint回路の測定ごとの恒等式**であり、uncontrolledな平均operatorの一致や平均channel一致をcoherent signalの一致に代用した議論ではない。

一方、strict operator距離では一般に `E_omega != L_omega` で、scalar差も誤差として残る。**この経路でe_emit=0という新しいstrict RZ certificateを作ることはできない。** 現在のstrict certificateを保全したまま、pure common scalarがraw signalへ寄与しないという別の保証を採用できるかが科学的判断となる。今回はそれをRPE誤差式へ代入せず、m17の数値PASSを認定していない。

特に次はこの議論で消せない。

- SOURCEのscalar／event／vacuum P角。これらはancilla等の**relative phase**を実装する。誤ったthetaや量子化はgateのrelative eigenphaseを変え、common scalarとは限らない。
- Uncontrolled F/Gのglobal phaseを、後でreference干渉／vacuum補正へ使うSOURCE経路。Field名がglobal_phaseだから無視できるわけではない。
- Logical wordとRZ targetの近似差。保存guardのWを削除したり、guardをprojective phase fitへ変更したりしない。
- 合成後にcontrolを付ける経路。そこでcommon local scalarがcontrolled relative phaseとなり得る。
- 新たな角再分解、numeric gate matrixの丸め、別backend、noise。Named ideal gateモデル以外へ自動移送しない。

## 4. Adapterが必要か

| 状況 | 技術的な分岐 |
|---|---|
| 同じemitterで厳しいstrict評価を閉じられる | 証明／certificateの限定補強が候補。新adapterは不要となり得る |
| Pure common scalarの信号因子分離を採用できる | 同じ実装のままRPE向け保証を補強できる可能性。Strict operator保証とは分ける |
| Strict phase保証が必要で、現SOURCEの証明が足りない | 位相bookkeeping等のadapter候補を別scopeで検討。現時点で必要とは断定しない |
| Logical guard／量子化／域が不足 | Emitter改善だけでは解消しない。別のprecision／provider条件が必要 |

Integer-W sidecar、symbolic pi、新しいdeferred phase処理は今回実装・採択していない。新adapterにはinverse／compose／serialization／relative phaseのcorrectnessが改めて必要。既存SOURCEを無断で書き換えない。

既存N3のfinite testsとN4-Bの局所検査は再利用する。数値Operator差をphase fitで小さくした値や、浮動診断tolerance1e-10をm17の10^-16級certificateへ格上げしない。次の限定検証は[NEXT_EXECUTION_SCOPE](NEXT_EXECUTION_SCOPE.md)へ固定し、今回はmandatory STOP。
