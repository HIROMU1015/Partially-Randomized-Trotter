# Q3-C：任意feedback角・全canonical contextへの移送

2026-11-10 JST。基準はQ3-B `1795d72b704cfd89aeac7c0075ebb576406ba624`。
これは固定SOURCEに対する著者の構造論証と技術検証であり、独立レビュー・研究上の正式採択・実機保証ではない。

## 1. 今回の固定scopeと実行前判定規則

物理条件はLinear H4、1.00 Å、STO-3G、DF12、元H厳密ground供給・準備費除外、Partial S2/LD3/h=.02/K2。保存18roundとcanonical finite-bit law、m17 q131072/r32/bits88、Compiler A/Qiskit1.3.0、ideal native RZ/SX/X/CX・reset・measurement、旧binary64算術・192-bit directed intervalモデルを保持する。Referenceのm17だけ既存V6-Q exact-binary B1 phase経路を使用する。

今回の新campaignは次に限定する。旧campaign/STOPは再開しない。

1. **C0**：固定B′4 contextとReference4 contextの保存catalogueを読み、Y有無の共通全recordのlowering/fence一致、P/H/Sdg、常在gate集合を照合。Q2の保存16 trace、Q3-B symbolic結果を照合する。新compile・operator計算なし。
2. **C1**：C0の構造的移送で十分なら実施しない。閉じないseamがあればその段階で未資格とし、今回の結果を使って後続費用を確定しない。恣意的fixtureを増やして全supportとしない。
3. **D**：保存Q0/Q3-B/V6-PのPASS、同一arm/rho/bias、δarith、B2余裕、failure和、energy精度の有理数検算1回。旧n探索・inverse/count全走査・二分探索を再実行しない。
4. **費用**：Dが閉じた場合のみ、保存TOTALS_18ROUND、V6-PのRZ会計、V5のround depth、catalogue/native gate多重度を用いる18round再会計1回。各armのm0–16は固定、m17は626 X+626 Yを340 Qへ変更。新角度・trajectory・compile・sampling・T評価なし。

SOURCE/Input hashは科学的再現性として新namespaceに記録する。入力不一致、core loweringの差、保証不成立はSTOP。旧runnerやone-shotを呼ばず、上記を新namespaceで完了・公開後mandatory STOPする。

## 2. Compiler contextの閉包

正本SOURCEは `track_c_compiler_a.py`、旧B′ `STRUCTURAL_APPENDIX.md`、`pr_reference_transfer_catalogue.py`/`pr_reference_bprime.py`。

| arm | 常在post-unroll gate | 任意gate（QではSdgを付けない） |
|---|---|---|
| B′ | p, xx_plus_yy, rz, rzz, cx, h, crz | cz |
| Reference | p, xx_plus_yy, rz, rzz, h, cswap | z |

Pは旧coreで既に常在する。B′ scalar Pは値0でもbuilderがemitし、Reference vacuum Pも無条件emitする。初段InverseCancellationにPはなく、UnrollはPの既存equivalence entryを保持する。新Qは旧Xと**同じglobal source basis**、同じparameter構造となる。source集合が変わるとBasisTranslatorのDijkstra結果も変わり得るため、単なるP単体の恒等式だけを根拠としない。

固定TargetなしのBasisTranslatorは(gate名,qubit数)ごとの変換mapを作り、placeholderへ各nodeのparameterを束縛する。wire配置/全register幅/重複回数/角度値を使う数値最適化はない。旧catalogueが含むunbound Pのtemplateは全contextで `RZ(alpha)`。Q3-B symbolicのP(−φ)も `RZ(−φ)`であり、φ依存global phaseは−φ/2。

全recordでY0/Y1の共通loweringが同じであることを保存データから確認する。従ってQのcore/Hは旧X/Yと同じnative列。SdgをPへ置換する末尾では、どちらも1 RZが残る。初段source passで旧SdgはHとinverse対を作らず、新Pも取消対象ではない。最終passはX/CXだけで、末尾RZがfenceになりcoreとの新たな取消を作らない。RZ(0)も消さない。Pの値の変化がglobal phaseへ与える丸めは**既にcontrolledなjoint全体に共通**でありobservableに寄与しない。lower後にcontrolを追加しない。

これにより任意の適格trajectoryと有限binary64 feedback角について旧coreとHの局所誤差chargeを継承する。F.inverse()は既存SOURCEそのまま、compiled logical wordのactual adjointと取り違えない。basis共有、D融合、signed products、scalar/branch relative P、vacuum補正を変更しない。

## 3. Observableと一般native誤差：virtual stateによる接続

理想readoutは `P(−φ)→H→Measure`。`P(−φ)† X P(−φ)=cosφ X+sinφ Y`なので、coherence zに対しmeanは `Re(exp(−iφ) z)`。

native HをH̃、RをRZ(−φ)、coreと初期Hを適用したnative joint stateをṽとする。純粋状態で書けば

`wφ = R† H H̃ R ṽ`

は正規化されたvirtual stateであり、physical meanは厳密に `Re(exp(−iφ) z(wφ))`。mixed stateにも線形に拡張できる。各trajectoryの `z(wφ)=2〈w0|w1〉` は絶対値≤1、その平均も≤1。state側へnative readout誤差を移す数学的表現であり、新しい回路を実行するのではない。

`||R† H H̃ R−I||=||H̃−H||` はφに依存しない。旧core誤差と初期/終端Hの誤差をtelescopingすれば、既存native joint error E以下。旧Eには2Hに加えSdgの非負上界も入っているが、それを除かず安全側に保持する。P→RZのangle保存は解析的恒等式であり、新しいoperator誤差はない。

coherenceの摂動は既存の `2√2 E` raw disk以下（Hermitian X/Y observableのノルム評価）。φ依存のvirtual meanでも、**履歴を固定した条件付き実験ではφは一定**。全φに同じdisk、radius下界、phase bias上界が使え、Q0が未知ρ・biasについて一様に囲む許容集合へ入る。ρ上限1はphysical/virtual stateの正規化から従い、lower radiusから上限を推測しない。

一般operator誤差をradで足していない。PF、finite-bit、SOURCE、旧native、B1 vacuum diskは旧会計に一度だけ保持する。追加Pの純粋角度差は**0 rad（固定ideal-nativeモデルの解析結果）**であり、旧H/core誤差を0にした意味ではない。feedback binary生成のδarithはQ3-Bの正の上界を別に保持する。

## 4. 実consumerと独立性

新 `pr_reference_feedback_q3cd.AdaptiveReadout` はQ3-Bのfixed-history branch selectorとQ1のfeedback objectを一度bindし、全shotで同じimmutable objectを使う。`physical_shot`は1回ごとに既存CanonicalFiniteBitProducerを呼び、新event列を作る。feedbackを再推定せず、seed/RNGを新設・変更しない。

Referenceは既存 `assemble_reference_for_saved_round(...exact_binary_m17)`→Q1 feedback_readout→Compiler A→既存_compile_and_reset。resetはcompile**後にprefix**し、physical実行順はresetが最初。B′は既存builder→rewrite_rte→symmetric_fused→同じQ readout→Compiler A。X/Y consumerを呼んでaxis名だけ変更しない。m0–16は旧X/Y consumerのまま。

実m17 wholeを今回build/compile/実測していない。人工testは4回のproducer/compile要求、共通角、reset順とvirtual-state恒等式を確認する。巨大な実m17 callを実行した証拠とは区別する。340個のphysical ±1結果を得るbackend loopは呼出側の仕事であり、trial間/round間のfresh independent uniform bits・ideal測定は固定モデルの前提。任意の実RNGや実機ノイズを資格化しない。

## 5. 移送範囲と限界

上の有限context閉包＋parameter template＋norm補題により、有限q1 fixtureからの外挿を使わず、固定SOURCE全supportへ移送する。旧SOURCEのgate family、compiler/equivalence library、phase/provider、precision、canonical lawを変更した場合は対象外。T word/CT誤差/domain問題を解消した証明ではない。正式な保証の研究上の採否はGPT/ユーザーへ戻す。
