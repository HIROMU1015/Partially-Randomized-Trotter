# Pauli-JW純RTE／Hybrid研究：論文化向け証拠統合・残課題追補

2026-11-02 JST。**既存の条件付き資源比較は、追加科学計算なしでMethods／Resultsと図表の材料として整理できる。全law実装精度や母期待価格を含む比較を主張する場合は、別の証拠が必要である。**
本書は[2026-11-01論文用統合資料][PAPER]と[主張・図表対応表][CLAIMS]への追補で、原稿や旧資料を置き換えない。
中心claim、学術的新規性、独立論文／補助研究の採否、投稿可否、詳細科学レビューの判断はユーザー/GPTへ返す。

## 1. 目的・作業範囲・研究全体との関係

主題は、**Pauli-JW時間発展での決定論的保持、有限ランダム化、normalization、bias、shot負担、compiled単回価格の競合**である。
Full PF、B0/B0I、Hybrid PR、純RTEという既存4方式の定義と有限候補範囲は旧統合§1–3を継承する。
純RTEはL_D=0で全非identity項をランダム化するが、constant scalar phaseは残す。

本研究全体の中心目的は、固定DF Hamiltonianに対する部分ランダム化PFと決定論的PFを比較し、最終的にRPE全roundの資源へ接続することである。
Pauli-only研究はこの目的を置き換えない。[研究目的][OBJECTIVE]はその長期目標と限定signal研究を区別し、
[研究全体地図][LANDSCAPE]・[claim/gap表][GAPS]は完成済みのDF有限時間結果、短round検証、全roundモデル会計と、最終保証の不足を区別している。
「RPEはすべて未着手」「Pauliの固定T結果でRPE全体まで完成」のどちらにも読み替えない。

今回実施したのは固定文書・一次CSV/JSONの読取、保存文字列の照合、出典・図表の版差確認と文書作成だけである。
新科学計算、保存値の数値再解析、新G/比率/SE/信頼区間、候補探索・再ランキング、sampling/RNG replay、QPY decode/生成、回路build/compile、状態/reference生成、誤差/価格tail評価は0。
新しい数値図・比較CSV・数値解析runnerも作成しない。既存Methods／Results・論文本体・caption/画像の原本は変更しない。

## 2. 既存資料の版と今回の参照規則

| 資料・段階 | 固定版／時点 | 今回の使い方・注意 |
|---|---|---|
| 論文用統合・claim/figure map | `4591b8d75a2a6a55c74a51225f0c61648023c8ee`の[PAPER]／[CLAIMS] | 全文を読み、既存Table 1/S1、Methods§8、Results§9、C1–C10を再利用。旧文書bytesは保持 |
| 研究全体の目的・到達点 | [OBJECTIVE]／[LANDSCAPE]／[GAPS] | 正本の目標と作成時点を確認。11/01時点のU01/U02の不足には後続P1/P2が部分証拠を追加した |
| Pauli総合資料 | 11/01作成の[SUMMARY]／[旧latest15][OLD15] | H2/H4の整理は再利用。H6欄はHybrid大event拡張前なので、最新H6比較は旧論文用統合のE6へ従う |
| A / A-1 / B | [A] `0e1c7aae386986f6c1bb9483548671655447f516`／[A1] `f36437a662212001abcbd5b65a8cf138cd3da0f5`／[TAIL] `0d5558344ed7b94b264f79cbbb9c7deaa6bf0071` | 実装誤差・価格不確かさの別々の調査。固定commitで参照し、branchをmerge/rebaseしない |

旧latest15のH6保存H/R約1.814–5.938を、最新H32/R64の倍率として使わない。
旧論文用統合には既にH6の5行をE6へ更新する規則があり、最新の保存H32/R64比は約1.129/1.152/1.183/1.489/2.092。
これらは保存表示の引用で、新しく割り算した値ではない。最新純RTE対決定論familyの未保存倍率を、旧PR/D比から作らない。

| 比較の科学SOURCE | 固定SOURCE／役割 |
|---|---|
| H2追加確認 | [SOURCE][H2SRC] `bf5f2339f723f4edb7bf91a8ca88f33251bbef57`。Phase Aは旧target/Nを再利用し確認価格を追加。元Hybrid/決定論familyの複数SOURCEは旧統合§3のE1に保持 |
| H4探索主表 | worker `52865921930b3afd11b1447dff0a6cbcb6085af8`、選択 `4a7f91fa7ba8aa3ae1ce4ccb30bde9b973869410`、最終比較 `a17ad381b6ff2ac615e3cd930c641947c551bcbc`。後続確認後minとは役割を分ける |
| H4/H6純RTE event拡張 | [SOURCE][EXTSRC] `e755ecf581600192fd2745d7066e410cebd933d3`。保存HybridやH4旧値の新計算SOURCEではない |
| H6 Hybrid拡張 | [SOURCE][H6SRC] `887b69e449cb3c378286f09d0d7626599850fb67`、解析 `af891fd45805454c6592bbee5441266f14235ffd`。原結果00958ae1と科学実行/解析SOURCEを区別 |

Full PF/B0対照のSOURCEと旧直接比較SOURCEは旧統合§3 E1/E2/E5bへ委ね、文書統合commitを科学実行SOURCEとして扱わない。

## 3. 固定条件と分子別の主要結果

共通条件はlinear H2/H4/H6、隣接1.00 Å、STO-3G、中性singlet、保存Pauli-JW、S2、T=.8。
非identity項数/system qubitsはH2が14/4、H4が184/8、H6が918/12。Pauli-onlyなのでDF rankは適用外。
分子内で同じ保存Hamiltonian/raw state/reference/exact νを用い、方式別再正規化はしない。
制御はcentral rotation、測定はcosine／+Im sineのFull Hadamard wrapper、状態準備P=0。
compilerはQiskit1.3.0、rz/sx/x/cx、opt1、seed17、追加backend/coupling/layout/routing指定なし。
要求ε_sigは.05/.01/.005/.001/.0005、軸配分と失敗確率は既存のbinary64規則によるε_sig/√2、α_a=.025を維持する。
K2/K4は最大event次数0,2／0,2,4の違いで、S2の次数とは別である。

| 対象 | 既存比較と保存候補 | 使える結果／選抜上の限定 |
|---|---|---|
| H2 | [追加確認][H2]。Hybrid `PR_p0001_q1_r4_K4` n96。純RTEは.05/.01で`PR_p0000_q1_r4_K2`、他3精度で`PR_p0000_q2_r2_K4`、各n384 | 全5精度で条件付き純RTE低G。主要更新は既存s4候補の追加確認で、大event拡張の成功ではない。確認後min・逐次延長を含む |
| H4 | [探索主表][H4]と[後続直接比較][EXT]。確認後minは.05でHybrid `p036_q2_r8_K4` n48、純RTE `p000_q8_r8_K2` n384。 .01は純RTE K2、他3精度は純RTE K4 | .05だけHybrid低G、他4精度は純RTE低G。探索nominee主表の.05 Hybrid K2／.01 RTE K4とは別の選抜規則。s128/256での費用増加も保持 |
| H6 family | [純RTE拡張][EXT]のR64=`H6_p000_q8_r64_K2` n48と、[Hybrid拡張][H6]のH32=`PR_p0078_q8_r32_K2` n64 | 全5精度で条件付きR64低G。rも異なる既存有力候補同士。H32は探索nomineeと確認後minが一致 |
| H6 q/r/K一致 | 同じR64対H64=`PR_p0078_q8_r64_K2` n64。[保存comparison_details][H6DETAIL] | 全5精度の保存判定は`REFERENCE_LOWER_EMPIRICAL_G`。prefix78の有無で辞書・law・B・biasも変わり、保持回路だけの一因子対照ではない |

H6の固定2比較はq8/K2、H32はr32、H64/R64はr64である。同じR64標本を二つの独立確認と数えず、方式間の同じindexをpaired標本にしない。
H2/H4/H6の5精度はそれぞれ価格標本を共有する。**14/15は保存された比較判定の要約で、15独立実験、母平均winner、全parameter最適性ではない。**
Full PF/B0/B0Iは旧統合Table 1/S1の有限family対照をそのまま使い、B0/B0Iを別の独立方式として二重計上しない。
H2の高精度B0欠測は適格登録候補なしで、0/∞費用や全B0の不可能性ではない。

## 4. B・bias・N・C・G：会計上の説明と因果の境界

既存会計では候補/軸別に`h_a=ε_a−|b_a|−u_a−ρ_a`を置き、正のheadroomと保存guard条件の下で、B_upを使うshot十分式からN_aを得ている。
比較量は`G_hat=N_cos C_hat_cos+N_sin C_hat_sin`、C_hatは状態準備を除く**Full compiled RZの保存標本平均**、単位はRZ×shot。
Nは量子測定shotの会計値、nは古典的trajectory価格標本数であり、N回の量子測定を実施した結果ではない。
主結果のρ=0は仮想条件で、actual compiledρ=0やguard所属の確認ではない。
B_savedの信号補正、B_upのrange、数値allowance、log marginの旧政策は候補/版ごとに保持し、同一視しない。

| 観測されている競合 | 既存証拠 | 言える範囲 |
|---|---|---|
| B/Nが大きくてもC低下でGが低い | 旧統合§6.3のH2粗精度・H4厳精度・H6粗精度の保存内訳 | normalizationの小ささだけでは純RTE低Gを説明できない |
| H4粗精度でHybrid低G | 同じ§6.3とTable 1 | 低CだけではG優位を決められず、Hybridのshot節約が勝つ例を保持 |
| H6厳精度ではbias/headroomも重要 | 同じ§6.3の保存bias/u/h/Nと[P1] | 純RTEは大きいBでも小bias・広いhによりNが小さい。保持内部PF誤差という説明はP1の可換prefixには当てはまらない |
| event数増加はGを単調に下げない | 旧統合§6.1–2、[event_cost_trends][EVENTCSV]、[Hybrid拡張][H6] | H6純RTE s512→1024、Hybrid r32→64で再上昇。探索/確認を区別した有限grid観測で、単峰性・連続最適性ではない |

同じtrajectoryの両軸価格の依存を保存G SEの会計で保持し、方式間をpairedにしない。名目SE・t区間・2 combined-SE表示は、未観測tailや選抜後/逐次観察後の同時coverageを保証しない。
prefixを変えるとtail・law・angle・normalization・bias・compiler入力も変わるため、保持部だけの独立因果寄与率は未確定。
以下のP1/P2は原因説明を具体化するが、この因果上の限定を解除しない。

## 5. 最近追加されたH6証拠と、更新されるLimitations

| 証拠 | 固定結果／SOURCE | 追加された内容 | 残る限定・原稿での扱い |
|---|---|---|---|
| P1 bias分解 | [P1] `bfb4ca75437a9b3605a8f57b5f82100615dd034f`／`a0a3aff915f98456733d6bf09ed641a1be423628` | H16/H32/H64の共通外側split差が解像。保持78項は可換I/Z、理想内部保持PF差は0という構造と数値整合。R64/H16 finite項は診断分解能を超えた | H32/H64 finite項と全grid差は未解像。1e-10はengineering診断分解能。未解像を測定上の0にせず、分解の寄与率や実ρへ転用しない |
| P2部品価格 | [P2] `c9073274daee69c9b69d8db3dc332bf6132c2259`／`11e50e262f95630b14fa4070ad30894ba3f64b74` | 固定confirmation/0000でFull6軸を再利用、R/E/Empty 12compile完了。保持・event・wrapperの非加法性を保存 | 個別programの操作的削除対照。母平均価格の配賦・prefix単独の因果効果ではない |
| selected implementation | [LOCAL] `71432728ee0140fb3c1f8bf30ecf51b2faba86ec`／`9d392de7c8c6f8ae35faba9aeb1c8c6f6a215fef` | 固定R64/H32/H64 index0000、selected-target3＋Native6＋Compiled6＝15状態作用、6比較の局所engineering整合 | η/δ/dの直接差は未解像、厳密certificateでない。保存局所guard内という上側候補を全lawρにしない。H64/R64の当該標本はorder0だけ |
| A 全law転送調査 | [A] `0e1c7aae386986f6c1bb9483548671655447f516` | 既存R2/R2-Cから、Native body差とCompiled wrapper差を分ける条件付き転送式を再利用できる | 一様Native phase/compiler誤差は未証明。実装family全体の非自明な数値上界は未取得 |
| A-1 Native調査 | [A1] `f36437a662212001abcbd5b65a8cf138cd3da0f5` | 保持角度のSOURCE順序、Hybrid共通78係数、R64空和、K2位相更新順・回数と条件付きNative式を整理 | actual remainder/windingの一様誤差は未証明。78係数の新scalar評価は未実施で、Native全law上界は未取得 |
| B 価格tail調査 | [TAIL] `0d5558344ed7b94b264f79cbbb9c7deaa6bf0071` | Full価格・event列・cohort・law・元統計が追跡可能。追加samplingを条件付き報告の必須条件としない | Order2寄与率、未観測component列、母期待価格・母平均Gの順位は未保証。高価格tailが順位を覆すとも覆さないとも確定しない |

P1の共通Δ_outerは保存表示で`7.016398502702e−5 + 1.136269663863e−4 i`。
これは固定された保持/tail分解経路の観測で、H_DとH_Rの可換性や全biasの一因子配賦ではない。
保持I/Z内の可換性と、Nativeの角度binding/相対phase丸めは別であり、A-1の残条件はP1で消えていない。

P2の保存index0000 Full RZはR64=3081、H32=4732、H64=6735、共通Hybrid Retained-onlyは2500。
Event-onlyはH32=2143、H64=4163/4162、R64=3081/3080（cosine/sine）。Emptyは0/2。
保存`F−R−E+Empty`のRZはH32=89/91、H64=72/75、R64=−3/0。
これらは原P2値の引用で、新たな差分計算ではない。Fullを孤立R+Eの和としない。
R64にもscalar-only Rがある。H64とR64は同event位置数でもE価格が異なるため、Full差を保持78項だけに帰属できない。

Bの保存確認cohortではR64/H32/H64のorder2観測event数は13/10/3。これはtrajectory数でも確率でも独立価格寄与でもない。
K2のorder4はsupport外。order2が観測済みでも未観測component列の費用寄与は制限されず、order0だけの層にも価格変動がある。
有限辞書・有限長の希少価格部分をtailと呼んでおり、無限supportのheavy-tailや無限期待値を実証した意味ではない。
R64のraw価格配列があることを理由に、未保存の分散/共分散/観測範囲を今回新集計して補わない。

H6の保存targetは既に53-bit実効lawとbinary64 event角度を含む。P3のfinite-bit=0という仮定へ戻したり、P3の有限bit検証をH6へ再実装したりする必要はない。
一方、保存平均targetと個別trajectory targetは別であり、平均値を個別QPYから引いて実装誤差としない。

## 6. 主張候補・証拠・追加検証の対応

IDは旧claim/figure mapのC1–C10を優先する。依頼文中の「normalization競合」「event非単調性」「保持費」「実装/tail」は、既存C9、C6–C8、C9/C10とLimitationsへ対応付ける。
新しいC番号を設けず、実装/tailはC1等の成立範囲を更新する補助証拠とする。RPEは既存全体gap表のR01/R02・U07を参照する。

| 主張候補 | 支持する既存証拠 | 科学的成立範囲 | 未解決事項 | 追加計算が必須か |
|---|---|---|---|---|
| 純RTE低G：C1、H6はC8 | [H2比較][H2COMPARE]、[EXT比較][EXTCOMPARE]、[H6比較][H6COMPARE]、旧Table 1/S1 | 固定有限候補・保存価格標本・仮想ρ=0で14/15。5精度は標本共有 | 母平均winner、選抜補正、全law実装精度、大域最適性 | この条件付き報告には不要。強い母平均/実装込み主張には必要な追加証拠を別設計 |
| H4粗精度の例外：C2 | ε=.05のHybrid K4対RTE K2の保存比較 | Hybridが低G。元探索nominee K2との時点差を維持 | K差や近い候補の母平均順位 | 既存例外の報告には不要。純RTE一様優位を採択しない |
| Normalizationと回路価格の競合：C9 | 旧統合§6.3のB/N/C/G、[MECH] | H2粗/H4厳/H6粗は大Nを低Cが補う会計上の説明 | 個別要因の独立因果寄与 | 内訳の記述には不要。寄与率を主張する場合だけ識別条件を別検討 |
| H6のbias／headroom：C9＋P1 | 保存bias/u/h/N、[P1]の外側split・可換保持 | q8/prefix78で内部保持PFと外側splitを区別。厳精度のN/C競合と整合 | H32/H64 finite・grid未解像、相対phase/実ρ | 現在の分解能付き観測には不要。小finite値や全精度保証を主張する場合に限定追加 |
| Event数と費用の非単調性：C6/C7/C8 | [EVENTCSV]、H6 Hybrid r系列 | 指定gridでB/N節約とC増加が競合。探索/確認を分ける | 連続最適性、単峰性、母平均最小 | 有限grid観測には不要。大域最適を求めるなら別問題 |
| 保持prefixと費用差：C9、C10の対照とは分離 | [P2]、既存generic/central D2対照 | 固定Full/R/E/Emptyの操作的価格と非加法性 | 母平均寄与率、保持除去だけの因果効果 | 固定例の報告には不要。因果claimを採る場合だけ必要な介入を設計 |
| 実装精度：C1/C9のLimitations追補 | [LOCAL]、[A]、[A1] | 3固定trajectoryの局所engineering確認、条件付き転送構造 | Native remainder/winding、Compiler一様誤差、全lawρ | 条件付きρ=0報告には不要。実装精度込みの主張には全law証拠とshot会計接続が必要 |
| 母期待compiled価格：C1のLimitations追補 | 保存価格・統計・設計、[TAIL] | 観測価格平均/SEとlawへの所在対応 | tail/層別価格/残余寄与、母平均比較を分離する根拠 | 保存標本報告には不要。母期待Gを主張するなら必要。新samplingが必須とは未判定 |
| 決定論対照：C3/C4 | 旧Table 1/S1、H4 nominee対最良決定論family主表 | Full PFとB0/B0Iを別対照として保存範囲で使用 | 最新候補の未保存対D倍率、全方式の対称最適化 | 既存表の使用には不要。旧比を新候補へ流用しない |
| H2確認範囲の更新：C5 | [H2]のcoverage/Phase A/B | 追加独立確認による比較集合の更新と、大s探索の区別 | 逐次延長・確認後選抜のcoverage | 履歴の記述には不要。選抜後母平均保証には別根拠 |
| RPE全roundへの一般化：R01/R02・U07 | [OBJECTIVE]、[LANDSCAPE]の既存RPE証拠 | 背景・別研究のモデル段階を説明する材料 | Pauli全roundの位相/半径/分枝/energy精度/失敗率/long-round費用 | Pauli固定T報告には不要。全RPE主張には対応する別検証が必要 |

## 7. 主張を強めるときだけ必要な証拠

| 主張の範囲 | 現状で使用できる内容 | 追加証拠が必要になる条件 |
|---|---|---|
| 固定候補・保存価格標本・仮想ρ=0 | 旧Table 1/S1・Methods/Resultsと本追補のP1/P2/局所診断を、各限定付きで使用 | 現在の未解決項目をすべて終える必要はない。中心claim/新規性の判断は別 |
| 母期待compiled価格、ρ=0の母平均G | 既存平均/SEは推定値として引用可能 | 母平均を分離する推論・tail/残余費用の根拠、選抜/逐次設計と軸間依存。保存値再解析・構造上界・必要時samplingの具体scopeを別許可 |
| 全law実装精度を含む総費用 | A/A-1で必要な辺と条件を特定したところまで | Native＋Compilerの妥当な全law上界、B/bias/uの重複を避けたshot/N/G再接続。真の期待G順位なら母期待価格の証拠も独立に必要 |

**全lawρが保存guard内に入るだけでは、ρ=0時のN/Gや費用順位は自動的に維持されない。**
一方、母期待価格の根拠だけ得ても実装精度は保証されない。A/A-1とBの残課題を一つのPASS/FAILへまとめない。
P1の小finite biasの高精度化、全lawの大規模状態作用、価格標本の大量追加は、条件付きcase studyを記述するための無条件の必須作業ではない。

| 一般化・追加の主張 | 今回の固定証拠で支持できない範囲 |
|---|---|
| 他分子/geometry・サイズ則 | H2/H4/H6の1Å/STO-3Gという固定例から普遍則を出さない |
| Pauli以外の表現 | DF fragment、一体保持、辞書・scalar・law・control条件の違いがある |
| 全parameter最適性 | 有限候補と確認後選抜は全prefix/q/r/Kの最適性でない |
| DF B2との直接順位 | 異なる分割/表現・候補/参照来歴のGを混ぜた新表を作らない |
| RPE/energy・実機/FT資源 | 単一Tのsignal/RZ×shotはenergy精度、全round、状態準備、noise、FT合成・routing/physical resourceの保証ではない |
| 新アルゴリズム/priority | 既存RTE/PF/組合せの利用であり、本作業で新手法や先取性を認定しない |

## 8. 既存Methods／Results・図表への修正候補

以下は**採用前の編集候補**であり、旧草稿・論文本体・画像を変更していない。

| 既存箇所 | 確認結果 | 必要な追補候補／配置提案 |
|---|---|---|
| Table 1／Table S1 | 固定4591版の15行と60追跡行は、候補/G/cohort/n/SEを一次CSVの保存文字列と照合できた。15個のH/R文字列と保存比較判定も一致 | 数値の差替えは不要。条件付きρ=0・保存標本・15独立実験ではない旨をcaptionに維持。Table S1を本文表のSOURCE/選抜追跡に使用 |
| H6のTable 1/2とResults§9.1 | 最新H32/r32/n64を既に反映。H64も元comparison_detailsで対照がある | H32/R64 familyとH64/R64 q/r/K一致を区別する短文を本文、H64の詳細は補足。旧H6倍率を流用しない |
| H4 .05/.01とTable S2 | 粗精度例外とnominee/確認後minの区別は既に記載 | .05 Hybrid K4／元nominee K2、.01 RTE K2／元nominee K4を保持。Table S2の旧PR/D比を最新候補の比にしない |
| Methods§8.1–3 | 分子/input、S2/T、Full制御/測定、compiler/P=0、精度配分、ρ=0、選抜/cohortの限定が既にある | H6保存targetは実効53-bit law/角度を含むことを明記する候補。実装検証が局所engineeringと全law条件整理に留まること、Nとn・理想数値包含と実ρを再確認 |
| 統合§6.2の「bias成分別帰属は未分離」・Results§9.3 | 旧時点ではP1前の記述 | 「P1の固定分解経路では外側split差を解像し、可換保持内部PF差と区別した。ただし小finite/grid差は未解像で、prefix選択の独立因果配賦はない」へ更新する候補 |
| C9機構説明・Table S3 | 既存S3は旧Pauli generic/central D2の対照 | P2のH6固定Full/R/E/Emptyを別panel/補足例として追加する候補。旧D2と同じ処置/対象としない。本文は保持価格と非加法性の存在まで |
| Table S4・Discussion/Limitations | 旧表はbias/u/headroom・保証範囲 | P1未解像、LOCALの15作用/6比較とengineering限定、A/A-1の未証明辺、Bの母期待価格不足を追加する候補。本文では結論の条件、詳細は補足 |
| Fig.1 H4、Fig.S1 H2 | 固定画像・関連CSV/coverageが存在する。既存captionは探索/確認・H4例外・H2追加確認と大sを区別 | 既存図を再利用可能。2SEをtail保証と書かず、K/選抜規則を維持。新図は必須でない |
| Fig.2 H6拡大図／Fig.S3全s図 | 10/26原図のHybrid参照線は旧n48で、10/27 H32/n64ではないことを旧mapが明示 | 原図を使う場合は時点差をcaptionで明示し、最新比較はTable 1/2へ誘導。線を最新値と読み替えない。描き直す場合も今回は未許可・未実施 |
| Fig.3 H6 B/N/C | B/C/Nの図で、biasを描いた図ではない | headroom説明には旧Table 3/保存bias-h表とP1を併記。曲線の最小を連続域の最適と呼ばない |

今回の図表確認は固定参照の存在・保存表・captionの時点対応の技術照合で、画像からの値の再抽出や新しい作図・fitはしていない。
既存本文Table 1/2/3とSupplementary S1–S4の構成を再利用し、新しいClaim IDや図番号を完成条件として増やさない。

## 9. 論文の位置付け二案：判断材料

| 観点 | 案A：独立したPauli-JW資源比較 | 案B：DF部分ランダム化PF研究の比較・補助 |
|---|---|---|
| 既存証拠で構成できる内容 | H2/H4/H6有限候補のB/bias/N/C/G競合、14/15とH4例外、選抜/event数の非単調性、P1/P2の限定機構例 | 表現/分割が変わると費用構造も変わる比較上の注意、純RTE境界と非自明保持を持つDF手法の違い |
| 再利用する原稿 | 旧Methods/Results、Table 1/S1、C1–C10、既存event図。本追補をLimitationsと補足証拠へ | DFの正本RQ/既存原稿を維持し、Pauli表を条件別の補助節/別補足へ配置する候補 |
| 不足する研究判断 | 条件付きcase studyとしての中心的意義・既存研究との差分・読者/投稿先との適合性 | DF本論のどの主張をPauli補助結果が支えるか。数値順位を横断せず論旨を接続できるか |
| 追加科学計算の扱い | 条件付き範囲で原稿材料を構成するためには不要。母平均/実装保証等をclaimへ加える場合だけ別検証 | 横断した同精度順位や全RPE効果をclaimとする場合は新しい比較条件の定義と証拠が必要。今回それを提案・実行の既定工程にしない |

**案A/案Bは未採択。** Codexは独立論文の新規性・投稿可能性やどちらを主題にすべきかを決めない。
Pauli純RTEとDF B3は同一方式ではない。B3は一体項を保持する。
P3のM_B/M_Cも共通の非自明なGaussian一体保持blockを持つH4 matched分割のcase studyであり、Pauli純RTEそのものではない。
P3 Stage 1–4/有限bitの成果は[固定結果][P3]へ参照できるが、本Pauli-only Table 1へそのGやB・価格標本を混ぜない。P3は一区切りを維持する。

## 10. 先行研究・新規性の未確認事項とユーザー/GPTへの論点

[既存先行研究整理][PRIOR]には、RTE理論、既知のComposite simulation、Günther et al.、SPRINT/GRADE、RC-DF等との重なりを検討した別研究gateがある。
これは今回のH2/H4/H6 Pauli-only成果の新規性・priority・投稿可否を最終判定した資料ではない。
本作業では外部文献の新しい網羅的検索、個別論文の再レビュー、新規性/先取性の比較採点を実施していない。
既存アルゴリズムを組み合わせたresource studyとして何が独立した寄与になるかは**未判断**であり、既存gateの採択をそのままPauli独立論文の受理判断に移さない。

ユーザー/GPTへ戻す具体的判断は次のとおり。

1. 中心範囲を「有限候補・保存価格標本・仮想ρ=0の条件付き比較」に置くか。置く場合、A/A-1/Bの全残課題を解消することを必須にしない。
2. 案Aの独立resource case studyか、案BのDF比較・補助としてまとめるか。現在のDF B2/RPE目的を変更する判断とは分ける。
3. 機構claimは会計上のN/C競合とP1/P2の固定例までか、保持prefixの独立因果寄与まで求めるか。後者には現証拠で足りない識別条件がある。
4. 母期待価格または実装精度込み順位が本当に論旨に必要か。必要な場合だけ、B側の期待価格証拠、A/A-1側の全law誤差とN/G接続を独立に選ぶ。
5. 既存先行研究との具体的な差分・主張・対象読者をどう定めるか。必要な詳細科学レビューや外部文献調査は、その範囲を別に明示して開始する。

これらは研究判断の材料であって、本追補による正式採択や新科学計算の実行許可ではない。

## 11. 技術照合・保全・終了

独立worktree/branch `pauli-pure-rte-paper-evidence-addendum-20261102`／`codex/pauli-pure-rte-paper-evidence-addendum-20261102`を、固定A-1 commitから作成した。
A/A-1/Bのbranchのmerge/rebase・書換えは行わず、Bは固定commitのGitHub本文とlocal Git objectから読取した。
旧2文書が4591版のbytesと一致することを確認し、既存の純粋lookup `one`を再利用して一次6 CSVの候補・G・n・SEを照合した。
Table S1の60行（適格B0なしの4行を含む）とTable 1の15行が対応し、別の保存比較CSV3系列で15個のH/R文字列・保存classificationも一致した。
H6 H32/H64の各5精度の保存`versus_RTE`判定と、主図/関連CSV・coverage/nominationの存在を確認した。
数値tokenは文字列のまま読み、G/比/SE/Nの演算や新しいwinner選択をしていない。

本書と必要な索引・status/manifest・ノートだけを追加し、既存論文本体・Methods/Results/figure map・画像・CSVは変更しない。
全科学SOURCE/raw/input/result/QPY/grant/ledger/marker/STOPを保全し、旧manifest entriesも変更しない。
新しい科学/人工数値testsを実行した件数は0。文書参照・hash・転記一致の技術確認を、独立科学レビューや新しい性能検証のPASSとは呼ばない。
共有library/環境・他worktree・他利用者processへ変更を波及させない。不要なarchive展開やrawの重複処理は行っていない。
最終文書整合性確認はPASS：変更は文書/索引/status/manifestの7ファイル、旧manifest211 entriesの内容不変、固定GitHub参照28件のGit objectとlocal link1件が存在、
B公開本文のblob SHAとlocal固定Git blobが一致、旧2草稿・A/A-1・局所STOPのbytes不変、`git diff --check`異常なし。
研究全体地図/gap表は4591に未収録なので、存在を確認した固定carrier f36437a6から引用した。作成日と収録commitを混同していない。

**文書commit/push・remote SHA/公開本文照合後mandatory STOP。** 既存のSTOPは解除せず、追加計算、scalar評価、Compiler証明、tail再解析、原稿改訂、詳細科学レビューへ自動移行しない。
次の担当：**ユーザー/GPT**が上記の中心claim・位置付け・追加検証の必要性を判断する。**Codexの次の実行作業なし。**

[PAPER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/pauli_h2_h4_h6_paper_evidence_synthesis_20261101.md
[CLAIMS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/pauli_h2_h4_h6_claim_figure_map_20261101.md
[OBJECTIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f36437a662212001abcbd5b65a8cf138cd3da0f5/docs/research/研究目的・研究課題.md
[LANDSCAPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f36437a662212001abcbd5b65a8cf138cd3da0f5/docs/research/pr_repository_scientific_landscape_20261101.md
[GAPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f36437a662212001abcbd5b65a8cf138cd3da0f5/docs/research/pr_repository_claim_evidence_gap_matrix_20261101.md
[SUMMARY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/pauli_jw_resource_evaluation_summary.md
[OLD15]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/data/pauli_jw_summary/latest15.csv
[H2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/docs/h2_pauli_rte_validation_20261026.md
[H2COMPARE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/artifacts/h2_pauli_rte_validation/2026-10-26/comparison_by_precision.csv
[H4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3fdfad880902b7ff35b710d22e1c9759f7087c2/docs/non_df_h4_pauli_pr_search_20261025.md
[EXT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/docs/pauli_rte_event_extension_20261026.md
[EXTCOMPARE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/best_by_precision.csv
[H6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/REPORT.md
[H6COMPARE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/COMPARISON_RESULTS.csv
[H6DETAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/comparison_details.json
[EVENTCSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/event_cost_trends.csv
[MECH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5432857b94c3714ea9f4d6031e493e8dfc308a91/docs/research/pauli_pure_rte_cost_mechanism_inventory_20261101.md
[P1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bfb4ca75437a9b3605a8f57b5f82100615dd034f/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/bias/REPORT.md
[P2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c9073274daee69c9b69d8db3dc332bf6132c2259/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/cost/REPORT.md
[LOCAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/71432728ee0140fb3c1f8bf30ecf51b2faba86ec/artifacts/h6_selected_implementation/2026-11-02/run01/REPORT.md
[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e1c7aae386986f6c1bb9483548671655447f516/docs/research/h6_whole_law_transfer_feasibility_20261102.md
[A1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f36437a662212001abcbd5b65a8cf138cd3da0f5/docs/research/h6_native_bridge_scalar_phase_feasibility_20261102.md
[TAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d5558344ed7b94b264f79cbbb9c7deaa6bf0071/docs/research/h6_pure_rte_hybrid_price_tail_evidence_audit_20261102.md
[P3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/426ccda5df044d2e94d4a18d98cde75a96ab1e6b/artifacts/df_pauli_matched_partition_validation/2026-11-02/finite_bit/run01/REPORT.md
[PRIOR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f36437a662212001abcbd5b65a8cf138cd3da0f5/docs/research/先行研究と未解決点.md
[H2SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/artifacts/h2_pauli_rte_validation/2026-10-26/SOURCE.json
[EXTSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/SOURCE_e755ecf581600192fd2745d7066e410cebd933d3.json
[H6SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/SOURCE.json
