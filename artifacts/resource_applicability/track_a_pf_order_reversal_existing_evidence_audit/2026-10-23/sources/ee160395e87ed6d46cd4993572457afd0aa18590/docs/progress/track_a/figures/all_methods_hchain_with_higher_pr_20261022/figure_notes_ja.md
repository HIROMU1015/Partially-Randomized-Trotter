# 読み方・保存範囲・残る注意点

## 研究上の二つの比較

「同じouter32 PR内でS2から4次へ変える効果」と「PRが決定論PFより低費用か」は別の問い。
Bは前者を扱い、Cは後者のうち同じnative系列のm5部分集合だけを扱う。
Aは旧全方式比較と新系列を保存域別に並べた参考overviewで、両系列のS2を同一対照に置き換えていない。

直鎖H2/H4/H6、隣接1 Å、STO-3G、tol-only1e-8の保存DF rank3/10/19、T=.8/P=0。
新outer32 PRはprefix1/3/5、K4、q=1/2/4/8、δ=.8/.4/.2/.1、各outerの全piece合計32 short steps。
S2・3段Yoshida4・固定binary64 m5_bestを区別する。通常signed・経験的allowance倍率1・α_axis=.025の保存会計を使用。
costは保存解析shot込みのfull cosine/sine Hadamard-wrapper compiled RZ総費用。εは有限時間複素信号の要求精度で、energy精度ではない。
状態準備、実量子shot、noise、FT T count、全RPE/QPE round総費用は含まない。
新探索のnは候補ごとに33〜255で異なる。pilot・旧n8・確認cohortとpoolしない。
独立確認は固定commit c7c1d822で準備段階。提案n_Cを取得済み標本数へ読み替えない。

## PR高次化の既報値を再現する分母

ε=.0005の報告値は、次の固定条件を持つ**同じouter32系列内**の保存選択どうしの比である。
以下の改善率は保存評価のフィールドを転記し、統計再解析をしていない。

| 系 / DF rank / prefix | 基準 | 高次側 | δ（基準 / 高次） | 保存削減率 |
|---|---|---|---|---:|
| H2 / 3 / 1 | S2/q4 | Yoshida4/q2 | .2 / .4 | 47.122758% |
| H4 / 10 / 3 | S2/q8 | m5_best/q1 | .1 / .8 | 53.303863% |
| H6 / 19 / 5 | S2/q8 | m5_best/q1 | .1 / .8 | 63.123232% |

これらを旧全方式図のS2費用に対する削減率として表示してはいけない。
旧Aはprefix/q/r/Kを含む別の登録候補域から選択され、outer32のS2はprefix1/3/5・K4・固定32政策である。
特にH4の厳しい精度の旧S2採用点はprefix2で、新outer32はprefix3。同じ「PR S2」の文字だけで交換しない。

## 互換性の監査

固定8613b3ecのinput_provenanceは旧強化S2/S4対照を
`PRESERVED_NOT_REPLACED_NOT_IDENTITY_JOINED`、H2の決定論比較を
`NULL_NO_IDENTITY_BOUND_SAME_SERIES_SIGNAL_SET`と明記する。
Hamiltonian/state/snapshot/compiler hashの4項目はH2/H4/H6で旧図と一致するが、
それだけで全native/normalization/shot/実runtime政策の同一性を認定したことにはならない。
今回の互換性監査はその保存metadataを整理したもので、未結合対照の科学的比較を新たに認定していない。

このためAは2パネル。絶対図は左右共通y範囲で値を併置するが、点をパネル間で結んだり、
高次PR/旧PFの比を算出したりしない。相対図は左が旧best full PF、右がouter32 PR S2。
右の1未満はPR内部高次化の点費用減であり、旧PFへの優位性を意味しない。

H2の既存compiler位相帳簿guardや旧PF側のprofile/runtime留保も保持する。
元Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、symmetric_directionalとscalar onceが記録されているが、
同じcompiler SOURCEや実runtimeで一括再実験したとは表現しない。

## Cの決定論候補域

| 系 | m5のmethod | 登録prefix | q | 保存状態 |
|---|---|---|---|---|
| H4 DF10 | B0/B0+I | 3,8,9 | 1,2,4,8 | 24候補。8候補は全対象精度で不適格、残り16候補に単価 |
| H4 DF10 | full | 10 | 1,2,4,8 | 4候補に単価 |
| H6 DF19 | B0/B0+I | 5,7,11,15 | 1,2,4,8 | 32候補。一部不適格、保存適格候補だけ表示 |
| H6 DF19 | full | 19 | 1,2,4,8 | 4候補に単価 |

全64候補中48候補の96軸が直接単価区分（92新軸＋4保存再利用）。残る16候補の単価未保存と不適格は別情報。
全適格行で保存Gが存在することを確認し、未保存単価を補完しない。候補を新たに追加していない。
方式別の最小G抽出は依頼で許可された表示操作だけで、新しい科学的順位表・winner認定を作っていない。
通常signedのm5部分集合最良は、保存summaryの既存候補ID・exact費用と一致する。
この部分集合の結果から、旧D190/強化136/184を含む全PF最良に対するPR優位性を主張しない。

非相殺ではB0/B0IのPF/discard分離成分が未保存。1000行表の280行をMISSING_COMPONENTS・費用nullとして保持する。
C非相殺のfull PF m5は別の部分診断であり、全m5部分集合最良の代替ではない。
同じnative系列のS2決定論集合はこの固定保存評価にないので、Cへ旧S2の値を入れない。

## 表示したouter32採用点

下表の全点は通常signedの新探索標本。全てT=.8/P=0、K4、outer32、tol-only1e-8、上記rank/prefixである。
非相殺の採用候補IDは同じだが整数shot・費用にはroundoff級の差があるため、保存NONCANCELLATION値を別に使用する。

| ε | 系 | formula | candidate | prefix | q | δ | n |
|---:|---|---|---|---:|---:|---|---:|
| 0.05 | H2 | 2nd | `H2O32_5e9f36e6e206954e` | 1 | 1 | 4/5 | 96 |
| 0.05 | H2 | 4th | `H2O32_450887092a956e28` | 1 | 1 | 4/5 | 60 |
| 0.05 | H2 | 4th(m5_best) | `H2O32_89c00dc5fccfe88d` | 1 | 1 | 4/5 | 33 |
| 0.05 | H4 | 2nd | `M5_MIN_8ba95fafad862531` | 3 | 1 | 4/5 | 255 |
| 0.05 | H4 | 4th | `M5_MIN_7e41d6aceeb812a3` | 3 | 1 | 4/5 | 54 |
| 0.05 | H4 | 4th(m5_best) | `M5_MIN_e49b5334798ef821` | 3 | 1 | 4/5 | 33 |
| 0.05 | H6 | 2nd | `M5_MIN_a6d345ce4b40f7a5` | 5 | 1 | 4/5 | 167 |
| 0.05 | H6 | 4th | `M5_MIN_ce073ae6e29fc116` | 5 | 1 | 4/5 | 33 |
| 0.05 | H6 | 4th(m5_best) | `M5_MIN_37c3e8eb2b18f310` | 5 | 1 | 4/5 | 33 |
| 0.01 | H2 | 2nd | `H2O32_5e9f36e6e206954e` | 1 | 1 | 4/5 | 96 |
| 0.01 | H2 | 4th | `H2O32_450887092a956e28` | 1 | 1 | 4/5 | 60 |
| 0.01 | H2 | 4th(m5_best) | `H2O32_89c00dc5fccfe88d` | 1 | 1 | 4/5 | 33 |
| 0.01 | H4 | 2nd | `M5_MIN_d8a8c6781cb3cea0` | 3 | 2 | 2/5 | 143 |
| 0.01 | H4 | 4th | `M5_MIN_7e41d6aceeb812a3` | 3 | 1 | 4/5 | 54 |
| 0.01 | H4 | 4th(m5_best) | `M5_MIN_e49b5334798ef821` | 3 | 1 | 4/5 | 33 |
| 0.01 | H6 | 2nd | `M5_MIN_dcf35cf1720c739b` | 5 | 2 | 2/5 | 110 |
| 0.01 | H6 | 4th | `M5_MIN_b702d0ea01460683` | 5 | 2 | 2/5 | 33 |
| 0.01 | H6 | 4th(m5_best) | `M5_MIN_37c3e8eb2b18f310` | 5 | 1 | 4/5 | 33 |
| 0.005 | H2 | 2nd | `H2O32_64c6f1cbf957bd1d` | 1 | 2 | 2/5 | 71 |
| 0.005 | H2 | 4th | `H2O32_450887092a956e28` | 1 | 1 | 4/5 | 60 |
| 0.005 | H2 | 4th(m5_best) | `H2O32_89c00dc5fccfe88d` | 1 | 1 | 4/5 | 33 |
| 0.005 | H4 | 2nd | `M5_MIN_46221d66178510d9` | 3 | 4 | 1/5 | 62 |
| 0.005 | H4 | 4th | `M5_MIN_49e3ac75076339d1` | 3 | 2 | 2/5 | 33 |
| 0.005 | H4 | 4th(m5_best) | `M5_MIN_e49b5334798ef821` | 3 | 1 | 4/5 | 33 |
| 0.005 | H6 | 2nd | `M5_MIN_645cd1aeaf52123d` | 5 | 4 | 1/5 | 33 |
| 0.005 | H6 | 4th | `M5_MIN_b702d0ea01460683` | 5 | 2 | 2/5 | 33 |
| 0.005 | H6 | 4th(m5_best) | `M5_MIN_37c3e8eb2b18f310` | 5 | 1 | 4/5 | 33 |
| 0.001 | H2 | 2nd | `H2O32_c7ec590a32a859bc` | 1 | 4 | 1/5 | 33 |
| 0.001 | H2 | 4th | `H2O32_8a2ecdd0294ffb74` | 1 | 2 | 2/5 | 34 |
| 0.001 | H2 | 4th(m5_best) | `H2O32_89c00dc5fccfe88d` | 1 | 1 | 4/5 | 33 |
| 0.001 | H4 | 2nd | `M5_MIN_72b305020499dd68` | 3 | 8 | 1/10 | 48 |
| 0.001 | H4 | 4th | `M5_MIN_49e3ac75076339d1` | 3 | 2 | 2/5 | 33 |
| 0.001 | H4 | 4th(m5_best) | `M5_MIN_e49b5334798ef821` | 3 | 1 | 4/5 | 33 |
| 0.001 | H6 | 2nd | `M5_MIN_a08e8bfb6642eb93` | 5 | 8 | 1/10 | 33 |
| 0.001 | H6 | 4th | `M5_MIN_b702d0ea01460683` | 5 | 2 | 2/5 | 33 |
| 0.001 | H6 | 4th(m5_best) | `M5_MIN_37c3e8eb2b18f310` | 5 | 1 | 4/5 | 33 |
| 0.0005 | H2 | 2nd | `H2O32_c7ec590a32a859bc` | 1 | 4 | 1/5 | 33 |
| 0.0005 | H2 | 4th | `H2O32_8a2ecdd0294ffb74` | 1 | 2 | 2/5 | 34 |
| 0.0005 | H2 | 4th(m5_best) | `H2O32_89c00dc5fccfe88d` | 1 | 1 | 4/5 | 33 |
| 0.0005 | H4 | 2nd | `M5_MIN_72b305020499dd68` | 3 | 8 | 1/10 | 48 |
| 0.0005 | H4 | 4th | `M5_MIN_ab57ad5bd40dc583` | 3 | 4 | 1/5 | 33 |
| 0.0005 | H4 | 4th(m5_best) | `M5_MIN_e49b5334798ef821` | 3 | 1 | 4/5 | 33 |
| 0.0005 | H6 | 2nd | `M5_MIN_a08e8bfb6642eb93` | 5 | 8 | 1/10 | 33 |
| 0.0005 | H6 | 4th | `M5_MIN_37ff3428bb58f5cd` | 5 | 4 | 1/5 | 33 |
| 0.0005 | H6 | 4th(m5_best) | `M5_MIN_37c3e8eb2b18f310` | 5 | 1 | 4/5 | 33 |

各図の高次bestは、Yoshida4とm5_bestの保存formula採用点のうち低い方で、S2を含まない。
同値の候補はexact_tiesへ保持する。他の高次公式を旧pilotから混合しない。
今回固定された現在の36 PR群はS2/Yoshida4/m5_bestだけで、追加の同条件formulaはない。

## 不確かさ・独立確認

保存費用はpaired cosine/sineのG平均。保存SEには同一trajectory内の共分散を含むが、今回再計算していない。
同じ候補の5精度は同じtrajectoryを共有するため、5個の独立な実験ではない。
formula/q選択は同じ探索標本上の点最良で、選択効果補正・母平均CI・rare層上界はない。
全36群でTaylor S4層は未観測。これはouter PFの4次を未実行という意味ではなく、RTE eventの次数層の話である。
S2_multiが1観測の群以外にも未観測層があり、0観測費用・分散を0に補完しない。
保存observed_strata_144.csvのnullを保持した証拠コピーから追える。

c7c1d822の独立確認は準備完了・science_started=false・有効grantなし。
主要3比較の提案n_C、H4補助ε=.005の設計、fixed_shotsは準備情報だけで、費用の取得済み確認結果ではない。
図のnに提案n_Cを使わず、独立確認曲線・CI・新標本設計を作らない。
旧H2/H4/H6/H8の独立確認Pも、新outer32の独立確認の代替にはしていない。

## カバレッジと「作れる範囲」

要求5精度×保存method×適用scopeを全列挙した。A10図、B通常10＋非相殺10、C通常10＋非相殺full-only10、D2図。
scopeを変えた無制限な組合せや未登録q/prefixの補完を「すべて」に含めない。
未作成の比較は次の理由を持つ：旧全方式PFと高次PRの直接overlay/比は未結合、
H2の同一系列決定論対照は未保存、H8outer32は未保存、非相殺B0/B0Iは成分欠測、
同一系列S2決定論対照は未保存・未結合、独立確認は準備段階。
旧H2のε=.01以下の打切りPFは「保存済み不適格」で、未保存と区別する。

statusは表示・比較可能性・独立確認の複数軸を持つ。A高次はnot comparableとplottedの両方が正しいが、
対象はそれぞれ「旧PFと直接比較」と「別outer32パネルでの表示」である。
監査図DのOKはplotted、Iはsaved-but-ineligible、Mはmissing、NCはnot comparable。
全ての理由・source・candidate・費用はcoverage_matrix.csvの600行で追える。

## 保存・検証範囲

原10図・元の主表・SOURCE・STOP・provenanceを変更せず、別directoryへ保存した。
新たに行った演算は、保存fractionの読取・表示用最小抽出・分母が明示された表示用費用比だけ。
新しいmean/SE/shot/総費用・signal・sampling・compile・fit・bootstrap・科学的再ランキングはない。
表示前にexact tokenを元JSONへ照合し、Bの表示比60件を保存exact比へ、C通常の分母を保存m5最良へ照合した。
PNG/SVG/PDFの形式・canvas配置・PDF実renderとPNGのcontact sheetを確認する。
これらは作者Codex自身のローカル技術確認で、独立科学レビュー・immutable CIや最終freezeではない。
