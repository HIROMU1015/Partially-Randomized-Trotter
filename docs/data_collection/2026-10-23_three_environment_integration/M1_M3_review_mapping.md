# Level 1レビューM1〜M3の対応

レビュー原文はユーザー指定のローカル文書。識別SHA・非公開状態は `review_source_receipt.json`。本作業は原文の科学的再レビューや新規結論の採択を行わず、§11のM1〜M3を参照索引へ対応付けた。レビューの中心claim案（§10）と追加検証案は提案のまま扱う。

## M1：主張と比較単位

`claim_evidence_overlay.csv` は既存CEM21行をID・元行番号で再利用する。`scientific_profiles.json` は目的、Hamiltonian/geometry/basis/DF、候補・時間・精度、cohort、費用定義、SOURCEと監査の入口を対応付ける。正確な各候補のprefix/q/r/K・保存値・単位は固定CSV/JSONのまま参照する。未記録fieldは推測しない。

- 主表は既存H4/H6/H8の独立固定endpoint6行。H4粗精度NOT_REPRODUCED、H6粗精度PF有利、H8のpublic audit未再検証を保持する。
- 共通588候補は採択済み固定域の探索点結果。15全式比較と45同一式比較、36PR探索groupとPF候補域を区別し、PR/PF全探索自由度が同等とはしない。全588候補評価と全候補compileは別。
- 再構成118軌跡はPR同士の固定確認で、新規PFとの独立確認ではない。旧run01/run02、高次PR探索、旧A n8/union n32、新S2 cohortは別。
- H8 mechanismの別名は同じ観測であり独立標本を増やさない。H4 DF12とDF10、元H2回収と新H4/H6/H8も結合しない。

結果：固定参照への対応は作成済み。全private原本のidentity成立・全SOURCE閉包を再認証したという意味ではない。主表/Figure2の差替えとAD03は未判断のまま保全。

## M2：誤差・費用契約

レビュー§4/§6/§9と既存SOURCEの静的読解を対応付けた。元signal契約 `4ec26921...:scripts/h46_m5_signal_contract.py`、shot契約 `9a77d2b...:scripts/d0_frozen_shot_rule_v1.py` は `references.csv` のM2_POLICY参照。コードをimport・実行していない。

- signed biasは保存固定入力の数値参照との差であり、未知信号を普遍的に予測する証明書ではない。経験的u、reference誤差未保証、normalization Bの来歴を維持する。
- 軸ごとのheadroomとELIGIBLE/INELIGIBLE/UNDETERMINEDを区別する。欠測bias/u/非相殺成分を0やfalseへ変換しない。Bとexp(log B)のbinary64相違を新転記で修正しない。
- analytic Nは宣言された有界fresh-IID測定モデルと経験的誤差予算に条件付けたHoeffding十分配分。必要最小shotや実機測定回数ではない。量子shot Nとcompiled費用MC標本数nは別。
- 保存Gはcos/sin軸別のNとcompiled RZ費用Cを結合する。MCのpaired軸共分散と量子測定失敗率は区別。DF閾値を厳密電子エネルギー誤差保証へ変換しない。
- CNGのG00/G01はN固定のC置換。N_circuit/N比/G10/G11はnull。新推定器や独立Nの取得を本主張の前提にしない。PR法則期待値と有限MC平均の差までcompiler因果効果へ帰属しない。
- COMMON588の採択契約と新H2共通政策/COMPAREは別。O00002発見、metadata/prep hashの一致だけでevent/runtime/operation/correctness joinの完全性を認定しない。現在のCOMPARE契約はBLOCKEDを維持する。

結果：条件・来歴の区分を対応付けた。保存値の再評価、契約の科学的同値性認定、新しい厳密bias/母平均保証は行っていない。旧新A shot原表の欠測やH6/H8新S2補助FAILなどは主張依存の制約として残る。

## M3：統計制約・先行研究

レビュー§5〜§8/§10の限定をclaimへ対応付けた。有限標本の点費用と名目SE/区間は、未観測stratumを含むcompiled母平均winnerの保証ではない。未観測費用はUNKNOWNであり0ではない。H8は既存正式closureの層別有限標本と残余条件付き感度の範囲を維持し、追加sampling/compileを必須化しない。

H6/H8新S2のunweighted-G補助FAILは条件付き表示に残し、別SOURCEのCOMMON588を包括的に否定する扱いもしない。H6旧COSTの保存187組受入・fresh22組と、元親processの終了状態未知を区別する。旧run01 H6 accepted0はNOT_EVALUATEDである。

先行研究は既存 `501b4232...:docs/research/prior_art_novelty_20261021/{report,references}.md` とLevel1レビューR/P参照を再利用する。今回、新しい文献調査・原論文の数値再現・引用版の再照合はしていない。レビューが挙げたPRX Quantum/Günther（DF、shot、truncation、synthesis）、Hagan–Wiebeのcomposite simulation、および後続PR関連研究は既知要素との対応先であり、新発明という採択をしない。引用版と強い新規性記述の精査は、該当claimを採用する場合の執筆時確認に限る。

結果：固定native-DF/P=0/T=.8/登録候補域の評価範囲を保つ。普遍的PR非優位、先行PR論文の反証、母平均最良、4次PFによる逆転原因、FT T/実機noise/全RPE最終費用は支持しない。レビュー§10中心claim案の正式採択は今回行わない。
