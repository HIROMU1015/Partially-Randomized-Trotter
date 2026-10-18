> 公開用projection。private内部パス、原入力・回路・運用記録、ローカル補助scriptは含まない。本文に記す照合件数は原調査の報告値であり、未公開原資料を再公開した意味ではない。

# D：H6非相殺会計の来歴確認

調査完了。全登録domainの非相殺PR削減率は未確定のまま。新しい費用・shot・順位・統計量は計算していない。

## 固定条件と区分

H6 linear、隣接1 Å、STO-3G、tol-only 1e-8の保存DF19、保存state、T=.8、P=0、Qiskit1.3.0のcompiled full-wrapper RZと保存shot。通常signedと経験的非相殺を分ける。非相殺は厳密bias上界ではない。

| 系列 | 固定結果commit / 科学SOURCE | 比較domain・標本 | 確認した状態 |
|---|---|---|---|
| V1の元会計（fc6bdにも収録） | `fc6bd039dc7171d5674ac04f0b3ad427f1b6495b` / 元V1 `2261a3ad1d6c23521474b31204572a6c3a51258b` | PF旧32 / S2強化96 / S2+S4強化136、PR確認union8 n32、5精度、倍率1/10 | 旧B0I12件の非相殺は`eligible:null`, `UNRESOLVED_SIGNED_DECOMPOSITION`。全登録最良は未確定 |
| D_ACCOUNTING | `fc6bd039dc7171d5674ac04f0b3ad427f1b6495b` / `31ef62a12407010e9c1a07f9b37facedb8dff5fe` | 旧32 / 強化136、確認union8 n32、5精度、倍率1/10。取得済み追加直接単価を反映 | 有限な削減率は利用可能subsetの点比較。保存`unresolved:[]`は完全domainの証拠に使えない |
| α感度 | `343e0c591d297f99174459d750206b01bf7b9412` / `93145a4b24aa2e77f1eefdb39c2d70c56f9bddcb` | H6 PF136＋確認PR8、5精度、倍率1、総α=.01/.05/.10 | 60入力欠測、3αの180出力欠測。H6非相殺の15比較は未確定 |
| 新A統合 | `64c85f7a833b5950ef2ccae623708b73081edcd6` / `8601b921342fc1cc83780a830e1f65bf64fe9e93` | 旧32 / 強化136、旧PR n32と新A16候補 n8を別々に比較、5精度、倍率1/10枠 | 旧12候補の未確定を保持し、全domainの新旧PR/PF比をnull。新A倍率10の保存shot欠測は別問題 |

εの5点は`.05/.01/.005/.001/.0005`。異なる系列のPF単価更新・α・標本を同じ結果と扱わない。

## 欠測の同一性と発生箇所

旧B0I12件は`p={5,10,15} × q={1,2,4,8} × formula=2nd`。新Aのprefix3/4/5や、強化S4のp11..15とは別の登録候補である。

元V1の`candidate_accounting.json.gz`では、この12件の非相殺行は保存signed分解とbaseline会計がなく、適格性はFalseではなくnull。倍率1/10を含む120条件行が保存されている。α入力bindingは倍率1だけを使うため同じ12件×5精度の60入力が欠測である。新Aの`comparisons.csv`のH6非相殺行の`unresolved_*_domain_ids`と、αbindingの12個のID集合が一致する。

H6/H8の新A倍率10の各160 endpoint欠測は、新A側に保存shotがないことによる。旧B0I12件の非相殺分解欠測と混同しない。通常signedからの代用、倍率1の流用、他候補からの補完は今回も行わない。

## Dの有限値を全domain値と読めないコード上の理由

SOURCE `31ef62...` の`prepare_pr_cost_batch.py:187`以降は、元V1の`eligible:null`と`G_exact:null`をD payloadに保持する。値は消去されていない。

一方、同SOURCEの`pr_cost_batch_science.py:98`以降は、利用可能な最小点の抽出と未確定の検出に、どちらも`r['eligible']`の真偽値を使う。`eligible:null`は真でないため、`unresolved`に入らない。したがって保存Dの`unresolved:[]`と有限な比が、元の12件の未確定と両立している。

これは**未確定候補の検出漏れという静的な技術所見**である。元結果の数値再解析や修正は実施していない。有限値そのものを誤計算と認定したのではなく、full-domainとする解釈が不成立である。

新A SOURCE `8601b921...` の`pr_new_a_accounting.py:98`以降の`old_row`は、`eligible is None`を`NUMERICALLY_UNRESOLVED`へ明示変換する。そのため新Aでは同じ保存旧rowを使ってもfull-domain比較をnullにする。新Aで欠測が新たに発生したという意味ではない。

## 転記できる既存subset点比較

下表はD完了報告からの転記（強化集合、倍率1、P=0）。比は今回計算していない。各PF ID・exact fraction・単価区分は`stored_subset_comparisons.json`にも保存。

| ε | PR削減率（保存値%） | 取得済みsubsetのPF選択ID | PF単価の保存区分 |
|---:|---:|---|---|
| .05 | 22.2791 | B0I_p11_q1_2nd | VALIDATED_MODEL_ESTIMATE |
| .01 | 32.2240 | B0I_p11_q2_2nd | VALIDATED_MODEL_ESTIMATE |
| .005 | 37.0663 | B0I_p11_q4_2nd | VALIDATED_MODEL_ESTIMATE |
| .001 | 36.2830 | B0I_p11_q2_4th | VALIDATED_MODEL_ESTIMATE |
| .0005 | 20.1071 | B0I_p11_q4_4th | DIRECT_MEASURED_SAVED_SHOTS |

全候補が直接実測という意味ではない。確認PR n32も有限標本の点推定であり、母平均優位を示さない。元の名目集合名`strengthened_S2_S4`だけでは、非相殺12候補の未確定は解消しない。

## 論文で使える記述と避ける記述

使える記述例：

> H6の経験的非相殺会計は、旧B0+I12候補のsigned成分分解が保存されていないため、全登録集合での比較は未確定である。取得済みsubsetに限定したD会計では、保存された5精度のPR点費用削減率は22.2791/32.2240/37.0663/36.2830/20.1071%である。このsubset値にはPFモデル推定を含み、全登録集合の最良費用を示さない。

避ける記述：

- 「H6全136候補に対して非相殺でも22〜37%低費用と確定した」
- 「α感度の60欠測は、新Aで補われた」
- 「Dが完了しunresolvedが空なので、全非相殺比較は成立した」
- 「非相殺会計は厳密なbias保証／母平均勝者を与える」

## 判定と追加承認

- 確認済み：同じ12候補の欠測来歴、Dのtruthinessによる検出漏れ、新Aのnull保持、αの60入力と3α展開の区別。
- 条件付きで引用可能：Dの利用可能subsetの有限標本点比較。
- 未確認・未確定：旧12候補の保存分解の回収可能性とfull-domain費用比。
- 不成立：Dの有限subset比を全登録domainの最良比とする解釈。

全登録domain値を必須としない論文化準備は、この欠測とsubset限定を明記して進められる。全登録domain値が必要なら、まず欠測分解の既存記録の有無を絞って調査し、既存値で埋まらなければ必要入力・SOURCE・対象endpoint・再会計／再ランキングを別承認で固定する。今回は旧SOURCE・結果・STOPを変更せず、修正実装・再実行もしない。
