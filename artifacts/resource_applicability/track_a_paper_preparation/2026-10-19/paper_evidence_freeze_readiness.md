# Paper evidence freeze readiness

2026-10-19 JST。有限標本・固定条件の証拠整理は**READY_WITH_EXPLICIT_LIMITATIONS**。機構分析の数値結果は**PENDING_NUMERICAL_AUTHORIZATION**。論文主張の科学的採択はユーザーの判断である。

| 項目 | 固定できる内容 | 残る条件 |
|---|---|---|
| H4 | 5ε、新A n8／旧fixed-anchor n32、強化測定subset16の保存点 | legacy DF12、登録96全体の最良ではない。n32を独立確認と呼ばない |
| H6 | 5ε、DF19、PF136、通常signedの保存点とmodel区分 | 非相殺の全登録domainは未確定。旧D有限subsetとeligible:nullの欠測を分離 |
| H8 | 5ε、DF28、PF184、新旧保存点と直接/model来歴 | 未観測rare、model残存、原profile bytes未確認 |
| H8 fixed n48 | 事前固定2候補の正常完了報告・点費用再現 | 母平均優位・全domain確認・外部再現を認定しない |
| boundary | 既存登録slice/q-r schedule・H4親domain制約 | singletonを内部にせず、境界外改善を推定しない |
| Figure 2 | 3系共通schema、固定contrast、算術、受入条件 | 新数値は未生成、production SOURCE/grantは未seal |
| Figure 1/3/Main Table | 保存値転記と出典索引、図表仕様 | 完成figureは未描画 |

## 入力準備の到達点

244固定contrastのfield存在をmetadataとして照合した。[readiness表](mechanism_input_readiness_DRAFT.csv)を参照。163枠はC/N参照fieldが存在し、81枠はcore入力欠測を保持する。これは数値一致や科学的適格性のPASS件数ではない。

H4の強化subset16・H6強化136・H8強化184を用いる通常factor1の主系列30 contrastはC/N参照fieldを結合できた。H8 n48主2条件のC/N参照も結合できた。一方、H4旧full-PF対照の一部はD要約／payloadに軸単価がなく、G/Nから補間しない。新A factor10欠測、H8固定PF非相殺不適格等は仕様どおりの欠測である。

normalization/bias/headroomの完全な軸別説明入力は全旧候補で揃ったと認定していない。新Aの保存accountingを固定し、旧signal未結合は説明欠測として明記。費用coreと説明の完了は分ける。全244枠を強制的に数値化しない。

公開fileのcommit/blob/SHAと、元契約が指定するprivate入力のbyte snapshotは内部に固定した。private SOURCE/hash/pathは公開projectionへ混ぜない。別チャットの通常変更や他worktreeのHEADはこのtaskの拒否条件へ含めない。

## 追加の承認が必要な内容

1. 機構表に必要な保存単価平均、回路単価比、shot比、G比、軸重み、normalization/headroom比、丸め因子、および元Gとの算術一致検査。
2. 明示的に承認する場合のpaired SEの再構成。差の検定・CI・bootstrapは本契約外。
3. 欠測入力を新生成して埋める場合は別の科学契約。本準備からsampling/compile/signalへ進まない。

Tier R、母平均保証、非Clifford／T／FT資源、全格子実測、scaling law、energy/RPE全roundは今回の証拠freezeに含めない。それらを論文化の必要条件とするかはユーザーが別途判断する。

## mandatory STOP

本準備完了後はSTOP。既存科学jobを止める意味ではない。次の担当は**ユーザー**：契約の対象・算術・欠測許容・SE再構成を確認し、保存値数値解析を承認するか判断する。Codexは自動実行・公開push・新レビューへ進まない。
