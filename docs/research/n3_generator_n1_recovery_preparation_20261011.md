# N3保証生成・N1技術復旧：次作業の準備

2026-10-11 JST。完了済みGPT科学レビューを受領し、二つの作業を別契約にした。
今の状態は**契約準備完了、実装・科学batchは未実施、Mandatory STOP維持**。
同じ結果の再レビュー、GPT付録replay、旧N3/N2再実行は行っていない。

| 優先度 | 作業 | 入口 | 現在の状態 |
|---|---|---|---|
| 最優先 | N3一体LMIから証拠を生成し、未使用入力へ限定移送 | [数学・移送契約](n3_generator_transfer_contract_v1_20261011.md) / [JSON](n3_generator_transfer_contract_v1_20261011.json) | 新算法の設計のみ |
| 高 | N1実保存vectorをdecodeし、同条件でN1のみ一回実行 | [復旧契約](n1_decode_recovery_contract_v1_20261011.md) / [JSON](n1_decode_recovery_contract_v1_20261011.json) | 修正・実形式回帰・科学再実行待ち |
| 完了 | N2固定SのL1 fit修正 | [元結果](n13_redesign_results_20261011.md) | 保存・追加作業なし |

N3ではμの最小許容値を代入し、一変数κ∈[0,c−U+δ]のPSD条件へ整理した。
数値小行列探索は候補生成に限定し、有理数PSDの別checkerだけが証拠を認める。
zero pivotやκ=0も扱い、生成失敗は数学的・物理的な不可能性と区別する。
この整理は今回の設計上の導出で、実装・計算・独立科学検証はまだない。

同classの12有理係数入力（active m4/5/7、ν1/2/3、外部一mode、非負density）を事前recipeで固定した。
結合振幅二つ・同normの方向二つを各baseへ用意し、結果を見て選別していない。
旧scalar、共通U/c norm-only、共通五点方向、直接生成、強いscalarを分け、
下界/trial改善を全primary対照へ公平に与える。係数packの作成・shape/hash確認だけで、受理・真値は未計算。

N1は元selector packをbyte単位で保持する。decode境界以外の科学式、44 factor/178 pair、
q1/2/4/8、誤差・wrapper・compiler・対照を変えない。
今はdecoderが直ったとは主張しない。後の実装で実pack全contextの回帰を通し、sourceを固定する。
その後にだけN1専用の一回実行へ進む。N3/N2を混合runnerで再実行しない。

利用者の今回指示は「新科学batch未承認」と明記している。
N3実装・科学synthetic検証・移送、N1修正source・科学synthetic検証・一回再実行は、
後の明示指示に従い、それぞれ別source-bound planとauthorizationを作る。
今回のmetadata監査、文書保存、GitHubへの準備資料公開は科学実行の認可にならない。

保存基点は`10d91e518520e6f3fd853c85d8eff078357b7edb`。
旧source・failure・一次結果は不変。他Trackはmergeせず、元rootのdirty状態を保つ。
今回のread-only監査は[scripts/verify_n3_generator_n1_recovery_preparation.py](../../scripts/verify_n3_generator_n1_recovery_preparation.py)、
証拠は[preparation artifacts](../../artifacts/n3_generator_n1_recovery/2026-10-11/preparation/)へ保存する。
検査対象はinput identity、metadata、上限、STOP、旧証拠の保持のみ。
solver/PSD数値検証、対角化、LP、compile、新科学tests、量子shots、GPU callsはすべてzero。
既存global manifest checkerのdescription欠落は旧保守事項として残し、global PASSとは報告しない。
今回のmetadata監査は全N1 contextの形式・件数、旧source/結果220 blob、旧manifest88 entriesの保持を確認した。
これはdecoder回帰・有理PSD証拠・算法の有用性を検証した結果ではない。
hostはread-only inventoryでCPU32、NVIDIA device nodeなしを確認。将来は小行列/係数cacheと最大2 CPU workersを使い、launch前に資源を再確認する。
中心テーマの採択・量子資源改善・一般二体への拡張は未判断。
