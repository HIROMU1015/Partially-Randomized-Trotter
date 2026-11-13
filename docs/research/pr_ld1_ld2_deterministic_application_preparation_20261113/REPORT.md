# 保存PR-LD1/LD2：初回決定論的risk適用の準備

2026-11-13 JST。B基準`d71721f30c7b7fc13ba0789eefffd6238aca3ebc`、既存branch `codex/rpe-decoder-statistics-20261113`を再利用した。保存SOURCE・metadataと未採択条件を整理した。**保存rawから確率・位相・unsafe/zero/tie・MSE・必要Nを計算していない。MC・実C/G・最適化も未実行。**

## 確認できたこと

固定A `91349e703e0d09ef702ccd108745dfaf1acc29f7`のB_MEANS_LD1/LD2を、元configuration `a9abf8de75bf3432fcd714e868e5c14c8884b8f8`、数値実装`fefd548ecd56f60d72031b3d385f6f5e6ee95d06`、raw carrier `16210932e5dabe7d12d41f35fea0e0a0d7a9b271`と照合した。各13roundのidentity/law/SOURCE・C provider_bindingは既存K.accept_savedで対応する。元bundleやcarrierを変更していない。

| 項目 | 保存metadataの状態 |
|---|---|
| LD1 canonical ID | `a4dda8f7e3fc07666ca2aabf634917e8bced14eb62151eb629ef23c0cce21d3d` |
| LD2 canonical ID | `1fa33556c17303f51ba03c4eae11189d35e59b3f57faad0e69d9fd08bf9245ae` |
| 共通物理系 | 元H4 DF10/linear 1Å/STO-3G、保存H/state/snapshot、元DF-H target |
| 固定schedule | S2、ΔPF=h=.1の元binary64由来exact値、K2、M12、元共通κ。prefixは各LDのまま |
| decoder | consecutive_dyadic_nearest_branch_principal_energy_v1、元R3、prior/branch/alias/shift0を保持 |
| raw形式 | 各roundのreal/imagは保存JSONの有限binary64 float。範囲・形式確認のみ、確率への変換なし |
| target | solver参照値はあるがenergy_haはなく、新proxy採択false、ground資格false |
| 科学資格 | 全mean UNQUALIFIED。元policy.scientific_statusはPROPOSED_NOT_ADOPTED、経験的数値gateを全RPE保証へ昇格しない |
| N/G | 両bundleのshots/sufficient_N/global_RMSE_bound/G/winnerはnull |

全26roundのphysical candidate、T/qPF/R_outer/event、raw SOURCEとlaw locatorを[SOURCE_AND_CONDITIONS.json](SOURCE_AND_CONDITIONS.json)へ**コピー**した。raw数値や新しい統計量は保存していない。元solver参照値はそのまま引用し、bba1508のPHYSICAL_INPUTS.targetへの一致を確認した。SOURCE/input/rawの数値精度・不確かさは元参照で保持する。

## 初回の提案とユーザー判断

[CONDITIONS_AND_FINITE_PLAN](CONDITIONS_AND_FINITE_PLAN.md)に、既に固定された条件と未採択条件を分けた。推奨案は、**full DF-H solver proxyに対する条件付きrawモデル診断を、LD1/LD2各1本・計2評価だけ実施**すること。正式な十分N判定、N探索、LDの優劣、Minkowskiへの適合、Gは判定しない。この案は未採択である。

評価target、未解決loss family、確率rule/IIDの今回の適用scope、共通whole-shot vector、phase安全集合、有限検査範囲が未選択。機械可読ファイルのselected_inputsはすべてnullで、実行設定や承認証明書ではない。旧S3のnominal配分は任意のcontrol案として転記しただけで、別configurationのN資格やRMSEをLD1/LD2へ移していない。

## 有効Hamiltonian・AのδEとの分担

利用者が採択した**PFのTrotter誤差をH_effの固有値シフトで主に評価する方針**は保持する。これはBのtargetやriskを自動変更する指示ではない。

| 評価 | 扱い |
|---|---|
| AのδE | 対応するPF unitary・H/state/sector/formula/幅に結び付く誤差証拠。Bで再計算しない |
| B：raw、target=元DF-H proxy | shot/branch/aliasに加えて時間発展近似・finite-K/normalization・保存stateの影響も含み得る全rawモデルrisk |
| A+B | Bのriskを純shot誤差と同定せず、δE又はδE²を機械的に加算/控除しない。HaとHa²の区別だけでなく、誤差の重複とcoupling条件の確認が必要 |
| 元Minkowski | 元budgetを保持するが、今回の直接rawモデル診断だけで各予算項への適合を認定しない |
| PR finite-K | PFのH_effをHermitian generatorとして流用しない。別の成立根拠なしに単一位相・固有値shiftへ還元しない |

A/Bの正式な誤差統合が未確定でも、この条件付き診断は独立した結果として報告できる。AのδEを待ってB単独の診断を止める必要はない。一方、Aの新しいδE数値や有効Hamiltonianを今回取得・解析・適用していない。

## 技術整合と確認範囲

新しいdecoder/controller/科学runnerは不要だった。[HANDOFF](HANDOFF.md)のK.accept_saved→K.prepare→既存D.evaluateを使用する構成で、後段は既存W.compare/B.selectを維持する。B worktreeにはCの全provider graphを複製せず、固定Cのmetadata関数を既存test fixture経由で照合した。C全runtime・signal producer・price providerを実行した証拠ではない。

[CHECKS.txt](CHECKS.txt)：既存の保存fixtureテスト**2件PASS**。加えて全26binding/形式/target lineage・原nullを確認した。Dの数値入口、decoder/statistics、MC、G、physical providerをmockで禁止して行った。d717の45件のrisk/会計テストは今回は再実行していない。

固定Pythonは3.11.0rc1で整数decimal桁数制限APIがなく、人工の2^20000分母は標準JSONでexactに往復できた。これは保存risk数値の分析ではない。将来の安定版Pythonで制限がある場合は既存integer_io_limit等の適用を検討できるが、今回serializerや共有設定は追加・変更していない。

初回は価格評価を行わないので、C/Dへ新しい価格要求一覧を発行していない。将来は今回確認したcanonical round/physical ID・SOURCE・profileを既存価格requestへ渡す。適格E[C]や正式Nが不足する間はG/winnerをnullに保つ。

次は**ユーザー**が未採択条件と有限2評価を選び、科学実行を明示許可した後に、**B**がその範囲だけ適用する。許可はチャットの指示で足り、追加のgrant・証明書は不要。今回はcommit/push・remote照合後に**mandatory STOP**。
