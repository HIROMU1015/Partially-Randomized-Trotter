# PR-LD1/LD2：全round compiled価格requestの接続

2026-11-13 JST。**A固定menuの2構成・各13roundを既存D.from_bへ接続し、26 paired requestの選択索引と52軸のcoverageを生成した。全requestがmetadata上は対応し、科学実行へ選択したものは0である。** 実H4のpreparation・sampling・回路build/compile、保存科学値の再解析、残差fit、新E[C]/Gは行っていない。

[PRICE_REQUEST_COVERAGE.json](PRICE_REQUEST_COVERAGE.json)／[round別一覧](PRICE_REQUEST_COVERAGE.md)／[選択索引](PRICE_REQUEST_INDEX.json)／[A/B/C HANDOFF](HANDOFF.md)／[NEXT_SCIENCE_PLAN](NEXT_SCIENCE_PLAN.md)／[SOURCE・テスト](SOURCES.md)。

## 固定SOURCEと接続

- A menu：`a9abf8de75bf3432fcd714e868e5c14c8884b8f8`のMENU、CONFIGURATION_LD1/2、SCIENCE_PLAN/HANDOFF。canonical configurationを変更・生成し直して採択せず、Bの既存validatorによる再現照合だけを行った。
- D基点：`0c6e16342b3c804fa0bd89a4028d6b5bea7b0322`、同じ既存D branchを使用。元prepare/compose/Accounting/compiler/hash/cacheは今回起動・変更していない。
- B：`437ba4acdb82708b6fc98e3cd119a2a734f7cf64`、`b9b9ffa6233db9dd2b413dc07b27e7039324c666`。controllerとfixed-grid integrationの該当fileはD保存版とbyte一致を確認し、価格extension/sidecarとwhole-shotの資格条件を継承した。

共通はLinear H4/1Å/STO-3G、保存tol-only DF10/sector(2,2)、S2/ΔPF=.1/K2/h=.1/M12/P0、元H/state/snapshot・exact有理数κ。LD1 configurationは`a4dda8f7e3fc07666ca2aabf634917e8bced14eb62151eb629ef23c0cce21d3d`、LD2は`1fa33556c17303f51ba03c4eae11189d35e59b3f57faad0e69d9fd08bf9245ae`。旧S0 IDと同一視しない。

Bのconfiguration ID・round identity/law_binding、Aのphysical IDをそのまま保持する。exactκをfloatへ丸めてRを選び直さない。Bのfixed-grid validatorとD.from_bがq/R・piece配分・binary64 T/Δ・formula/K/prefixを照合し、coverageは元のsigned-time算術`T*w/q`とhexを保存する。このmenuのsigned_weightsは[1.]だが、signed pieceの形式・順序を消してevent総数だけに縮約しない。

## prefix別の物理context

| 対象 | 保存descriptorのλ_R | 保存descriptorのtotal scalar | deterministic fragments | randomized fragments |
|---|---:|---:|---|---|
| LD1 | 2.8629570945298686 | 0.4177175120626939 | [0]＋deterministic one-body | [1,…,9] |
| LD2 | 1.6030956277929231 | 0.2223759094107018 | [0,1]＋deterministic one-body | [2,…,9] |

数値はA指定MENUのcopyで、新しい計算・input復元による確認ではない。出典のcontext SHA/selectorを各coverage cellへ保持する。λ_RはB splitのexact比ともmetadata照合した。**実preparation/head/tail/basis/component hashは今回未取得。** 将来runtimeの元prepareでH hash・rank・prefix順・λ_R・scalar・preparationを照合するまで、coverageのpreparation statusはNOT_BUILTのままである。

必要なadapter修正は`from_b(..., expected_scalar=None)`だけ。指定時はcontext keyへ固定し、NativeRuntime.contextが元の`ham.constant + prep.extracted_identity_coefficient`と一致を確認してからheadを作る。未指定時の既存request構造と算術は維持する。scalar検査はmock descriptorだけで確認し、実H4 contextを構築していない。

## 追加したmetadata接続

[src/trottertracks/ld12_price_requests.py](../../../src/trottertracks/ld12_price_requests.py)は固定GitからMENU/configを読み、既存D.from_bを呼ぶだけの薄い接続である。

- `prepare(root, execution_source=...)`：全26 paired requestを一度metadata生成し、52軸の対応/不足を返す。
- `request_for(root, label, m, axes=..., execution_source=...)`：後から必要なround/axisだけの完全なD requestを生成する。axis変更はrequest IDに、物理/source変更はcontext keyにも反映する。
- 公開索引は完全なB configurationを重複保存せず、pinned commit/file/row selectorとID/keyを持つ。選択時は元canonical objectへ戻る。

input_sourceにはA menu、prefix別tail descriptor、元input出典を含める。context keyは候補/split・λ・expected scalar・law・compiler/control/wrapper・新lowering・D実行SOURCEを区別する。新ZZはCONTROLLED_ZZ_CX_CRZ_CX_V1、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17・measurement付きfull Hadamard X/Yである。

生成物のexecution_sourceは**技術metadata生成用tagと使用code blob**で、科学run SOURCEではない。将来の取得時は、承認された新execution SOURCEでrequest_forを呼び直す。過去索引を書換えたり古いrequest keyを再ラベルしない。fingerprint/cacheの実照合は回路が存在する取得時だけで、今回cache hitを認定していない。

## 既存価格と残る不足

LD1 m0/m6には既存P1/P2 development/pilotの参照があるが、新canonical IDへのSOURCE付きbridge・同law/profile確認と有限cohort/全law資格の区別が必要で、今回受入れは0。LD1全26 E[S]もSTRUCTURAL_ONLYの参照で、compiled Cや厳密boundではない。LD2へ旧LD1価格・E[S]・残差modelを移植しない。指定固定SOURCE内でLD2の一致compiled価格は特定していない。

全52軸でN・mean_C・価格区間・実compile時間/peak RSSはnull。actual sampler/ideal law差、rare、mean CI、長q/window tailは未資格。m12 metadataのevent数はLD1=33,423,360、LD2=10,481,664であり、巨大wordのcompose/hash/compile費用が障害になり得る。これは論理schedule規模で、wall/RAMの実測・比例外挿ではない。全round compileを実行計画にしていない。

## 技術確認とSTOP

新metadata/mock12項目＋既存D/B回帰26項目＝38 PASS。全52軸のidentity/law/時間/配分・prefix別λ/scalar/fragment予定、単独axis選択、SOURCE/key変更、LD取り違え、同event数の別R候補、law/lowering変更拒否、passive importを確認した。実科学入口を禁止したtestであり、Cのpreparation/構造kernel統合や実H4テストを重複実行していない。

過去SOURCE/input/raw/結果/KEEP/grant/ledger/marker/STOPと、進行中のA/B/Cの作業へ非介入。状態準備費を除く価格単位を保持し、N評価・shot配分・decoder・G比較を追加しない。

**次：A**がsignal、**B**が予備分析/shot評価から必要価格を指定し、**ユーザー**が候補/round/axis/path数・範囲を明示承認した後に限り、**D**が選択requestを実行する。**C**の構造/law成果は必要に応じSOURCE付きで受ける。今回はcommit/push・remote SHA/tree/主要成果物照合後mandatory STOP。
