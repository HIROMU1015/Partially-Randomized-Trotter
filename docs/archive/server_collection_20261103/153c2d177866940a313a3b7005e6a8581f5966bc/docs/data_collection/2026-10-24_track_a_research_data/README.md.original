## 2026-10-27追加への入口

[境界拡張addendum](../2026-10-27_pr_boundary_addendum/REPORT.md)と[追加証拠CSV](../2026-10-27_pr_boundary_addendum/EVIDENCE_ADDENDUM.csv)を参照。H4/H6の固定境界比較、H8未成立会計、公開実行来歴を別区分で接続する。この2026-10-24索引の機械可読表・元CEM/ledger/paper manifest・旧STOPは再生成せず、正式採択を変更しない。


# Track A：PRリソース評価研究のデータ体系

2026-10-24 JST。保存結果を探し、科学条件を確かめて利用するための索引です。新科学計算・保存値の数値再解析・図表生成・原稿執筆・主張の再採択は行っていません。**mandatory STOP**。

## 目的別の入口

|探したいもの|入口|使い方|
|---|---|---|
|研究全体の結果・注意点|[REPORT](REPORT.md)|研究系列と採用範囲を読む|
|分子・DF・cohort・費用指標・SOURCE|[study_index.csv](study_index.csv)|study_idまたはcategoryで選択。condition_referenceで原manifestへ戻る|
|直接比較できる組・比較できない組|[comparability.csv](comparability.csv)|物理条件・q・公式・会計政策・cohortの差を確認|
|主張から証拠へ|[claim_usage.csv](claim_usage.csv)|既存CEMの同じclaim_idから固定結果を参照。新しいCEMではない|
|主表・Figure 2・CSV/JSON|[paper_data_index.csv](paper_data_index.csv)|asset_idから原データと列の意味へ。新しい図表は作らない|
|不足・失敗が論文化へ与える影響|[unresolved_issues.csv](unresolved_issues.csv)|impact_classとpaper_impactを読む。欠測を0にしない|
|固定commit・path・SHA|[source_catalog.json](source_catalog.json)|sourcesの参照IDから固定URLへ|
|field・unknown・n/Nの定義|[schema.json](schema.json)|未記録・非該当・今回未検証を区別|
|技術検査・保全・停止|[validation_summary.json](validation_summary.json)、[preservation_audit.json](preservation_audit.json)、[STOP.json](STOP.json)|技術PASSと科学的優位を混同しない|

## 三層の関係

1. 固定原結果：既存commit・SOURCE・CSV/JSON・CEM v3・ledger v5・paper manifestが正本です。保存値や採択を変更しません。
2. 共通メタデータ：study_indexは既存49 catalog行と最新9参照ビューを結びます。**58独立研究・58実験という意味ではありません。** 同じcohortを複数の環境索引が参照します。
3. 論文利用ビュー：claim_usage・paper_data_indexは掲載検討の入口です。Main/Supplementaryの分類は配置候補であり正式採択ではありません。

## 参照の解決

例えば `batch/primary_endpoints_60.csv` は、source_catalog.jsonの `sources` にある同名keyを参照します。`main_table#comparison_id=H4_DF10_P_1` は固定主表の該当行です。

`three/references.csv#ref_id=R0135` は、固定3環境catalogのR0135を参照します。3,538参照をこのbundleへ再複製していません。`commit:path#/series/0` は原manifestのJSON pointerです。studyのfamily profileは広い分類情報で、個別入力・候補の条件を置換しません。

`unknown` はこの限定取得範囲で記録・取得できない値、`not_applicable` は非該当、`not_verified` は宣言や参照はあるが今回その事実を確立していない値です。SOURCEの保存 `[]` は元索引に記録されたSOURCEがないというmetadataで、科学処理が存在しなかった証明ではありません。

- `n`：古典的なPR回路費用trajectory標本数。cos/sinはpairedであり2標本にはしません。
- `N`：保存解析式による量子測定shot数。実量子測定を実施した回数ではありません。
- `C`：元fieldに指定された単価。compiled RZ命令数、理論値、RZ depth、CX等を混ぜません。
- `G`：元保存の軸別N×C会計。ここでは加算・乗算・比・順位・SE/区間を新しく計算しません。

原結果の取得環境・保存環境は、元計算を実行したhostの証明ではありません。result_commitは参照可能な固定snapshotを示すことがあり、元科学SOURCE・元result commitはcondition_reference、CEM原記述、元REPORTに従います。

## 技術検査の再実行

repository rootで以下を実行できます。固定metadata・参照・scopeの検査だけです。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/index_track_a_research_data.py --check
```

`--build` は今回のmetadata bundleだけを再生成します。原SOURCE、結果、grant、ledger、marker、旧STOP、主表、Figure 2へは書きません。人向けREPORTやSTOPを科学実行許可として使うことはできません。第2研究C0の成果や承認は取り込みません。
