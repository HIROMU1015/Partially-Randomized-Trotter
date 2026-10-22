# PF次数による優劣逆転：既存証拠の技術監査packet

2026-10-23 JST。[監査報告書](REPORT.md)／[全15条件](comparison_ABC_15.csv)／[75採用record](evidence_rows_75.csv)／[SOURCE索引](SOURCE_INDEX.md)。

保存順位の逆転候補はH2 ε=.005/.001/.0005、H4 ε=.005の4条件。他11条件は同方向。allowance政策差とnative/runtime未結合を保持し、科学的比較Cや因果機構を認定していない。

## コミット作業の範囲

利用者の「コミットして プッシュコマンドを出して」に基づき、完成済みの監査資料を専用ブランチへ固定する。push自体は実行しない。基点は固定commit ee160395e87ed6d46cd4993572457afd0aa18590。既存SOURCE・研究資料・結果・STOPの変更はない。

REPORT.md、verification.json、completion_status.jsonのcommit=0/push=0や「今回のcommit・pushは0」は、先行する監査作業が終了した時点の記録であり、その原文を保全した。今回の後続のコミット用格納はpublication_packet.jsonに別来歴として記録する。科学計算のmandatory STOPは継続する。

sources/は固定Git blobから取得した147ファイルの証拠コピー。acquisition_manifest.jsonで元full commit・元path・blob SHA1・SHA256へ追跡できる。上流SOURCE内の相対リンクは元repositoryの配置を前提とする。private specの全文・有効grant・seed一覧・科学arrayの新たなコピーは含めない。

検証はコピー前後のbyte/hash一致、147 blobのSHA、75/15/60行の件数、報告書リンクの存在を対象に行った。科学moduleのimport・科学計算・再会計・compile・レビュー開始は0。

次の作業：ユーザーが提示されたコマンドでこのブランチをpushする。Codexは科学的な追加作業へ進まずmandatory STOP。
