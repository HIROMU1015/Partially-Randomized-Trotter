# 2026-11-14 D：DF head＋Pauli tailの回路と価格

D3d39f8c0基点、既存compiled価格研究branchを継続。P3/O1-M1のbuilderと価格条件を調査した。
P3はPauli D2 central rotation control、O1はevent全体generic control/seed7で別domain。
P3後続run04（e44e87a3）は32 path/armまで完了しており、その再取得は提案しない。

既存D2のtailをnative DF controlled_tailsへ接続する9人工作用素testがPASS。
full matrix/control phase、正負時間/K0/K2、scalar一回、measured XYを確認した。
compiler/価格取得と分子入力/preparation/samplingは起動していない。
実分子のPauli辞書/λ/signal/N/G、保存価格の再解析も0。

調査中にA75fb0のPauli技術仕様が公開されたため取り込んで読解した。
full-space eventとsector meanの区別、辞書付きidentity・元scalarの契約に合わせる。
新辞書/λ/R/全seedは未取得なので、LD1/m0 1 normal path XY最大2 compile案は未実行。
人工20 wrapper案や後続mean設計pilotは代替/別段階で、合算して自動実行しない。

C3a8b37のguardはDF2 routeのみでPauliが未対応。
同lowering文字列を強制せず、共通費用/compile条件と辞書別実装SOURCEを識別する拡張をCへ提案する。
Dでguard・signal・risk/Nを重複実装しない。
[REPORT](../d_df_head_pauli_tail_20261114/REPORT.md)／[A/C HANDOFF](../d_df_head_pauli_tail_20261114/BUILDERS_AND_HANDOFF.md)。
原本/科学SOURCE/STOPを保全し、公開・remote照合後mandatory STOP。
