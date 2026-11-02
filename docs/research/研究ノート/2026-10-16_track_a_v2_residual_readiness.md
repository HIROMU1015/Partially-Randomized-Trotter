# 2026-10-16 V2：保存結果の残差解析に向けた再開準備

利用者がV1成果整理後のV2準備と公開を指示し、科学計算・数値再解析・compileを別承認待ちとした。専用branchをV1の63e5773b…から分離し、旧V2 dd695e65…と理論結果7e416705…は読み取り専用にした。旧V2未追跡scriptも保存。

[報告](../track_a_v2_residual_analysis_readiness_20261016.md)。832理論行・10,480保存接続行・262 scope entriesのschema/ID/hashを照合。1,212 primitive digest・144 lawを含む旧数値stageの重複実行は不要。uniform A/Bは実native A-prefix/B-tailと別で、mixed期待費用の追加は今回の最小計画へ含めない。

最初の数値stage案を保存RZ単価の条件付き較正/heldout残差だけに絞る。精度/factor反復・PF再利用を重複標本化せず、SOURCE/profile/variant/scope/system/axisを分離。critical来歴不明は不可評価、PR/PF平均signed相対残差差.05以上で直ちにSTOP。lower/upperを中点へ潰さず、数値境界・小標本・rank不足をPASSにしない。

新数値・回帰・費用比・ranking・primitive/law/DP/compile・有効grant0。人工検証と数値runner実装済みとは主張せず、実行前の必要項目を案へ固定した。公開文書・metadataだけをcommit/pushしてremote SHA確認後mandatory STOP。旧SOURCE/結果/raw/grant/ledger、V1、H5/H7を保全する。
