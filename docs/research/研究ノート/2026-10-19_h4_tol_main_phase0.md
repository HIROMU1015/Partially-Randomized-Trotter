# 2026-10-19 H4 tol-only rank10 主比較のPhase 0

利用者のCodex B指示と「そのまま進めて大丈夫」に基づき、技術準備・SOURCE実装・人工テスト・local commitまで実施し、科学実行は別承認とした。[統合報告](../track_a_h4_tol_main_phase0_20261019.md)。

F/FPのrank10保存入力を維持する。DF threshold1e-8、生成順、signed係数、cutoff0、weighted projection等はH6/H8の元SOURCEと照合した。H4 dense eighとH6/H8 eigsh、正規化・残差gateの実装差を明示し、状態を再生成しなかった。rank12 D/Pの数値は流用しない。

Dは全truncated prefix1〜9とfull10・q1/2/4/8/16・二次/四次で190候補、Aはrank構造規則で50候補の提案。FP同一条件の8＋64軸と保存統計を再利用し、新A49候補だけn8案とする。FPはn32 anchorのまま保持する。将来の新compile上限はD372軸、A784軸で、資源上限ではない。

純stdlibによるprimitive scheduleの構造的列挙で、負時間・merged PFを含むD365/A40 time pair、3 probeで1095/120と確認した。保存Hamiltonianや状態をdecodeせず、signal・shot計算を行わなかった。

SOURCE `84db3909639b34ec521004c3d8f7ef9b5e6ad851`、新32＋再利用42人工/回帰PASS、実DRAFT拒否8経路PASS。774保全レコード、SOURCE closure17、provider251、runtime34273をSHA監査した。AI delegation・共有環境変更・科学run・実統計解析・pushは0。

Pの候補・PF・n・SOURCE・seed・予算はPENDING_D_A_RESULT。次はユーザーが状態生成差とDの具体的scopeを確認して別実行承認する。A、S_STATS、Pは依存結果を受け入れてからそれぞれ承認対象にする。Phase 0完了時点でmandatory STOP。
